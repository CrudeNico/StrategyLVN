#!/usr/bin/env python3
"""Construct initial trade plan based on LVN transitions and swing structure."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Backtest.Logic.trading_rules import (
    RiskManager,
    annotate_with_windows,
    load_price_data,
)
RESULTS_DIR = PROJECT_ROOT / "Findings" / "Results"
LVN_FILE = RESULTS_DIR / "zigzag_lvn_summary.csv"
SWINGS_FILE = RESULTS_DIR / "zigzag_swings.csv"
OUTPUT_FILE = PROJECT_ROOT / "Backtest" / "Results" / "trade_plan.csv"


@dataclass
class TradeScenario:
    transition_type: str
    direction: str
    from_leg_id: int
    to_leg_id: int
    target_leg_id: Optional[int]
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    stop_leg_id: int
    target_time: pd.Timestamp
    target_price: float
    units: float
    risk_eur: float
    reward_eur: float
    lvn_price_center: float
    lvn_price_min: float
    lvn_price_max: float
    time_delta_minutes: float


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    price_df = annotate_with_windows(load_price_data())
    if not LVN_FILE.exists():
        raise FileNotFoundError(
            f"LVN summary not found at {LVN_FILE}. Run extract_lvn.py first."
        )
    lvn_df = pd.read_csv(
        LVN_FILE,
        parse_dates=["from_datetime", "to_datetime"],
    )
    swings_df = pd.read_csv(
        SWINGS_FILE,
        parse_dates=["datetime"],
    ).sort_values("leg_id", ignore_index=True)
    return price_df, lvn_df, swings_df


def find_next_pivot(
    swings: pd.DataFrame, current_leg: int, target_label: str
) -> Optional[pd.Series]:
    current_idx = swings.index[swings["leg_id"] == current_leg]
    if current_idx.empty:
        return None
    subset = swings.loc[current_idx[0] + 1 :]
    target = subset[subset["structure_label"] == target_label]
    if target.empty:
        return None
    return target.iloc[0]


def find_entry(
    price_df: pd.DataFrame,
    lvn_min: float,
    lvn_max: float,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> Optional[tuple[pd.Timestamp, float]]:
    """Find the first LVN retest after the leg completes."""
    window = price_df.loc[start_time:end_time]
    if window.empty or len(window) < 2:
        return None

    zone_min = min(lvn_min, lvn_max)
    zone_max = max(lvn_min, lvn_max)

    for idx in range(len(window) - 1):
        bar = window.iloc[idx]
        if bar.name <= start_time:
            continue

        next_bar = window.iloc[idx + 1]
        if not bar["can_open_trades"] or not next_bar["can_open_trades"]:
            continue

        touches_zone = bar["low"] <= zone_max and bar["high"] >= zone_min
        if not touches_zone:
            continue

        entry_time = next_bar.name
        entry_price = next_bar["open"]
        return entry_time, float(entry_price)

    return None


def build_trade_scenarios() -> list[TradeScenario]:
    price_df, lvn_df, swings = load_inputs()
    risk_manager = RiskManager()
    risk_eur = risk_manager.risk_amount()

    scenarios: list[TradeScenario] = []

    for row in lvn_df.itertuples():
        direction = "long" if row.transition_type == "HL_to_HH" else "short"
        stop_leg_id = row.from_leg_id
        stop_price = float(row.from_price)

        target_label = "HH" if direction == "long" else "LL"
        next_pivot = find_next_pivot(swings, row.to_leg_id, target_label)
        if next_pivot is None:
            continue

        target_time = next_pivot["datetime"]
        target_idx = price_df.index.get_indexer([target_time])
        if target_idx.size == 0 or target_idx[0] == -1:
            continue
        target_loc = target_idx[0]
        if target_loc + 1 >= len(price_df):
            continue
        target_exec_time = price_df.index[target_loc + 1]
        target_price = float(price_df.iloc[target_loc + 1]["open"])

        entry_window_start = row.to_datetime
        entry_window_end = target_time
        entry = find_entry(
            price_df,
            lvn_min=float(row.lvn_price_min),
            lvn_max=float(row.lvn_price_max),
            start_time=entry_window_start,
            end_time=entry_window_end,
        )
        if entry is None:
            continue
        entry_time, entry_price = entry

        if entry_time >= target_exec_time:
            continue

        if direction == "long":
            stop_distance = entry_price - stop_price
            reward = target_price - entry_price
        else:
            stop_distance = stop_price - entry_price
            reward = entry_price - target_price

        if stop_distance <= 0:
            continue

        units = risk_eur / stop_distance

        scenarios.append(
            TradeScenario(
                transition_type=row.transition_type,
                direction=direction,
                from_leg_id=row.from_leg_id,
                to_leg_id=row.to_leg_id,
                target_leg_id=int(next_pivot["leg_id"]),
                entry_time=entry_time,
                entry_price=float(entry_price),
                stop_price=stop_price,
                stop_leg_id=stop_leg_id,
                target_time=target_exec_time,
                target_price=target_price,
                units=float(units),
                risk_eur=float(risk_eur),
                reward_eur=float(reward * units),
                lvn_price_center=float(row.lvn_price_center),
                lvn_price_min=float(row.lvn_price_min),
                lvn_price_max=float(row.lvn_price_max),
                time_delta_minutes=float(row.time_delta_minutes),
            )
        )

    return scenarios


def scenarios_to_dataframe(scenarios: list[TradeScenario]) -> pd.DataFrame:
    if not scenarios:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "transition_type": s.transition_type,
                "direction": s.direction,
                "from_leg_id": s.from_leg_id,
                "to_leg_id": s.to_leg_id,
                "target_leg_id": s.target_leg_id,
                "entry_time": s.entry_time,
                "entry_price": s.entry_price,
                "stop_price": s.stop_price,
                "stop_leg_id": s.stop_leg_id,
                "target_time": s.target_time,
                "target_price": s.target_price,
                "units": s.units,
                "risk_eur": s.risk_eur,
                "reward_eur": s.reward_eur,
                "lvn_price_center": s.lvn_price_center,
                "lvn_price_min": s.lvn_price_min,
                "lvn_price_max": s.lvn_price_max,
                "time_delta_minutes": s.time_delta_minutes,
            }
            for s in scenarios
        ]
    )


def main() -> None:
    scenarios = build_trade_scenarios()
    df = scenarios_to_dataframe(scenarios)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Generated {len(df)} trade scenarios → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
