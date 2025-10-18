#!/usr/bin/env python3
"""Run LVN-based backtest and save trade/equity results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import heapq
import sys
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Backtest.Logic.trading_rules import (
    RISK_PER_TRADE,
    STARTING_BALANCE_EUR,
    annotate_with_windows,
    load_price_data,
)

TRADE_PLAN_FILE = PROJECT_ROOT / "Backtest" / "Results" / "trade_plan.csv"
TRADE_RESULTS_FILE = PROJECT_ROOT / "Backtest" / "Results" / "trade_results.csv"
EQUITY_FILE = PROJECT_ROOT / "Backtest" / "Results" / "equity_curve.csv"


@dataclass
class BaseTrade:
    trade_id: int
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
    stop_distance: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    r_multiple: float
    holding_minutes: float
    lvn_price_center: float
    lvn_price_min: float
    lvn_price_max: float


def load_trade_plan() -> pd.DataFrame:
    if not TRADE_PLAN_FILE.exists():
        raise FileNotFoundError(
            f"Trade plan not found at {TRADE_PLAN_FILE}. Run construct_trade_plan.py first."
        )
    df = pd.read_csv(
        TRADE_PLAN_FILE,
        parse_dates=["entry_time", "target_time"],
    )
    df.sort_values("entry_time", inplace=True, ignore_index=True)
    return df


def simulate_exit(
    row: pd.Series,
    price_df: pd.DataFrame,
) -> tuple[pd.Timestamp, float, str]:
    entry_time = row["entry_time"]
    target_time = row["target_time"]
    direction = 1 if row["direction"] == "long" else -1
    stop_price = float(row["stop_price"])
    target_price = float(row["target_price"])

    if entry_time not in price_df.index:
        raise ValueError(f"Entry timestamp {entry_time} not found in price data.")

    subset = price_df.loc[entry_time:]
    if subset.empty:
        raise ValueError("Price subset after entry is empty.")

    exit_time = None
    exit_price = None
    exit_reason = "data_end"

    for idx, (timestamp, bar) in enumerate(subset.iterrows()):
        bar_low = bar["low"]
        bar_high = bar["high"]

        # stop logic
        stop_hit = (
            bar_low <= stop_price if direction == 1 else bar_high >= stop_price
        )

        if idx == 0:
            if stop_hit:
                exit_time = timestamp
                exit_price = stop_price
                exit_reason = "stop"
                break
            if timestamp == target_time:
                exit_time = timestamp
                exit_price = target_price
                exit_reason = "target"
                break
            continue

        if bar["force_exit"]:
            exit_time = timestamp
            exit_price = bar["open"]
            exit_reason = "friday_exit"
            break

        if stop_hit:
            exit_time = timestamp
            exit_price = stop_price
            exit_reason = "stop"
            break

        if timestamp == target_time:
            exit_time = timestamp
            exit_price = target_price
            exit_reason = "target"
            break

    if exit_time is None:
        last_timestamp = subset.index[-1]
        exit_time = last_timestamp
        exit_price = float(subset.iloc[-1]["close"])
        exit_reason = "data_end"

    return exit_time, float(exit_price), exit_reason


def build_base_trades(price_df: pd.DataFrame, plan_df: pd.DataFrame) -> List[BaseTrade]:
    trades: List[BaseTrade] = []
    trade_id = 1

    for _, row in plan_df.iterrows():
        entry_time = row["entry_time"]
        stop_price = float(row["stop_price"])
        entry_price = float(row["entry_price"])
        stop_distance = abs(entry_price - stop_price)
        if stop_distance <= 0:
            continue

        try:
            exit_time, exit_price, exit_reason = simulate_exit(row, price_df)
        except ValueError:
            continue

        direction = 1 if row["direction"] == "long" else -1
        r_multiple = direction * (exit_price - entry_price) / stop_distance
        holding_minutes = float((exit_time - entry_time).total_seconds() / 60.0)

        trades.append(
            BaseTrade(
                trade_id=trade_id,
                transition_type=row["transition_type"],
                direction=row["direction"],
                from_leg_id=int(row["from_leg_id"]),
                to_leg_id=int(row["to_leg_id"]),
                target_leg_id=int(row["target_leg_id"])
                if not pd.isna(row["target_leg_id"])
                else None,
                entry_time=entry_time,
                entry_price=entry_price,
                stop_price=stop_price,
                stop_leg_id=int(row["stop_leg_id"])
                if "stop_leg_id" in row else int(row["from_leg_id"]),
                target_time=row["target_time"],
                target_price=float(row["target_price"]),
                stop_distance=float(stop_distance),
                exit_time=exit_time,
                exit_price=exit_price,
                exit_reason=exit_reason,
                r_multiple=float(r_multiple),
                holding_minutes=holding_minutes,
                lvn_price_center=float(row["lvn_price_center"]),
                lvn_price_min=float(row["lvn_price_min"]),
                lvn_price_max=float(row["lvn_price_max"]),
            )
        )
        trade_id += 1

    return trades


def size_trades(trades: Iterable[BaseTrade]) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades_sorted = sorted(trades, key=lambda t: t.entry_time)

    balance = STARTING_BALANCE_EUR
    pending_heap: list[tuple[pd.Timestamp, int, dict]] = []
    realized_results: list[dict] = []
    equity_rows: list[dict] = []
    unique_idx = 0

    for trade in trades_sorted:
        # Realize trades that have exited before this entry
        while pending_heap and pending_heap[0][0] <= trade.entry_time:
            _, _, pending_trade = heapq.heappop(pending_heap)
            balance += pending_trade["pnl_eur"]
            pending_trade["balance_after_trade"] = balance
            realized_results.append(pending_trade)
            equity_rows.append(
                {
                    "trade_id": pending_trade["trade_id"],
                    "timestamp": pending_trade["exit_time"],
                    "balance": balance,
                }
            )

        risk_eur = balance * RISK_PER_TRADE
        units = risk_eur / trade.stop_distance
        pnl_eur = risk_eur * trade.r_multiple

        trade_dict = {
            "trade_id": trade.trade_id,
            "transition_type": trade.transition_type,
            "direction": trade.direction,
            "from_leg_id": trade.from_leg_id,
            "to_leg_id": trade.to_leg_id,
            "target_leg_id": trade.target_leg_id,
            "entry_time": trade.entry_time,
            "entry_price": trade.entry_price,
            "stop_price": trade.stop_price,
            "stop_distance": trade.stop_distance,
            "target_time": trade.target_time,
            "target_price": trade.target_price,
            "exit_time": trade.exit_time,
            "exit_price": trade.exit_price,
            "exit_reason": trade.exit_reason,
            "holding_minutes": trade.holding_minutes,
            "lvn_price_center": trade.lvn_price_center,
            "lvn_price_min": trade.lvn_price_min,
            "lvn_price_max": trade.lvn_price_max,
            "risk_eur": risk_eur,
            "units": units,
            "pnl_eur": pnl_eur,
            "balance_before_trade": balance,
            "balance_after_trade": np.nan,
        }

        heapq.heappush(pending_heap, (trade.exit_time, unique_idx, trade_dict))
        unique_idx += 1

    # Realize remaining trades
    while pending_heap:
        _, _, pending_trade = heapq.heappop(pending_heap)
        balance += pending_trade["pnl_eur"]
        pending_trade["balance_after_trade"] = balance
        realized_results.append(pending_trade)
        equity_rows.append(
            {
                "trade_id": pending_trade["trade_id"],
                "timestamp": pending_trade["exit_time"],
                "balance": balance,
            }
        )

    realized_results.sort(key=lambda r: r["exit_time"])
    equity_rows.sort(key=lambda r: r["timestamp"])

    trades_df = pd.DataFrame(realized_results)
    equity_df = pd.DataFrame(equity_rows)
    return trades_df, equity_df


def run_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    price_df = annotate_with_windows(load_price_data())
    plan_df = load_trade_plan()
    base_trades = build_base_trades(price_df, plan_df)
    trades_df, equity_df = size_trades(base_trades)
    return trades_df, equity_df


def main() -> None:
    trades_df, equity_df = run_backtest()
    TRADE_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(TRADE_RESULTS_FILE, index=False)
    equity_df.to_csv(EQUITY_FILE, index=False)
    print(
        f"Backtest complete: {len(trades_df)} trades saved to {TRADE_RESULTS_FILE}\n"
        f"Equity curve saved to {EQUITY_FILE}"
    )


if __name__ == "__main__":
    main()
