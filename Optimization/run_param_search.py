#!/usr/bin/env python3
"""Parameter sweep for ZigZag threshold and volume profile bins."""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Backtest.Logic.trading_rules import (
    RiskManager,
    annotate_with_windows,
    load_price_data,
)
from Backtest.Logic.run_backtest import build_base_trades, size_trades
from Backtest.Logic.construct_trade_plan import (
    TradeScenario,
    find_next_pivot,
    find_entry,
    scenarios_to_dataframe,
)
from Findings.Logic.compute_zigzag import (
    build_transition_pairs,
    build_zigzag_indicator,
    label_structure,
)
from Findings.Logic.compute_volume_profile import compute_volume_profile
from Findings.Logic.extract_lvn import build_lvn_records


RESULTS_DIR = PROJECT_ROOT / "Optimization"
SUMMARY_FILE = RESULTS_DIR / "param_search_summary.csv"
BEST_TRADE_RESULTS_FILE = RESULTS_DIR / "best_trade_results.csv"
BEST_EQUITY_FILE = RESULTS_DIR / "best_equity_curve.csv"

# Parameter grid (expand or adjust as needed)
ZIGZAG_PCTS = [0.0005]
VOLUME_BINS = [140]
EMA_SPANS = [10, 15, 20, 25, 30, 40, 50]


def build_trade_scenarios_for_params(
    price_df: pd.DataFrame,
    swings: pd.DataFrame,
    lvn_df: pd.DataFrame,
    ema_series: pd.Series,
) -> List[TradeScenario]:
    risk_eur = RiskManager().risk_amount()
    lvn_df = lvn_df.sort_values("from_datetime").reset_index(drop=True)
    last_transition_type: Optional[str] = None
    scenarios: List[TradeScenario] = []

    for row in lvn_df.itertuples():
        direction = "long" if row.transition_type == "HL_to_HH" else "short"
        stop_price = float(row.from_price)
        stop_leg_id = int(row.from_leg_id)

        required_type = row.transition_type
        if last_transition_type != required_type:
            last_transition_type = row.transition_type
            continue

        target_label = "HH" if direction == "long" else "LL"
        next_pivot = find_next_pivot(swings, row.to_leg_id, target_label)
        if next_pivot is None:
            continue

        target_time = next_pivot["datetime"]
        if target_time not in price_df.index:
            continue
        target_loc = price_df.index.get_loc(target_time)
        if isinstance(target_loc, slice):
            target_loc = target_loc.start
        if target_loc is None or target_loc == -1 or target_loc + 1 >= len(price_df):
            continue
        target_exec_time = price_df.index[target_loc + 1]
        target_price = float(price_df.iloc[target_loc + 1]["open"])

        entry = find_entry(
            price_df,
            lvn_min=float(row.lvn_price_min),
            lvn_max=float(row.lvn_price_max),
            start_time=row.to_datetime,
            end_time=target_time,
        )
        if entry is None:
            continue
        entry_time, entry_price = entry
        try:
            ema_value = float(ema_series.loc[entry_time])
        except KeyError:
            continue
        if np.isnan(ema_value):
            continue

        if direction == "long" and ema_value > entry_price:
            continue
        if direction == "short" and ema_value < entry_price:
            continue
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
                from_leg_id=int(row.from_leg_id),
                to_leg_id=int(row.to_leg_id),
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

        last_transition_type = row.transition_type

    return scenarios


def compute_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    total_trades = len(trades_df)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "win_rate": np.nan,
            "total_pnl": 0.0,
            "final_balance": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
        }

    win_rate = (trades_df["pnl_eur"] > 0).mean()
    total_pnl = trades_df["pnl_eur"].sum()
    final_balance = trades_df["balance_after_trade"].iloc[-1]

    balances = equity_df["balance"].astype(float).to_numpy()
    cumulative_max = np.maximum.accumulate(balances)
    drawdowns = (balances - cumulative_max) / cumulative_max
    max_drawdown = drawdowns.min() if cumulative_max.size > 0 else np.nan

    returns = trades_df["pnl_eur"] / trades_df["balance_before_trade"]
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = (
        returns.mean() / returns.std() * np.sqrt(len(returns))
        if len(returns) > 1 and returns.std() > 0
        else np.nan
    )

    return {
        "total_trades": int(total_trades),
        "win_rate": float(win_rate),
        "total_pnl": float(total_pnl),
        "final_balance": float(final_balance),
        "max_drawdown": float(max_drawdown),
        "sharpe": float(sharpe),
    }


def main() -> None:
    raw_prices = load_price_data()
    price_df = annotate_with_windows(raw_prices.copy())
    ema_series = raw_prices["close"].ewm(span=20, adjust=False, min_periods=20).mean()

    zigzag_indicator = build_zigzag_indicator()

    scenarios_summary = []
    best_result = None

    for zigzag_pct, vp_bins, ema_span in product(ZIGZAG_PCTS, VOLUME_BINS, EMA_SPANS):
        zigzag = zigzag_indicator.run(raw_prices["close"], pct=zigzag_pct)
        swings = label_structure(raw_prices, zigzag, zigzag_pct)

        hl_to_hh = build_transition_pairs(swings, "HL", "HH")
        lh_to_ll = build_transition_pairs(swings, "LH", "LL")
        if hl_to_hh.empty and lh_to_ll.empty:
            continue

        profile = compute_volume_profile(raw_prices, bins=vp_bins)
        if profile.empty:
            continue

        lvn_records = []
        lvn_records.extend(build_lvn_records(hl_to_hh, profile, "HL_to_HH"))
        lvn_records.extend(build_lvn_records(lh_to_ll, profile, "LH_to_LL"))
        lvn_df = pd.DataFrame(lvn_records)
        if lvn_df.empty:
            continue

        ema_series = raw_prices["close"].ewm(
            span=ema_span, adjust=False, min_periods=ema_span
        ).mean()

        trade_scenarios = build_trade_scenarios_for_params(
            price_df, swings, lvn_df, ema_series
        )
        if not trade_scenarios:
            continue
        plan_df = scenarios_to_dataframe(trade_scenarios)

        base_trades = build_base_trades(price_df, plan_df)
        trades_df, equity_df = size_trades(base_trades)
        metrics = compute_metrics(trades_df, equity_df)
        metrics.update(
            {
                "zigzag_pct": zigzag_pct,
                "volume_bins": vp_bins,
                "ema_span": ema_span,
            }
        )
        scenarios_summary.append(metrics)

        if best_result is None or (
            np.nan_to_num(metrics["sharpe"], nan=-np.inf)
            > np.nan_to_num(best_result["metrics"]["sharpe"], nan=-np.inf)
        ):
            best_result = {
                "metrics": metrics,
                "trades_df": trades_df.copy(),
                "equity_df": equity_df.copy(),
            }

        print(
            f"Evaluated pct={zigzag_pct:.4f}, bins={vp_bins}, ema={ema_span} → "
            f"trades={metrics['total_trades']} win_rate={metrics['win_rate']:.2%} "
            f"PnL={metrics['total_pnl']:.2f}€ Sharpe={metrics['sharpe']:.2f}"
        )

    results_df = pd.DataFrame(scenarios_summary)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.sort_values(
        ["sharpe", "final_balance"], ascending=[False, False], inplace=True
    )
    results_df.to_csv(SUMMARY_FILE, index=False)

    if best_result is not None:
        best_result["trades_df"].to_csv(BEST_TRADE_RESULTS_FILE, index=False)
        best_result["equity_df"].to_csv(BEST_EQUITY_FILE, index=False)
        print("\nTop scenario:")
        for key, value in best_result["metrics"].items():
            print(f"  {key}: {value}")
    else:
        print("No viable scenarios evaluated.")

    print(f"\nSaved summary to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
