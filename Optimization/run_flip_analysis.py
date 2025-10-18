#!/usr/bin/env python3
"""Quick optimization: evaluate flipping trade directions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Backtest.Logic.trading_rules import STARTING_BALANCE_EUR

TRADE_RESULTS_FILE = PROJECT_ROOT / "Backtest" / "Results" / "trade_results.csv"
SUMMARY_FILE = PROJECT_ROOT / "Optimization" / "flip_scenario_summary.csv"
FLIPPED_RESULTS_FILE = PROJECT_ROOT / "Optimization" / "flip_trade_results.csv"
FLIPPED_EQUITY_FILE = PROJECT_ROOT / "Optimization" / "flip_equity_curve.csv"


@dataclass
class ScenarioMetrics:
    name: str
    total_trades: int
    win_rate: float
    total_pnl: float
    final_balance: float


def load_trade_results() -> pd.DataFrame:
    if not TRADE_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Trade results not found at {TRADE_RESULTS_FILE}. Run run_backtest.py first."
        )
    trades = pd.read_csv(
        TRADE_RESULTS_FILE,
        parse_dates=["entry_time", "exit_time"],
    )
    trades.sort_values("exit_time", inplace=True, ignore_index=True)
    return trades


def compute_metrics(name: str, trades: pd.DataFrame) -> ScenarioMetrics:
    total_trades = len(trades)
    win_rate = (
        float((trades["pnl_eur"] > 0).sum()) / total_trades if total_trades else 0.0
    )
    total_pnl = float(trades["pnl_eur"].sum())
    final_balance = STARTING_BALANCE_EUR + total_pnl
    return ScenarioMetrics(name, total_trades, win_rate, total_pnl, final_balance)


def build_flipped_results(trades: pd.DataFrame) -> pd.DataFrame:
    flipped = trades.copy()
    flipped["direction"] = flipped["direction"].map(
        {"long": "short", "short": "long"}
    )
    flipped["pnl_eur"] = -flipped["pnl_eur"]
    flipped["risk_eur"] = trades["risk_eur"]
    flipped["balance_before_trade"] = STARTING_BALANCE_EUR

    running_balance = STARTING_BALANCE_EUR
    balances = []
    for pnl in flipped["pnl_eur"]:
        running_balance += pnl
        balances.append(running_balance)
    flipped["balance_after_trade"] = balances
    return flipped


def main() -> None:
    trades = load_trade_results()
    base_metrics = compute_metrics("Base Strategy", trades)

    flipped_trades = build_flipped_results(trades)
    flipped_metrics = compute_metrics("Flipped Direction", flipped_trades)

    FLIPPED_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    flipped_trades.to_csv(FLIPPED_RESULTS_FILE, index=False)

    flipped_equity = flipped_trades[["exit_time", "balance_after_trade"]].rename(
        columns={"exit_time": "timestamp", "balance_after_trade": "balance"}
    )
    flipped_equity.to_csv(FLIPPED_EQUITY_FILE, index=False)

    summary = pd.DataFrame(
        [
            {
                "scenario": base_metrics.name,
                "total_trades": base_metrics.total_trades,
                "win_rate": base_metrics.win_rate,
                "total_pnl": base_metrics.total_pnl,
                "final_balance": base_metrics.final_balance,
            },
            {
                "scenario": flipped_metrics.name,
                "total_trades": flipped_metrics.total_trades,
                "win_rate": flipped_metrics.win_rate,
                "total_pnl": flipped_metrics.total_pnl,
                "final_balance": flipped_metrics.final_balance,
            },
        ]
    )
    summary.to_csv(SUMMARY_FILE, index=False)

    print(summary.to_string(index=False))
    print(f"\nSaved flipped scenario results to {FLIPPED_RESULTS_FILE}")
    print(f"Saved flipped equity curve to {FLIPPED_EQUITY_FILE}")
    print(f"Summary written to {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
