#!/usr/bin/env python3
"""Render the backtest equity curve and export to PNG/HTML."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Backtest.Logic.trading_rules import STARTING_BALANCE_EUR

EQUITY_FILE = PROJECT_ROOT / "Backtest" / "Results" / "equity_curve.csv"
OUTPUT_HTML = Path(__file__).resolve().parent / "equity_curve.html"
OUTPUT_PNG = Path(__file__).resolve().parent / "equity_curve.png"


def load_equity() -> pd.DataFrame:
    if not EQUITY_FILE.exists():
        raise FileNotFoundError(
            f"Equity data not found at {EQUITY_FILE}. Run run_backtest.py first."
        )
    equity = pd.read_csv(EQUITY_FILE, parse_dates=["timestamp"])
    equity.sort_values("timestamp", inplace=True, ignore_index=True)

    if equity.empty or equity.iloc[0]["timestamp"] != equity["timestamp"].min():
        equity = pd.concat(
            [
                pd.DataFrame(
                    {
                        "trade_id": [0],
                        "timestamp": [equity["timestamp"].min()],
                        "balance": [STARTING_BALANCE_EUR],
                    }
                ),
                equity,
            ],
            ignore_index=True,
        )
    return equity


def build_figure(equity: pd.DataFrame):
    fig = px.line(
        equity,
        x="timestamp",
        y="balance",
        title="StrategyLVN Equity Curve",
        labels={"timestamp": "Time", "balance": "Account Balance (€)"},
    )
    fig.update_layout(template="plotly_white")
    fig.update_traces(line=dict(color="#1565c0", width=2.2))
    fig.update_yaxes(ticksuffix=" €", showgrid=True)
    return fig


def main() -> None:
    equity = load_equity()
    fig = build_figure(equity)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn", full_html=True)
    fig.write_image(OUTPUT_PNG, width=1280, height=640, scale=1.5)
    print(f"Equity curve exported to {OUTPUT_HTML} and {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
