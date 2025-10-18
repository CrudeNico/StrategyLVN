#!/usr/bin/env python3
"""Render the flipped-direction equity curve as HTML and PNG."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Backtest.Logic.trading_rules import STARTING_BALANCE_EUR

EQUITY_FILE = PROJECT_ROOT / "Optimization" / "flip_equity_curve.csv"
OUTPUT_HTML = PROJECT_ROOT / "Optimization" / "flip_equity_curve.html"
OUTPUT_PNG = PROJECT_ROOT / "Optimization" / "flip_equity_curve.png"


def load_equity() -> pd.DataFrame:
    if not EQUITY_FILE.exists():
        raise FileNotFoundError(
            f"Flipped equity data not found at {EQUITY_FILE}. "
            "Run run_flip_analysis.py first."
        )
    equity = pd.read_csv(EQUITY_FILE, parse_dates=["timestamp"])
    equity.sort_values("timestamp", inplace=True, ignore_index=True)

    if equity.empty:
        raise ValueError("Flipped equity data is empty.")

    first_point = equity.iloc[0]
    if first_point["balance"] != STARTING_BALANCE_EUR:
        equity = pd.concat(
            [
                pd.DataFrame(
                    {
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
        title="Flipped-Direction Equity Curve",
        labels={"timestamp": "Time", "balance": "Account Balance (€)"},
    )
    fig.update_layout(template="plotly_white")
    fig.update_traces(line=dict(color="#c62828", width=2.2))
    fig.update_yaxes(ticksuffix=" €", showgrid=True)
    return fig


def main() -> None:
    equity = load_equity()
    fig = build_figure(equity)
    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn", full_html=True)
    fig.write_image(OUTPUT_PNG, width=1280, height=640, scale=1.5)
    print(f"Flipped equity curve exported to {OUTPUT_HTML} and {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
