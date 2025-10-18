#!/usr/bin/env python3
"""Plot candlesticks with LVN trades overlaid."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from Findings.plot.plot_candles import (  # type: ignore
    EMA20_FILE,
    HL_TO_HH_FILE,
    LH_TO_LL_FILE,
    LVN_FILE,
    load_data,
    load_lvn_summary,
    load_swings,
    load_transitions,
    load_ema,
    build_figure,
)

TRADE_RESULTS_FILE = PROJECT_ROOT / "Backtest" / "Results" / "trade_results.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "backtest_trades.html"


def load_trade_results() -> pd.DataFrame:
    if not TRADE_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Backtest trade results not found at {TRADE_RESULTS_FILE}. "
            "Run run_backtest.py first."
        )
    trades = pd.read_csv(
        TRADE_RESULTS_FILE,
        parse_dates=["entry_time", "target_time", "exit_time"],
    )
    trades.sort_values("entry_time", inplace=True, ignore_index=True)
    return trades


def add_trade_markers(fig: go.Figure, trades: pd.DataFrame) -> None:
    if trades.empty:
        return

    long_trades = trades[trades["direction"] == "long"]
    short_trades = trades[trades["direction"] == "short"]

    if not long_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=long_trades["entry_time"],
                y=long_trades["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=9, color="#1b5e20"),
                name="Long Entry",
                hovertemplate=(
                    "Long #%{customdata[0]}<br>"
                    "Entry %{x}<br>"
                    "Price %{y:.2f}<br>"
                    "Risk €%{customdata[1]:.2f}<extra></extra>"
                ),
                customdata=np.stack(
                    [long_trades["trade_id"], long_trades["risk_eur"]], axis=-1
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=long_trades["exit_time"],
                y=long_trades["exit_price"],
                mode="markers",
                marker=dict(symbol="circle", size=9, color="#66bb6a", line=dict(color="#1b5e20", width=1)),
                name="Long Exit",
                hovertemplate=(
                    "Long #%{customdata[0]} exit (%{customdata[1]})<br>"
                    "Exit %{x}<br>"
                    "Price %{y:.2f}<br>"
                    "PnL €%{customdata[2]:+.2f}<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        long_trades["trade_id"],
                        long_trades["exit_reason"],
                        long_trades["pnl_eur"],
                    ],
                    axis=-1,
                ),
            )
        )

    if not short_trades.empty:
        fig.add_trace(
            go.Scatter(
                x=short_trades["entry_time"],
                y=short_trades["entry_price"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=9, color="#b71c1c"),
                name="Short Entry",
                hovertemplate=(
                    "Short #%{customdata[0]}<br>"
                    "Entry %{x}<br>"
                    "Price %{y:.2f}<br>"
                    "Risk €%{customdata[1]:.2f}<extra></extra>"
                ),
                customdata=np.stack(
                    [short_trades["trade_id"], short_trades["risk_eur"]], axis=-1
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=short_trades["exit_time"],
                y=short_trades["exit_price"],
                mode="markers",
                marker=dict(symbol="x", size=9, color="#ef5350", line=dict(color="#b71c1c", width=1)),
                name="Short Exit",
                hovertemplate=(
                    "Short #%{customdata[0]} exit (%{customdata[1]})<br>"
                    "Exit %{x}<br>"
                    "Price %{y:.2f}<br>"
                    "PnL €%{customdata[2]:+.2f}<extra></extra>"
                ),
                customdata=np.stack(
                    [
                        short_trades["trade_id"],
                        short_trades["exit_reason"],
                        short_trades["pnl_eur"],
                    ],
                    axis=-1,
                ),
            )
        )

    # Entry → exit connectors
    for direction, color in (("long", "#1b5e20"), ("short", "#b71c1c")):
        subset = trades[trades["direction"] == direction]
        if subset.empty:
            continue
        segment_x = []
        segment_y = []
        custom = []
        for row in subset.itertuples():
            segment_x.extend([row.entry_time, row.exit_time, None])
            segment_y.extend([row.entry_price, row.exit_price, None])
            custom.extend(
                [
                    [row.trade_id, row.exit_reason, row.pnl_eur],
                    [row.trade_id, row.exit_reason, row.pnl_eur],
                    [None, None, None],
                ]
            )
        fig.add_trace(
            go.Scatter(
                x=segment_x,
                y=segment_y,
                mode="lines",
                line=dict(color=color, width=1.2, dash="solid"),
                name=f"{direction.title()} Trades",
                hovertemplate=(
                    "Trade #%{customdata[0]} (%{customdata[1]})<br>"
                    "PnL €%{customdata[2]:+.2f}<extra></extra>"
                ),
                customdata=custom,
                showlegend=False,
            )
        )


def main() -> None:
    df = load_data()
    swing_df = load_swings()
    hl_to_hh = load_transitions(HL_TO_HH_FILE)
    lh_to_ll = load_transitions(LH_TO_LL_FILE)
    lvn_df = load_lvn_summary()
    ema20 = load_ema(EMA20_FILE)
    trades = load_trade_results()

    fig = build_figure(df, swing_df, hl_to_hh, lh_to_ll, lvn_df, ema20)
    add_trade_markers(fig, trades)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(OUTPUT_FILE, include_plotlyjs="cdn", full_html=True)
    print(f"Backtest trade chart saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
