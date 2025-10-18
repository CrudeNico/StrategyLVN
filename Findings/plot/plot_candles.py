#!/usr/bin/env python3
"""Generate a Plotly candlestick chart from the 1-minute data feed."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "1minute.txt"
RESULTS_FILE = PROJECT_ROOT / "Findings" / "Results" / "zigzag_swings.csv"
OUTPUT_FILE = Path(__file__).resolve().parent / "1minute_candles.html"


def load_data() -> pd.DataFrame:
    """Load and prepare the historical 1-minute OHLCV data."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Could not find data file at: {DATA_FILE}")

    df = pd.read_csv(
        DATA_FILE,
        header=None,
        names=["date", "time", "open", "high", "low", "close", "volume"],
        dtype={
            "date": "string",
            "time": "string",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "float64",
        },
    )
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M")
    df.sort_values("datetime", inplace=True, ignore_index=True)
    return df


def load_swings() -> Optional[pd.DataFrame]:
    """Load ZigZag swing labels if they exist."""
    if not RESULTS_FILE.exists():
        return None
    swing_df = pd.read_csv(RESULTS_FILE, parse_dates=["datetime"])
    swing_df.sort_values("datetime", inplace=True, ignore_index=True)
    return swing_df


def build_figure(df: pd.DataFrame, swing_df: Optional[pd.DataFrame]) -> go.Figure:
    """Build a Plotly candlestick figure."""
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["datetime"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                name="Price",
            )
        ]
    )

    if swing_df is not None and not swing_df.empty:
        text_positions = np.where(
            swing_df["pivot_side"] == "High", "top center", "bottom center"
        )
        marker_symbols = np.where(
            swing_df["pivot_side"] == "High", "triangle-up", "triangle-down"
        )
        marker_colors = np.where(
            swing_df["pivot_side"] == "High", "#2ca02c", "#d62728"
        )

        custom_leg_data = np.stack(
            [
                swing_df["leg_id"],
                swing_df["structure_label"],
                swing_df["price_change"],
                swing_df["pct_change"],
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=swing_df["datetime"],
                y=swing_df["pivot_price"],
                mode="lines",
                line=dict(color="#6b6b6b", width=1.5, dash="dot"),
                name="Swing Legs",
                hovertemplate=(
                    "Leg %{customdata[0]}<br>"
                    "Datetime: %{x}<br>"
                    "Price: %{y:.2f}<br>"
                    "Label: %{customdata[1]}<br>"
                    "Change: %{customdata[2]:+.2f} (%{customdata[3]:+.2%})<extra></extra>"
                ),
                customdata=custom_leg_data,
                showlegend=False,
            )
        )

        custom_marker_data = np.stack(
            [
                swing_df["leg_id"],
                swing_df["pivot_side"],
                swing_df["structure_label"],
                swing_df["price_change"],
                swing_df["pct_change"],
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scatter(
                x=swing_df["datetime"],
                y=swing_df["pivot_price"],
                mode="markers+text",
                marker=dict(
                    size=10,
                    color=marker_colors,
                    symbol=marker_symbols,
                    line=dict(color="#1f1f1f", width=1),
                ),
                text=swing_df["structure_label"],
                textposition=text_positions,
                textfont=dict(size=10),
                name="ZigZag Pivots",
                hovertemplate=(
                    "Leg %{customdata[0]}<br>"
                    "Type: %{customdata[1]}<br>"
                    "Label: %{customdata[2]}<br>"
                    "Price: %{y:.2f}<br>"
                    "Change: %{customdata[3]:+.2f} (%{customdata[4]:+.2%})<extra></extra>"
                ),
                customdata=custom_marker_data,
            )
        )
    fig.update_layout(
        title="1-Minute Candlestick Chart (December 2024)",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white",
        hovermode="x unified",
        dragmode="zoom",
        xaxis=dict(
            rangeslider=dict(visible=True, thickness=0.06),
            rangeselector=dict(
                buttons=[
                    dict(count=60, step="minute", stepmode="backward", label="1h"),
                    dict(count=4, step="hour", stepmode="backward", label="4h"),
                    dict(count=1, step="day", stepmode="backward", label="1d"),
                    dict(count=5, step="day", stepmode="backward", label="5d"),
                    dict(step="all", label="All"),
                ]
            ),
        ),
        yaxis=dict(fixedrange=False),
        modebar_add=["zoom2d", "pan2d", "select2d", "lasso2d", "resetScale2d"],
        modebar_remove=["autoScale2d"],
    )
    return fig


def main() -> None:
    df = load_data()
    swing_df = load_swings()
    fig = build_figure(df, swing_df)
    fig.write_html(OUTPUT_FILE, include_plotlyjs="cdn", full_html=True)
    print(f"Candlestick chart saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
