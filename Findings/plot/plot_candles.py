#!/usr/bin/env python3
"""Generate a Plotly candlestick chart with ZigZag structure, transitions, and LVNs."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "1minute.txt"
RESULTS_DIR = PROJECT_ROOT / "Findings" / "Results"
SWINGS_FILE = RESULTS_DIR / "zigzag_swings.csv"
HL_TO_HH_FILE = RESULTS_DIR / "zigzag_swings_hl_to_hh.csv"
LH_TO_LL_FILE = RESULTS_DIR / "zigzag_swings_lh_to_ll.csv"
LVN_FILE = RESULTS_DIR / "zigzag_lvn_summary.csv"
EMA20_FILE = RESULTS_DIR / "ema20.csv"
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
    if not SWINGS_FILE.exists():
        return None
    swing_df = pd.read_csv(SWINGS_FILE, parse_dates=["datetime"])
    swing_df.sort_values("datetime", inplace=True, ignore_index=True)
    return swing_df


def load_transitions(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["from_datetime", "to_datetime"])
    df.sort_values(["from_datetime", "to_datetime"], inplace=True, ignore_index=True)
    return df


def load_lvn_summary() -> Optional[pd.DataFrame]:
    if not LVN_FILE.exists():
        return None
    df = pd.read_csv(LVN_FILE, parse_dates=["from_datetime", "to_datetime"])
    df.sort_values(["transition_type", "from_datetime"], inplace=True, ignore_index=True)
    return df


def load_ema(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    df.sort_values("datetime", inplace=True, ignore_index=True)
    return df


def _append_segment_data(
    transitions: pd.DataFrame,
    from_col: str,
    to_col: str,
    price_from_col: str,
    price_to_col: str,
    extra_columns: list[str],
) -> tuple[list, list, list]:
    xs: list = []
    ys: list = []
    custom: list = []
    for row in transitions.itertuples():
        xs.extend([getattr(row, from_col), getattr(row, to_col), None])
        ys.extend([getattr(row, price_from_col), getattr(row, price_to_col), None])
        extras = [getattr(row, col) for col in extra_columns]
        custom.extend([extras, extras, [None] * len(extras)])
    return xs, ys, custom


def build_figure(
    df: pd.DataFrame,
    swing_df: Optional[pd.DataFrame],
    hl_to_hh: Optional[pd.DataFrame],
    lh_to_ll: Optional[pd.DataFrame],
    lvn_df: Optional[pd.DataFrame],
    ema20: Optional[pd.DataFrame],
) -> go.Figure:
    """Build a Plotly candlestick figure with additional overlays."""
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

    if ema20 is not None and not ema20.empty:
        fig.add_trace(
            go.Scatter(
                x=ema20["datetime"],
                y=ema20["ema_20"],
                mode="lines",
                line=dict(color="#2962ff", width=1.6),
                name="EMA 20",
                hovertemplate="EMA20<br>%{x}<br>%{y:.2f}<extra></extra>",
            )
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

    if hl_to_hh is not None and not hl_to_hh.empty:
        hl_x, hl_y, hl_custom = _append_segment_data(
            hl_to_hh,
            from_col="from_datetime",
            to_col="to_datetime",
            price_from_col="from_price",
            price_to_col="to_price",
            extra_columns=[
                "from_leg_id",
                "to_leg_id",
                "price_change",
                "pct_change",
                "time_delta_minutes",
            ],
        )
        fig.add_trace(
            go.Scatter(
                x=hl_x,
                y=hl_y,
                mode="lines",
                line=dict(color="#1b9e77", width=1.5),
                name="HL → HH",
                legendgroup="transitions",
                hovertemplate=(
                    "HL → HH<br>"
                    "From leg %{customdata[0]}, to %{customdata[1]}<br>"
                    "Price Δ %{customdata[2]:+.2f} (%{customdata[3]:+.2%})<br>"
                    "Duration %{customdata[4]:.1f} min<extra></extra>"
                ),
                customdata=hl_custom,
            )
        )

    if lh_to_ll is not None and not lh_to_ll.empty:
        lh_x, lh_y, lh_custom = _append_segment_data(
            lh_to_ll,
            from_col="from_datetime",
            to_col="to_datetime",
            price_from_col="from_price",
            price_to_col="to_price",
            extra_columns=[
                "from_leg_id",
                "to_leg_id",
                "price_change",
                "pct_change",
                "time_delta_minutes",
            ],
        )
        fig.add_trace(
            go.Scatter(
                x=lh_x,
                y=lh_y,
                mode="lines",
                line=dict(color="#d95f02", width=1.5),
                name="LH → LL",
                legendgroup="transitions",
                hovertemplate=(
                    "LH → LL<br>"
                    "From leg %{customdata[0]}, to %{customdata[1]}<br>"
                    "Price Δ %{customdata[2]:+.2f} (%{customdata[3]:+.2%})<br>"
                    "Duration %{customdata[4]:.1f} min<extra></extra>"
                ),
                customdata=lh_custom,
            )
        )

    if lvn_df is not None and not lvn_df.empty:
        lvn_x, lvn_y, lvn_custom = _append_segment_data(
            lvn_df,
            from_col="from_datetime",
            to_col="to_datetime",
            price_from_col="lvn_price_center",
            price_to_col="lvn_price_center",
            extra_columns=[
                "transition_type",
                "from_leg_id",
                "to_leg_id",
                "lvn_price_min",
                "lvn_price_max",
                "lvn_volume",
                "lvn_volume_pct",
            ],
        )
        fig.add_trace(
            go.Scatter(
                x=lvn_x,
                y=lvn_y,
                mode="lines",
                line=dict(color="#636363", width=2, dash="dash"),
                name="LVN Band",
                legendgroup="lvn",
                hovertemplate=(
                    "%{customdata[0]} LVN<br>"
                    "Legs %{customdata[1]} → %{customdata[2]}<br>"
                    "Band [%{customdata[3]:.2f}, %{customdata[4]:.2f}]<br>"
                    "Vol %{customdata[5]:.0f} (%{customdata[6]:.2%})<extra></extra>"
                ),
                customdata=lvn_custom,
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
    hl_to_hh = load_transitions(HL_TO_HH_FILE)
    lh_to_ll = load_transitions(LH_TO_LL_FILE)
    lvn_df = load_lvn_summary()
    ema20 = load_ema(EMA20_FILE)
    fig = build_figure(df, swing_df, hl_to_hh, lh_to_ll, lvn_df, ema20)
    fig.write_html(OUTPUT_FILE, include_plotlyjs="cdn", full_html=True)
    print(f"Candlestick chart saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
