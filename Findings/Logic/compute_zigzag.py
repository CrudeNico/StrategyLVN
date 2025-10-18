#!/usr/bin/env python3
"""Compute ZigZag swing structure using vectorbt."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "1minute.txt"
OUTPUT_FILE = PROJECT_ROOT / "Findings" / "Results" / "zigzag_swings.csv"
HL_TO_HH_FILE = PROJECT_ROOT / "Findings" / "Results" / "zigzag_swings_hl_to_hh.csv"
LH_TO_LL_FILE = PROJECT_ROOT / "Findings" / "Results" / "zigzag_swings_lh_to_ll.csv"


def load_data() -> pd.DataFrame:
    """Load minute OHLCV data with a combined datetime index."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Expected data file at {DATA_FILE}")

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
    df.set_index("datetime", inplace=True)
    return df


def build_zigzag_indicator() -> vbt.IndicatorFactory:
    """Build a custom ZigZag indicator powered by vectorbt."""

    def zigzag_apply(close: np.ndarray, pct: float) -> Tuple[np.ndarray, np.ndarray]:
        """Return ZigZag pivots and pivot direction arrays."""
        close_arr = np.asarray(close, dtype=np.float64)
        if close_arr.size == 0:
            return close_arr, close_arr

        pivots = np.full_like(close_arr, np.nan)
        pivot_dir = np.zeros_like(close_arr, dtype=np.int64)

        last_pivot_idx = 0
        last_pivot_price = close_arr[0]
        trend = 0  # 1 for uptrend pivot (high), -1 for downtrend pivot (low)

        for i in range(1, close_arr.size):
            price = close_arr[i]
            if trend == 0:
                up_move = price / last_pivot_price - 1.0
                down_move = last_pivot_price / price - 1.0 if price != 0 else 0.0
                if up_move >= pct:
                    trend = 1
                    last_pivot_idx = i
                    last_pivot_price = price
                    pivots[i] = price
                    pivot_dir[i] = 1
                elif down_move >= pct:
                    trend = -1
                    last_pivot_idx = i
                    last_pivot_price = price
                    pivots[i] = price
                    pivot_dir[i] = -1
            elif trend == 1:
                if price >= last_pivot_price:
                    pivots[last_pivot_idx] = np.nan
                    pivot_dir[last_pivot_idx] = 0
                    last_pivot_idx = i
                    last_pivot_price = price
                    pivots[i] = price
                    pivot_dir[i] = 1
                elif (last_pivot_price - price) / last_pivot_price >= pct:
                    trend = -1
                    last_pivot_idx = i
                    last_pivot_price = price
                    pivots[i] = price
                    pivot_dir[i] = -1
            else:  # trend == -1
                if price <= last_pivot_price:
                    pivots[last_pivot_idx] = np.nan
                    pivot_dir[last_pivot_idx] = 0
                    last_pivot_idx = i
                    last_pivot_price = price
                    pivots[i] = price
                    pivot_dir[i] = -1
                elif (price - last_pivot_price) / last_pivot_price >= pct:
                    trend = 1
                    last_pivot_idx = i
                    last_pivot_price = price
                    pivots[i] = price
                    pivot_dir[i] = 1

        pivots[0] = close_arr[0]
        pivot_dir[0] = 0
        if np.isnan(pivots[last_pivot_idx]):
            pivots[last_pivot_idx] = last_pivot_price
            pivot_dir[last_pivot_idx] = trend

        return pivots, pivot_dir

    ZigZag = vbt.IndicatorFactory(
        class_name="ZigZag",
        short_name="zigzag",
        input_names=["close"],
        param_names=["pct"],
        output_names=["pivots", "pivot_dir"],
    ).from_apply_func(zigzag_apply, keep_pd=False)

    return ZigZag


def label_structure(df: pd.DataFrame, zigzag, pct: float) -> pd.DataFrame:
    """Create a labeled swing structure DataFrame."""
    pivots = pd.Series(zigzag.pivots, index=df.index, name="pivot_price")
    pivot_dir = pd.Series(zigzag.pivot_dir, index=df.index, name="pivot_dir")

    mask = pivot_dir != 0
    pivot_points = df.loc[mask, ["close"]].copy()
    pivot_points["pivot_price"] = pivots.loc[mask]
    pivot_points["pivot_dir"] = pivot_dir.loc[mask]
    pivot_points["pivot_side"] = np.where(pivot_points["pivot_dir"] > 0, "High", "Low")

    structure_labels = []
    prev_high = None
    prev_low = None
    for _, row in pivot_points.iterrows():
        price = row["pivot_price"]
        if row["pivot_side"] == "High":
            if prev_high is None:
                label = "HH"
            else:
                label = "HH" if price >= prev_high else "LH"
            prev_high = price
        else:
            if prev_low is None:
                label = "LL"
            else:
                label = "HL" if price >= prev_low else "LL"
            prev_low = price
        structure_labels.append(label)

    pivot_points["structure_label"] = structure_labels
    pivot_points["leg_id"] = np.arange(1, len(pivot_points) + 1)
    pivot_points["price_change"] = pivot_points["pivot_price"].diff().fillna(0.0)
    pivot_points["pct_change"] = pivot_points["pivot_price"].pct_change().fillna(0.0)
    pivot_points["swing_direction"] = np.where(
        pivot_points["price_change"] >= 0, "Up", "Down"
    )
    pivot_points.reset_index(inplace=True)
    pivot_points.rename(columns={"index": "datetime"}, inplace=True)
    pivot_points["threshold_pct"] = pct

    ordered_columns = [
        "leg_id",
        "datetime",
        "pivot_side",
        "structure_label",
        "pivot_price",
        "price_change",
        "pct_change",
        "swing_direction",
        "threshold_pct",
    ]

    return pivot_points[ordered_columns]


def build_transition_pairs(
    swing_df: pd.DataFrame, from_label: str, to_label: str
) -> pd.DataFrame:
    """Create a DataFrame describing sequential label transitions."""
    records = []
    entries = swing_df.sort_values("leg_id").reset_index(drop=True)

    for idx in range(len(entries) - 1):
        current_row = entries.iloc[idx]
        next_row = entries.iloc[idx + 1]
        if (
            current_row["structure_label"] == from_label
            and next_row["structure_label"] == to_label
        ):
            time_delta = next_row["datetime"] - current_row["datetime"]
            records.append(
                {
                    "from_leg_id": int(current_row["leg_id"]),
                    "from_datetime": current_row["datetime"],
                    "from_label": current_row["structure_label"],
                    "from_pivot_side": current_row["pivot_side"],
                    "from_price": float(current_row["pivot_price"]),
                    "to_leg_id": int(next_row["leg_id"]),
                    "to_datetime": next_row["datetime"],
                    "to_label": next_row["structure_label"],
                    "to_pivot_side": next_row["pivot_side"],
                    "to_price": float(next_row["pivot_price"]),
                    "price_change": float(next_row["pivot_price"] - current_row["pivot_price"]),
                    "pct_change": float(next_row["pivot_price"] / current_row["pivot_price"] - 1.0)
                    if current_row["pivot_price"] != 0
                    else 0.0,
                    "time_delta_minutes": float(time_delta.total_seconds() / 60.0),
                    "threshold_pct": float(next_row["threshold_pct"]),
                }
            )

    columns = [
        "from_leg_id",
        "from_datetime",
        "from_label",
        "from_pivot_side",
        "from_price",
        "to_leg_id",
        "to_datetime",
        "to_label",
        "to_pivot_side",
        "to_price",
        "price_change",
        "pct_change",
        "time_delta_minutes",
        "threshold_pct",
    ]

    return pd.DataFrame.from_records(records, columns=columns)


def main(pct_threshold: float = 0.001) -> None:
    df = load_data()
    zigzag_cls = build_zigzag_indicator()
    zigzag = zigzag_cls.run(df["close"], pct=pct_threshold)
    swing_df = label_structure(df, zigzag, pct_threshold)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    swing_df.to_csv(OUTPUT_FILE, index=False)

    hl_to_hh_df = build_transition_pairs(swing_df, "HL", "HH")
    lh_to_ll_df = build_transition_pairs(swing_df, "LH", "LL")
    hl_to_hh_df.to_csv(HL_TO_HH_FILE, index=False)
    lh_to_ll_df.to_csv(LH_TO_LL_FILE, index=False)
    print(f"Saved swing structure to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
