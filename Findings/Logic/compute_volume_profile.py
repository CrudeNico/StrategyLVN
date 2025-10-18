#!/usr/bin/env python3
"""Compute a volume profile by price level using vectorbt."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "1minute.txt"
OUTPUT_FILE = PROJECT_ROOT / "Findings" / "Results" / "volume_profile.csv"


def load_data() -> pd.DataFrame:
    """Load OHLCV data with timestamp index."""
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


def build_volume_profile_indicator() -> vbt.IndicatorFactory:
    """Create a vectorbt indicator that bins volume by price range."""

    def volume_profile_bin_apply(
        close: np.ndarray,
        volume: np.ndarray,
        lower_bound: float,
        upper_bound: float,
        include_upper: bool,
    ) -> np.ndarray:
        close_arr = np.asarray(close, dtype=np.float64)
        volume_arr = np.asarray(volume, dtype=np.float64)

        cumulative = np.zeros(close_arr.size, dtype=np.float64)
        running_total = 0.0

        for i, (price, vol) in enumerate(zip(close_arr, volume_arr)):
            if not np.isfinite(price) or not np.isfinite(vol):
                cumulative[i] = running_total
                continue
            if include_upper:
                in_range = lower_bound <= price <= upper_bound
            else:
                in_range = lower_bound <= price < upper_bound
            if in_range:
                running_total += vol
            cumulative[i] = running_total

        return cumulative

    VolumeProfileBin = vbt.IndicatorFactory(
        class_name="VolumeProfileBin",
        short_name="vpbin",
        input_names=["close", "volume"],
        param_names=["lower_bound", "upper_bound", "include_upper"],
        output_names=["cum_volume"],
    ).from_apply_func(volume_profile_bin_apply, keep_pd=False)

    return VolumeProfileBin


def compute_volume_profile(df: pd.DataFrame, bins: int = 150) -> pd.DataFrame:
    """Run the volume profile indicator and structure the output."""
    price_min = float(df["low"].min())
    price_max = float(df["high"].max())

    indicator_cls = build_volume_profile_indicator()
    lower_bounds = np.linspace(price_min, price_max, bins, endpoint=False)
    bin_width = (price_max - price_min) / bins if bins > 0 else 0.0
    upper_bounds = lower_bounds + bin_width
    # Ensure the last bin captures any rounding errors
    upper_bounds[-1] = price_max
    include_upper = np.zeros(bins, dtype=bool)
    include_upper[-1] = True

    indicator = indicator_cls.run(
        df["close"],
        df["volume"],
        lower_bounds,
        upper_bounds,
        include_upper,
        param_product=False,
    )

    bin_edges = np.concatenate([lower_bounds, [price_max]])
    bin_edges = np.concatenate([lower_bounds, [price_max]])
    price_min_levels = lower_bounds
    price_max_levels = upper_bounds
    price_centers = (price_min_levels + price_max_levels) / 2.0
    bin_totals = indicator.cum_volume.iloc[-1].to_numpy(dtype=np.float64)

    num_bins = len(price_min_levels)
    profile = pd.DataFrame(
        {
            "bin_id": np.arange(num_bins, dtype=int) + 1,
            "price_min": price_min_levels,
            "price_max": price_max_levels,
            "price_center": price_centers,
            "volume": bin_totals,
        }
    )
    profile["volume_pct"] = np.where(
        profile["volume"].sum() > 0,
        profile["volume"] / profile["volume"].sum(),
        0.0,
    )
    profile["cumulative_volume_pct"] = profile["volume_pct"].cumsum()
    profile["threshold_bins"] = bins

    return profile[
        [
            "bin_id",
            "price_min",
            "price_max",
            "price_center",
            "volume",
            "volume_pct",
            "cumulative_volume_pct",
            "threshold_bins",
        ]
    ]


def main() -> None:
    df = load_data()
    profile_df = compute_volume_profile(df)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    profile_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved volume profile to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
