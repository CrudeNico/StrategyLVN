#!/usr/bin/env python3
"""Identify Low Volume Nodes (LVNs) for ZigZag swing transitions."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "Findings" / "Results"
VOLUME_PROFILE_FILE = RESULTS_DIR / "volume_profile.csv"
HL_TO_HH_FILE = RESULTS_DIR / "zigzag_swings_hl_to_hh.csv"
LH_TO_LL_FILE = RESULTS_DIR / "zigzag_swings_lh_to_ll.csv"
OUTPUT_FILE = RESULTS_DIR / "zigzag_lvn_summary.csv"


def load_volume_profile() -> pd.DataFrame:
    if not VOLUME_PROFILE_FILE.exists():
        raise FileNotFoundError(
            f"Volume profile not found at {VOLUME_PROFILE_FILE}. "
            "Run compute_volume_profile.py first."
        )
    df = pd.read_csv(VOLUME_PROFILE_FILE)
    required_cols = {"price_min", "price_max", "price_center", "volume", "volume_pct"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Volume profile missing columns: {missing}")
    return df


def load_transitions(path: Path, transition_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Transition data '{transition_name}' not found at {path}. "
            "Run compute_zigzag.py first."
        )
    return pd.read_csv(path, parse_dates=["from_datetime", "to_datetime"])


def find_lvn_for_transition(
    profile: pd.DataFrame,
    low_price: float,
    high_price: float,
) -> pd.Series | None:
    """Return the volume profile row representing the LVN within a price window."""
    if low_price == high_price:
        mask = np.isclose(profile["price_center"], low_price)
    else:
        lower = min(low_price, high_price)
        upper = max(low_price, high_price)
        mask = (profile["price_center"] >= lower) & (profile["price_center"] <= upper)

    window = profile.loc[mask]
    if window.empty:
        # relax by intersecting using bin edges
        lower = min(low_price, high_price)
        upper = max(low_price, high_price)
        mask = (profile["price_max"] >= lower) & (profile["price_min"] <= upper)
        window = profile.loc[mask]

    if window.empty:
        return None
    idx = window["volume"].idxmin()
    return profile.loc[idx]


def build_lvn_records(
    transitions: pd.DataFrame,
    profile: pd.DataFrame,
    transition_type: str,
) -> Iterable[dict]:
    for row in transitions.itertuples():
        low_price = float(row.from_price)
        high_price = float(row.to_price)
        lvn_row = find_lvn_for_transition(profile, low_price, high_price)
        if lvn_row is None:
            continue

        yield {
            "transition_type": transition_type,
            "from_leg_id": int(row.from_leg_id),
            "to_leg_id": int(row.to_leg_id),
            "from_datetime": row.from_datetime,
            "to_datetime": row.to_datetime,
            "from_label": row.from_label,
            "to_label": row.to_label,
            "from_price": float(row.from_price),
            "to_price": float(row.to_price),
            "lvn_price_min": float(lvn_row["price_min"]),
            "lvn_price_max": float(lvn_row["price_max"]),
            "lvn_price_center": float(lvn_row["price_center"]),
            "lvn_volume": float(lvn_row["volume"]),
            "lvn_volume_pct": float(lvn_row.get("volume_pct", 0.0)),
            "price_change": float(row.price_change),
            "pct_change": float(row.pct_change),
            "time_delta_minutes": float(row.time_delta_minutes),
        }


def main() -> None:
    profile = load_volume_profile()
    hl_to_hh = load_transitions(HL_TO_HH_FILE, "HL->HH")
    lh_to_ll = load_transitions(LH_TO_LL_FILE, "LH->LL")

    records = list(build_lvn_records(hl_to_hh, profile, "HL_to_HH"))
    records.extend(build_lvn_records(lh_to_ll, profile, "LH_to_LL"))

    if not records:
        print("No LVN records were generated; check input data.")
        return

    output_df = pd.DataFrame.from_records(records)
    output_df.sort_values(
        ["transition_type", "from_leg_id"], inplace=True, ignore_index=True
    )
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved LVN summary to {OUTPUT_FILE} ({len(output_df)} rows)")


if __name__ == "__main__":
    main()
