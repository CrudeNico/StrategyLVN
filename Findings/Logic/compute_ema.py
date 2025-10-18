#!/usr/bin/env python3
"""Compute exponential moving averages for the minute dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "1minute.txt"
OUTPUT_FILE = PROJECT_ROOT / "Findings" / "Results" / "ema20.csv"


def load_data() -> pd.DataFrame:
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
    df.sort_values("datetime", inplace=True, ignore_index=True)
    return df


def compute_ema(df: pd.DataFrame, span: int = 50) -> pd.DataFrame:
    ema_series = df["close"].ewm(span=span, adjust=False, min_periods=span).mean()
    return pd.DataFrame(
        {
            "datetime": df["datetime"],
            "close": df["close"],
            f"ema_{span}": ema_series,
        }
    )


def main(span: int = 50) -> None:
    df = load_data()
    ema_df = compute_ema(df, span=span)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ema_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved EMA({span}) series to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
