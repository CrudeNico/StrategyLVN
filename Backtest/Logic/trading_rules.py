#!/usr/bin/env python3
"""Shared trading rules and risk controls for the StrategyLVN backtest."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = PROJECT_ROOT / "data" / "1minute.txt"

# Capital configuration
STARTING_BALANCE_EUR: float = 10_000.0
RISK_PER_TRADE: float = 0.01  # 1% of current account

# Session rules
FRIDAY_WEEKDAY = 4  # Monday = 0, Friday = 4
FRIDAY_CUTOFF_HOUR = 16
FRIDAY_CUTOFF_MINUTE = 0


def load_price_data() -> pd.DataFrame:
    """Load raw minute data with a datetime index."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Expected price file at {DATA_FILE}")

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
    df.set_index("datetime", inplace=True)
    return df


def is_trading_allowed(timestamp: datetime) -> bool:
    """Return True if new trades may be opened at the given timestamp."""
    weekday = timestamp.weekday()
    if weekday >= 5:  # Saturday/Sunday
        return False

    if weekday == FRIDAY_WEEKDAY:
        cutoff = (
            timestamp.hour < FRIDAY_CUTOFF_HOUR
            or (timestamp.hour == FRIDAY_CUTOFF_HOUR and timestamp.minute < FRIDAY_CUTOFF_MINUTE)
        )
        return cutoff

    return True


def requires_forced_exit(timestamp: datetime) -> bool:
    """Flag the first minute bar on Friday at or after the cutoff as a forced exit."""
    if timestamp.weekday() != FRIDAY_WEEKDAY:
        return False
    return timestamp.hour == FRIDAY_CUTOFF_HOUR and timestamp.minute == FRIDAY_CUTOFF_MINUTE


def build_trading_windows(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate per-bar masks for trade eligibility and forced exits."""
    allowed = np.fromiter((is_trading_allowed(ts) for ts in index), dtype=bool, count=len(index))
    forced_exit = np.fromiter((requires_forced_exit(ts) for ts in index), dtype=bool, count=len(index))

    return pd.DataFrame(
        {
            "can_open_trades": allowed,
            "force_exit": forced_exit,
        },
        index=index,
    )


@dataclass
class RiskManager:
    """Handle position sizing based on account balance and risk limits."""

    balance: float = STARTING_BALANCE_EUR
    risk_fraction: float = RISK_PER_TRADE

    def risk_amount(self) -> float:
        """Capital to risk on the next position."""
        return self.balance * self.risk_fraction

    def position_size(self, entry_price: float, stop_price: float) -> float:
        """
        Contracts/shares to trade so that loss at stop equals risk allocation.

        Raises:
            ValueError: If entry_price equals stop_price or inputs are invalid.
        """
        if entry_price <= 0 or stop_price <= 0:
            raise ValueError("Entry and stop prices must be positive values.")

        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0:
            raise ValueError("Stop distance must be non-zero.")

        units = self.risk_amount() / stop_distance
        return units

    def update_balance(self, pnl: float) -> None:
        """Adjust internal balance after a trade closes."""
        self.balance += pnl


def annotate_with_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience helper that appends trading windows to the price DataFrame."""
    windows = build_trading_windows(df.index)
    return df.join(windows)


if __name__ == "__main__":
    prices = load_price_data()
    annotated = annotate_with_windows(prices)
    print(f"Loaded {len(prices):,} minute bars.")
    print(
        "Next trade risk (1%): "
        f"{RiskManager(balance=STARTING_BALANCE_EUR).risk_amount():.2f} EUR"
    )
    friday_windows = annotated[annotated.index.weekday == FRIDAY_WEEKDAY].tail(5)
    print("\nRecent Friday windows:")
    print(friday_windows[["can_open_trades", "force_exit"]])
