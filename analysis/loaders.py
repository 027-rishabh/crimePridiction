"""Data loading helpers for the Crime Vulnerability Analysis backend."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from . import config


def load_master_data(path: Path | None = None) -> pd.DataFrame:
    """Load the long-format master dataset.

    Expects columns: City, Year, Crime_Rate, Group.
    """

    csv_path = path or config.MASTER_DATA_FILE
    df = pd.read_csv(csv_path)
    expected_cols = {"City", "Year", "Crime_Rate", "Group"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"masterData.csv missing expected columns: {sorted(missing)}")

    # Enforce dtypes
    df["Year"] = pd.to_numeric(df["Year"], errors="raise").astype(int)
    df["Crime_Rate"] = pd.to_numeric(df["Crime_Rate"], errors="raise")
    df["City"] = df["City"].astype(str)
    df["Group"] = df["Group"].astype(str)
    return df


def basic_shape_checks(df: pd.DataFrame) -> Tuple[int, int]:
    """Return shape after asserting that basic expectations roughly hold."""

    if not set(df["Group"].unique()).issuperset(config.GROUPS):
        # We allow extra groups but ensure at least the expected ones are present.
        missing = set(config.GROUPS).difference(df["Group"].unique())
        if missing:
            raise ValueError(f"Missing expected groups in master data: {sorted(missing)}")

    if len(df["Year"].unique()) != len(config.YEARS):
        raise ValueError("Unexpected number of years in master data")

    return df.shape
