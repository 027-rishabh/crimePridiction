"""Data validation and health checks for master dataset."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import config
from .utils import write_json


def detect_missing(df: pd.DataFrame) -> pd.Series:
    return df.isna().sum()


def detect_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=["City", "Year", "Group"]).sum())


def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - config.IQR_FACTOR * iqr
    upper = q3 + config.IQR_FACTOR * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=series.index)
    z = (series - mean) / std
    return z.abs() > config.Z_THRESHOLD


def build_data_health_summary(df: pd.DataFrame) -> Dict[str, object]:
    missing = detect_missing(df)
    duplicate_count = detect_duplicates(df)
    outliers_iqr = detect_outliers_iqr(df["Crime_Rate"]).sum()
    outliers_z = detect_outliers_zscore(df["Crime_Rate"]).sum()

    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_by_column": missing.to_dict(),
        "duplicate_rows": duplicate_count,
        "outliers": {
            "iqr_count": int(outliers_iqr),
            "zscore_count": int(outliers_z),
        },
        "cities": int(df["City"].nunique()),
        "years": sorted(int(y) for y in df["Year"].unique()),
        "groups": sorted(str(g) for g in df["Group"].unique()),
    }


def flag_row_level_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with boolean flags for potential issues per row."""

    issues = pd.DataFrame(index=df.index)
    issues["missing_crime_rate"] = df["Crime_Rate"].isna()
    issues["crime_rate_outlier_iqr"] = detect_outliers_iqr(df["Crime_Rate"])
    issues["crime_rate_outlier_z"] = detect_outliers_zscore(df["Crime_Rate"])

    # Combine with identifier columns
    result = pd.concat([df[["City", "Year", "Group", "Crime_Rate"]], issues], axis=1)
    return result


def run_data_health_checks(df: pd.DataFrame, output_dir) -> None:
    """Run all health checks and write outputs to disk."""

    summary = build_data_health_summary(df)
    write_json(output_dir / "data_health_summary.json", summary)

    row_issues = flag_row_level_issues(df)
    row_issues.to_csv(output_dir / "data_health_issues.csv", index=False)

    # Simple distributions snapshot for frontend (histogram bins counts)
    hist_counts, bin_edges = np.histogram(df["Crime_Rate"], bins=10)
    dist_payload = {
        "crime_rate": {
            "bin_edges": bin_edges.tolist(),
            "counts": hist_counts.tolist(),
        }
    }
    write_json(output_dir / "feature_distributions.json", dist_payload)
