"""Exploratory Data Analysis utilities for crime trends and comparisons."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from . import config
from .utils import write_json


def compute_city_year_group_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Return tidy dataframe with city-year-group crime metrics and YoY growth."""

    base = df.copy()

    # Aggregate in case there are multiple entries per (City, Year, Group)
    grouped = (
        base.groupby(["City", "Year", "Group"], as_index=False)["Crime_Rate"]
        .mean()
        .rename(columns={"Crime_Rate": "Crime_Rate"})
    )

    # Year-on-year percentage change within each City & Group
    grouped = grouped.sort_values(["City", "Group", "Year"])  # type: ignore[list-item]
    grouped["YoY_Change"] = (
        grouped.groupby(["City", "Group"])["Crime_Rate"].pct_change() * 100.0
    )

    return grouped


def classify_trend(city_group_series: pd.Series) -> str:
    """Classify a time series as increasing, decreasing, or stable.

    Very simple heuristic based on slope of linear fit.
    """

    years = np.array(config.YEARS, dtype=float)
    values = city_group_series.values.astype(float)
    if len(values) != len(years):
        return "unstable"
    # Linear regression slope
    slope = np.polyfit(years, values, 1)[0]
    if slope > 0.0:
        return "increasing"
    if slope < 0.0:
        return "decreasing"
    return "stable"


def build_trend_summaries(trends: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    """Return nested mapping trend[city][group] -> label."""

    mapping: Dict[str, Dict[str, str]] = {}
    for (city, group), sub in trends.groupby(["City", "Group"]):
        label = classify_trend(sub["Crime_Rate"])
        mapping.setdefault(str(city), {})[str(group)] = label
    return mapping


def compute_group_city_aggregates(trends: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean crime rate per City & Group for comparison charts."""

    agg = (
        trends.groupby(["City", "Group"], as_index=False)["Crime_Rate"]
        .mean()
        .rename(columns={"Crime_Rate": "Mean_Crime_Rate"})
    )
    return agg


def compute_volatility(trends: pd.DataFrame) -> pd.DataFrame:
    """Compute variance and standard deviation of crime rate across years."""

    stats = (
        trends.groupby(["City", "Group"], as_index=False)["Crime_Rate"]
        .agg(["var", "std"])
        .reset_index()
        .rename(columns={"var": "Var_Crime_Rate", "std": "Std_Crime_Rate"})
    )
    return stats


def run_core_eda(df: pd.DataFrame, output_dir) -> pd.DataFrame:
    """Run core EDA computations and write outputs.

    Returns the main city-year-group trends dataframe for downstream use.
    """

    trends = compute_city_year_group_trends(df)
    trends.to_csv(output_dir / "city_year_group_trends.csv", index=False)

    agg = compute_group_city_aggregates(trends)
    agg.to_csv(output_dir / "group_comparison_city_agg.csv", index=False)

    volatility = compute_volatility(trends)
    volatility.to_csv(output_dir / "city_group_volatility.csv", index=False)

    trend_labels = build_trend_summaries(trends)
    write_json(output_dir / "crime_trend_summary.json", trend_labels)

    return trends
