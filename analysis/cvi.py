"""Crime Vulnerability Index (CVI) construction utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from . import config
from .utils import write_json


@dataclass
class NormalizedData:
    data: pd.DataFrame
    method: str


def normalize_minmax(series: pd.Series) -> pd.Series:
    min_v = series.min()
    max_v = series.max()
    if max_v == min_v:
        return pd.Series(0.0, index=series.index)
    return (series - min_v) / (max_v - min_v)


def normalize_zscore(series: pd.Series) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def normalize_crime_rates(trends: pd.DataFrame, method: str | None = None) -> NormalizedData:
    method = method or config.NORMALIZATION.method
    df = trends.copy()

    if method == "minmax":
        df["Crime_Rate_Normalized"] = normalize_minmax(df["Crime_Rate"])
    elif method == "zscore":
        df["Crime_Rate_Normalized"] = normalize_zscore(df["Crime_Rate"])
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

    return NormalizedData(data=df, method=method)


def compute_cvi(normalized: NormalizedData, weights: Dict[str, float] | None = None) -> pd.DataFrame:
    df = normalized.data.copy()
    weight_map = weights or config.WEIGHTS.as_mapping

    if not set(df["Group"].unique()).issubset(weight_map.keys()):
        missing = set(df["Group"].unique()).difference(weight_map.keys())
        if missing:
            raise ValueError(f"Missing weights for groups: {sorted(missing)}")

    df["Group_Weight"] = df["Group"].map(weight_map)
    df["Weighted_Score"] = df["Crime_Rate_Normalized"] * df["Group_Weight"]

    cvi_city_year = (
        df.groupby(["City", "Year"], as_index=False)["Weighted_Score"]
        .sum()
        .rename(columns={"Weighted_Score": "CVI"})
    )

    cvi_city_overall = (
        cvi_city_year.groupby("City", as_index=False)["CVI"]
        .mean()
        .rename(columns={"CVI": "CVI_Overall"})
    )

    return cvi_city_year, cvi_city_overall


def perturb_weights(base: Dict[str, float], factor: float) -> Iterable[Dict[str, float]]:
    """Yield a few perturbed weight dictionaries for sensitivity analysis."""

    groups = list(base.keys())
    for g in groups:
        for direction in (-1, 1):
            modified = base.copy()
            modified[g] = max(0.0, base[g] * (1 + direction * factor))
            # Renormalize to sum to 1
            total = sum(modified.values())
            modified = {k: v / total for k, v in modified.items()}
            yield modified


def run_cvi_pipeline(trends: pd.DataFrame, output_dir) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute CVI using configured normalization and weights and write outputs."""

    normalized = normalize_crime_rates(trends)
    cvi_city_year, cvi_city_overall = compute_cvi(normalized)

    normalized.data.to_csv(output_dir / "city_year_group_normalized.csv", index=False)
    cvi_city_year.to_csv(output_dir / "cvi_city_year.csv", index=False)
    cvi_city_overall.to_csv(output_dir / "cvi_city_overall.csv", index=False)

    # Sensitivity analysis: record rank correlations under perturbed weights
    base_weights = config.WEIGHTS.as_mapping
    sensitivity_records = []

    base_ranking = cvi_city_overall.sort_values("CVI_Overall", ascending=False).reset_index(drop=True)
    base_ranking["Base_Rank"] = base_ranking.index + 1

    for perturbed in perturb_weights(base_weights, factor=0.2):
        cvi_city_year_p, cvi_city_overall_p = compute_cvi(normalized, weights=perturbed)
        tmp = cvi_city_overall_p.sort_values("CVI_Overall", ascending=False).reset_index(drop=True)
        tmp["Rank"] = tmp.index + 1
        merged = base_ranking[["City", "Base_Rank"]].merge(tmp[["City", "Rank"]], on="City", how="inner")
        rank_diff = (merged["Base_Rank"] - merged["Rank"]).abs().mean()
        sensitivity_records.append(
            {
                "weights": perturbed,
                "mean_absolute_rank_shift": float(rank_diff),
            }
        )

    write_json(output_dir / "cvi_sensitivity.json", sensitivity_records)

    return cvi_city_year, cvi_city_overall
