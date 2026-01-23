"""Ranking & Insight generation."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from .utils import write_json


def rank_cities(cvi_city_year: pd.DataFrame, cvi_city_overall: pd.DataFrame, output_dir) -> None:
    """Rank cities by CVI and writing to CSV outputs."""

    # Ranking per year
    per_year = (
        cvi_city_year.copy()
        .sort_values(["Year", "CVI"], ascending=[True, False])
    )
    per_year["Rank"] = per_year.groupby("Year")["CVI"].rank(ascending=False, method="min")
    per_year.to_csv(output_dir / "rankings_cities.csv", index=False)

    # Overall ranking
    overall = cvi_city_overall.copy().sort_values("CVI_Overall", ascending=False)
    overall["Rank"] = overall["CVI_Overall"].rank(ascending=False, method="min")
    overall.to_csv(output_dir / "rankings_cities_overall.csv", index=False)


def rank_groups(normalized_city_group: pd.DataFrame, output_dir) -> None:
    """Rank vulnerable groups by average normalized crime rate."""

    group_scores = (
        normalized_city_group.groupby("Group", as_index=False)["Crime_Rate_Normalized"]
        .mean()
        .rename(columns={"Crime_Rate_Normalized": "Group_Vulnerability_Score"})
    )
    group_scores = group_scores.sort_values("Group_Vulnerability_Score", ascending=False)
    group_scores["Rank"] = group_scores["Group_Vulnerability_Score"].rank(
        ascending=False, method="min"
    )
    group_scores.to_csv(output_dir / "rankings_groups.csv", index=False)


def build_policy_insights(
    cvi_city_overall: pd.DataFrame, group_scores: pd.DataFrame
) -> Dict[str, object]:
    """Generate simple textual/policy-relevant insights from rankings."""

    top_cities = (
        cvi_city_overall.sort_values("CVI_Overall", ascending=False).head(5)[
            ["City", "CVI_Overall"]
        ]
    )
    bottom_cities = (
        cvi_city_overall.sort_values("CVI_Overall", ascending=True).head(5)[
            ["City", "CVI_Overall"]
        ]
    )

    top_groups = group_scores.sort_values(
        "Group_Vulnerability_Score", ascending=False
    ).head(5)[["Group", "Group_Vulnerability_Score"]]

    insights = {
        "top_high_risk_cities": top_cities.to_dict(orient="records"),
        "least_risk_cities": bottom_cities.to_dict(orient="records"),
        "most_vulnerable_groups": top_groups.to_dict(orient="records"),
        "n_cities": int(cvi_city_overall["City"].nunique()),
        "interpretation_notes": [
            "Higher CVI indicates greater relative vulnerability across vulnerable groups.",
            "Group vulnerability scores are based on normalized crime rates averaged across all cities and years.",
        ],
    }
    return insights


def run_ranking_and_insights(
    cvi_city_year: pd.DataFrame,
    cvi_city_overall: pd.DataFrame,
    normalized_city_group: pd.DataFrame,
    output_dir,
) -> None:
    """Run full ranking pipeline and write ranking tables and insights."""

    rank_cities(cvi_city_year, cvi_city_overall, output_dir)
    rank_groups(normalized_city_group, output_dir)

    group_scores = pd.read_csv(output_dir / "rankings_groups.csv")
    insights = build_policy_insights(cvi_city_overall, group_scores)
    write_json(output_dir / "policy_insights.json", insights)
