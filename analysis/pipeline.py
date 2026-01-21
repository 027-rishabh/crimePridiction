"""End-to-end analysis pipeline for Crime Vulnerability Analysis.

This module can be executed as a script:

    python -m analysis.pipeline
"""
from __future__ import annotations

import argparse

from . import config
from . import loaders
from . import validation
from . import eda
from . import cvi
from . import geo
from . import ranking


def run_pipeline(normalization: str | None = None) -> None:
    # Load
    df = loaders.load_master_data(config.MASTER_DATA_FILE)
    loaders.basic_shape_checks(df)

    # Data health
    validation.run_data_health_checks(df, config.OUTPUT_DIR)

    # Core EDA
    trends = eda.run_core_eda(df, config.OUTPUT_DIR)

    # CVI
    if normalization is not None:
        # temporarily override normalization method, but still persist all outputs
        normalized = cvi.normalize_crime_rates(trends, method=normalization)
        normalized.data.to_csv(
            config.OUTPUT_DIR / "city_year_group_normalized.csv", index=False
        )
        cvi_city_year, cvi_city_overall = cvi.compute_cvi(normalized)
        cvi_city_year.to_csv(config.OUTPUT_DIR / "cvi_city_year.csv", index=False)
        cvi_city_overall.to_csv(config.OUTPUT_DIR / "cvi_city_overall.csv", index=False)
    else:
        cvi_city_year, cvi_city_overall = cvi.run_cvi_pipeline(
            trends, config.OUTPUT_DIR
        )

    # Geo enrichment (uses trends so we include Group-level info)
    geo.export_geo_dataset(trends, config.OUTPUT_DIR)

    # Rankings and insights
    normalized_df = (
        trends.merge(
            cvi.normalize_crime_rates(trends).data[[
                "City",
                "Year",
                "Group",
                "Crime_Rate_Normalized",
            ]],
            on=["City", "Year", "Group"],
            how="left",
        )
    )
    ranking.run_ranking_and_insights(
        cvi_city_year=cvi_city_year,
        cvi_city_overall=cvi_city_overall,
        normalized_city_group=normalized_df,
        output_dir=config.OUTPUT_DIR,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run crime analysis pipeline")
    parser.add_argument(
        "--normalization",
        choices=["minmax", "zscore"],
        default=None,
        help="Override normalization method for this run.",
    )
    args = parser.parse_args(argv)
    run_pipeline(normalization=args.normalization)


if __name__ == "__main__":  # pragma: no cover
    main()
