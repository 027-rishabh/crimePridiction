"""Static geographic mapping and enrichment for Indian metros.

This module DOES NOT perform any external API calls. Coordinates are
static centroids for the 34 metropolitan cities, curated once.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd

from .utils import write_json


# NOTE: Coordinates are approximate city centroids (lat, lon in decimal degrees).
CITY_COORDS: Dict[str, tuple[float, float]] = {
    "Agra": (27.1767, 78.0081),
    "Amritsar": (31.6340, 74.8723),
    "Asansol": (23.6739, 86.9524),
    "Aurangabad": (19.8762, 75.3433),
    "Bhopal": (23.2599, 77.4126),
    "Chandigarh City": (30.7333, 76.7794),
    "Dhanbad": (23.7957, 86.4304),
    "Durg-Bhilainagar": (21.1900, 81.2833),
    "Faridabad": (28.4089, 77.3178),
    "Gwalior": (26.2183, 78.1828),
    "Jabalpur": (23.1815, 79.9864),
    "Jamshedpur": (22.8046, 86.2029),
    "Jodhpur": (26.2389, 73.0243),
    "Kannur": (11.8745, 75.3704),
    "Kollam": (8.8932, 76.6141),
    "Kota": (25.2138, 75.8648),
    "Ludhiana": (30.9000, 75.8573),
    "Madurai": (9.9252, 78.1198),
    "Malappuram": (11.0510, 76.0711),
    "Meerut": (28.9845, 77.7064),
    "Nasik": (19.9975, 73.7898),
    "Prayagraj": (25.4358, 81.8463),
    "Raipur": (21.2514, 81.6296),
    "Rajkot": (22.3039, 70.8022),
    "Ranchi": (23.3441, 85.3096),
    "Srinagar": (34.0837, 74.7973),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Thrissur": (10.5276, 76.2144),
    "Tiruchirapalli": (10.7905, 78.7047),
    "Vadodara": (22.3072, 73.1812),
    "Varanasi": (25.3176, 82.9739),
    "Vasai Virar": (19.3919, 72.8397),
    "Vijayawada": (16.5062, 80.6480),
    "Vishakhapatnam": (17.6868, 83.2185),
}


def enrich_with_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Attach latitude and longitude to a dataframe with a City column."""

    out = df.copy()
    out["Latitude"] = out["City"].map(lambda c: CITY_COORDS.get(str(c), (None, None))[0])
    out["Longitude"] = out["City"].map(lambda c: CITY_COORDS.get(str(c), (None, None))[1])
    return out


def export_geo_dataset(df: pd.DataFrame, output_dir) -> None:
    """Export geo-joined dataset for the frontend heatmap."""

    enriched = enrich_with_geo(df)
    enriched.to_csv(output_dir / "geo_city_group_year.csv", index=False)

    # Also export a distinct cities geo list for potential use in frontend
    cities = (
        enriched[["City", "Latitude", "Longitude"]]
        .drop_duplicates()
        .sort_values("City")
    )
    cities.to_csv(output_dir / "geo_cities.csv", index=False)
    write_json(output_dir / "geo_cities.json", cities.to_dict(orient="records"))
