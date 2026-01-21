"""Configuration for the Crime Vulnerability Analysis backend.

Central place for file paths, groups, years, weighting, and normalization
strategy so that both the pipeline and notebook-style exploration can rely
on the same settings.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# Root of the project (this file lives in analysis/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "modeling"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_DATA_FILE = DATA_DIR / "masterData.csv"

CITIES_EXPECTED = 34
YEARS: List[int] = [2021, 2022, 2023]
GROUPS: List[str] = [
    "Children",
    "Women",
    "SC",
    "ST",
    "Senior Citizens",
]


@dataclass(frozen=True)
class NormalizationConfig:
    method: str = "minmax"  # or "zscore"


@dataclass(frozen=True)
class WeightsConfig:
    """Weights for each vulnerable group used in CVI computation.

    These can be tuned based on policy / literature. They must sum to 1.0.
    """

    women: float = 0.30
    sc: float = 0.25
    st: float = 0.20
    children: float = 0.15
    senior_citizens: float = 0.10

    @property
    def as_mapping(self) -> Dict[str, float]:
        return {
            "Women": self.women,
            "SC": self.sc,
            "ST": self.st,
            "Children": self.children,
            "Senior Citizens": self.senior_citizens,
        }


NORMALIZATION = NormalizationConfig()
WEIGHTS = WeightsConfig()


# Outlier detection settings
IQR_FACTOR: float = 1.5
Z_THRESHOLD: float = 3.0
