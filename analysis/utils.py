"""Utility helpers for the Crime Vulnerability Analysis backend."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import json


def ensure_parent_dir(path: Path) -> None:
    """Create parent directory for a file path if it does not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Any, **json_kwargs: Any) -> None:
    """Write Python data as pretty JSON to the given path."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, **json_kwargs)


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass instance (possibly nested) to a dict."""

    try:
        return asdict(obj)
    except TypeError:
        return dict(obj)
