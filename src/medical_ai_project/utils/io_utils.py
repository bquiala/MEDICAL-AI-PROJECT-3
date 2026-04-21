"""I/O helper functions for project artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return it as a Path object."""
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    """Persist a dictionary as formatted JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
