"""Configuration loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(config_path: str) -> dict[str, Any]:
    """Load a JSON configuration file.

    Args:
        config_path: Path to JSON config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        json.JSONDecodeError: If config file is not valid JSON.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
