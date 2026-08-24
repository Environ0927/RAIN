"""Versioned JSON experiment configuration."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_SECTIONS = {"data", "model", "training", "protocol", "privacy"}


def load_config(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("configuration schema_version must be 1")
    missing = REQUIRED_SECTIONS - value.keys()
    if missing:
        raise ValueError(f"configuration is missing sections: {sorted(missing)}")
    if value["training"].get("rounds", 0) < 1:
        raise ValueError("training.rounds must be positive")
    if value["protocol"].get("aggregation") != "rain":
        raise ValueError("the protocol trainer requires protocol.aggregation='rain'")
    return value
