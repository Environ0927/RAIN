"""Versioned JSON experiment configuration."""

from __future__ import annotations

import json
from pathlib import Path


REQUIRED_SECTIONS = {"data", "model", "training", "protocol", "privacy"}
SUPPORTED_AGGREGATIONS = {"rain", "signsgd", "fedavg", "flod"}


def load_config(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("configuration schema_version must be 1")
    missing = REQUIRED_SECTIONS - value.keys()
    if missing:
        raise ValueError(f"configuration is missing sections: {sorted(missing)}")
    if value["training"].get("rounds", 0) < 1:
        raise ValueError("training.rounds must be positive")
    protocol = value["protocol"]
    aggregation = str(protocol.get("aggregation", "")).lower()
    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"protocol.aggregation must be one of {sorted(SUPPORTED_AGGREGATIONS)}"
        )
    backend = str(protocol.get("backend", "secure" if aggregation == "rain" else "plaintext")).lower()
    if backend not in {"secure", "plaintext"}:
        raise ValueError("protocol.backend must be 'secure' or 'plaintext'")
    if backend == "secure" and aggregation != "rain":
        raise ValueError("only RAIN has a secure execution backend")
    if aggregation in {"rain", "flod"} and not ({"tau", "calibration"} & protocol.keys()):
        raise ValueError(f"{aggregation} requires protocol.tau or protocol.calibration")
    preprocessing = str(protocol.get("preprocessing", "clip_noise" if backend == "secure" else "none"))
    if preprocessing not in {"none", "clip_noise"}:
        raise ValueError("protocol.preprocessing must be 'none' or 'clip_noise'")
    training = value["training"]
    if int(training.get("local_steps", 1)) < 1:
        raise ValueError("training.local_steps must be positive")
    if "client_learning_rate" in training:
        if float(training["client_learning_rate"]) <= 0:
            raise ValueError("training.client_learning_rate must be positive")
        momentum = float(training.get("client_momentum", 0.0))
        if not 0 <= momentum < 1:
            raise ValueError("training.client_momentum must lie in [0, 1)")
        if float(training.get("client_weight_decay", 0.0)) < 0:
            raise ValueError("training.client_weight_decay must be non-negative")
        if bool(training.get("client_nesterov", False)) and momentum <= 0:
            raise ValueError("training.client_nesterov requires positive momentum")
    return value
