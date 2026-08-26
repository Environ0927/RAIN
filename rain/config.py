"""Versioned JSON experiment configuration."""

from __future__ import annotations

import json
import math
from pathlib import Path

from rain.privacy.local_randomization import noise_multiplier_from_epsilon0
from rain.training.legacy_adapter import LEGACY_AGGREGATIONS, normalize_legacy_name


REQUIRED_SECTIONS = {"data", "model", "training", "protocol", "privacy"}
SUPPORTED_AGGREGATIONS = {
    "rain", "signsgd", "fedavg", "flod", "krum", "trim-mean", "median",
    "fltrust", "foundationfl", "rflpa",
} | set(LEGACY_AGGREGATIONS)
SUPPORTED_ATTACKS = {
    "none", "krum", "min-max", "scaling", "attack-dpfl", "raa", "woaa", "rsca",
}


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
    aggregation = normalize_legacy_name(str(protocol.get("aggregation", "")))
    protocol["aggregation"] = aggregation
    if aggregation not in SUPPORTED_AGGREGATIONS:
        raise ValueError(
            f"protocol.aggregation must be one of {sorted(SUPPORTED_AGGREGATIONS)}"
        )
    backend = str(protocol.get("backend", "secure" if aggregation == "rain" else "plaintext")).lower()
    if backend not in {"secure", "plaintext"}:
        raise ValueError("protocol.backend must be 'secure' or 'plaintext'")
    if backend == "secure" and aggregation != "rain":
        raise ValueError("only RAIN has a secure execution backend")
    if "integrity" in protocol and not isinstance(protocol["integrity"], bool):
        raise ValueError("protocol.integrity must be a Boolean")
    if bool(protocol.get("integrity", backend == "secure")) and backend != "secure":
        raise ValueError("RAIN-MAC integrity is available only with the secure backend")
    if "session_id" in protocol and not str(protocol["session_id"]).strip():
        raise ValueError("protocol.session_id must not be empty")
    if aggregation in {"rain", "flod"} and not ({"tau", "calibration"} & protocol.keys()):
        raise ValueError(f"{aggregation} requires protocol.tau or protocol.calibration")
    if aggregation == "divide-and-conquer":
        if int(protocol.get("dnc_niters", 1)) < 1:
            raise ValueError("divide-and-conquer requires positive protocol.dnc_niters")
        if not 0 < float(protocol.get("dnc_c", 1.0)) <= 1:
            raise ValueError("divide-and-conquer protocol.dnc_c must lie in (0, 1]")
        if int(protocol.get("dnc_b", 10_000)) < 2:
            raise ValueError("divide-and-conquer protocol.dnc_b must be at least two")
    if aggregation == "contra" and not 0 < float(protocol.get("selection_fraction", 1.0)) <= 1:
        raise ValueError("CONTRA protocol.selection_fraction must lie in (0, 1]")
    if aggregation == "flare" and int(protocol.get("flare_reference_batch", 32)) < 1:
        raise ValueError("FLARE protocol.flare_reference_batch must be positive")
    if aggregation == "foundationfl" and float(protocol.get("synthetic_ratio", 1.0)) <= 0:
        raise ValueError("FoundationFL protocol.synthetic_ratio must be positive")
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
    privacy = value["privacy"]
    if "epsilon0" in privacy:
        derived_noise = noise_multiplier_from_epsilon0(
            float(privacy["epsilon0"]),
            kappa=float(privacy.get("kappa", 1.0)),
            sensitivity_multiplier=float(privacy.get("sensitivity_multiplier", 2.0)),
        )
        if "noise_multiplier" in privacy and not math.isclose(
            float(privacy["noise_multiplier"]), derived_noise, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("privacy.noise_multiplier disagrees with epsilon0 mapping")
        privacy["noise_multiplier"] = derived_noise
    if float(privacy.get("clip_norm", 0.0)) <= 0:
        raise ValueError("privacy.clip_norm must be positive")
    if float(privacy.get("noise_multiplier", 0.0)) < 0:
        raise ValueError("privacy.noise_multiplier must be non-negative")
    attack = value.get("attack")
    if attack is not None:
        if not isinstance(attack, dict):
            raise ValueError("attack must be an object")
        attack_name = str(attack.get("name", "none")).lower()
        if attack_name not in SUPPORTED_ATTACKS:
            raise ValueError(f"attack.name must be one of {sorted(SUPPORTED_ATTACKS)}")
        malicious = int(attack.get("malicious_clients", 0))
        if malicious < 0:
            raise ValueError("attack.malicious_clients must be non-negative")
        clients = value["data"].get("clients")
        if isinstance(clients, int) and malicious >= clients and clients > 0:
            raise ValueError("attack.malicious_clients must be below data.clients")
        if attack_name == "none" and malicious != 0:
            raise ValueError("attack.name='none' requires zero malicious clients")
        if attack_name in {"raa", "rsca"} and int(attack.get("hamming_budget", -1)) < 0:
            raise ValueError(f"{attack_name} requires a non-negative hamming_budget")
        if attack_name == "woaa" and "hamming_budget" in attack:
            raise ValueError("woaa selects its own Hamming radius and must not set hamming_budget")
        if attack_name == "woaa" and not ({"tau", "calibration"} & protocol.keys()):
            raise ValueError("woaa requires protocol.tau or protocol.calibration")
        if attack_name == "scaling" and float(attack.get("scale", 10.0)) <= 0:
            raise ValueError("scaling attack requires a positive scale")
    return value
