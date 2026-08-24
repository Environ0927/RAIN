"""Noise-aware, offline RAIN threshold calibration."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, slots=True)
class CalibrationRecord:
    schema_version: int
    tau: float
    quantile: float
    sample_count: int
    dimension: int
    clip_norm: float
    noise_multiplier: float
    seed: int
    mismatch_mean: float
    mismatch_std: float
    source: str = "independent-calibration-data"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def calibrate_threshold(
    benign_updates: np.ndarray,
    reference_bits: np.ndarray,
    *,
    clip_norm: float,
    noise_multiplier: float,
    quantile: float = 0.95,
    seed: int = 0,
) -> CalibrationRecord:
    updates = np.asarray(benign_updates, dtype=np.float64)
    reference = np.asarray(reference_bits, dtype=np.uint8)
    if updates.ndim != 2 or min(updates.shape) < 1:
        raise ValueError("benign_updates must be a non-empty matrix")
    if reference.shape != (updates.shape[1],) or np.any(reference > 1):
        raise ValueError("reference_bits must be a bit vector matching the update dimension")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    if not math.isfinite(noise_multiplier) or noise_multiplier < 0:
        raise ValueError("noise_multiplier must be finite and non-negative")
    if not 0 < quantile < 1:
        raise ValueError("quantile must lie strictly between zero and one")
    if not np.all(np.isfinite(updates)):
        raise ValueError("benign_updates contains non-finite values")

    norms = np.linalg.norm(updates, axis=1)
    scale = np.minimum(1.0, clip_norm / np.maximum(norms, np.finfo(np.float64).tiny))
    clipped = updates * scale[:, None]
    rng = np.random.default_rng(seed)
    if noise_multiplier:
        clipped += rng.normal(0.0, noise_multiplier * clip_norm, size=clipped.shape)
    bits = (clipped >= 0).astype(np.uint8)
    mismatch = np.mean(bits ^ reference[None, :], axis=1)
    # Higher interpolation is conservative: the fixed threshold never rounds down.
    tau = float(np.quantile(mismatch, quantile, method="higher"))
    return CalibrationRecord(
        schema_version=1,
        tau=tau,
        quantile=float(quantile),
        sample_count=updates.shape[0],
        dimension=updates.shape[1],
        clip_norm=float(clip_norm),
        noise_multiplier=float(noise_multiplier),
        seed=int(seed),
        mismatch_mean=float(np.mean(mismatch)),
        mismatch_std=float(np.std(mismatch)),
    )


def load_calibration(path: str | Path) -> CalibrationRecord:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("unsupported calibration schema_version")
    record = CalibrationRecord(**value)
    if not 0 <= record.tau <= 1:
        raise ValueError("calibration tau must lie in [0, 1]")
    return record
