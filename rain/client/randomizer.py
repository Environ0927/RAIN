"""Client-side L2 clipping, Gaussian noise, and dense sign encoding."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .encoder import encode_signs


@dataclass(frozen=True, slots=True)
class RandomizedUpdate:
    bits: np.ndarray
    elapsed_seconds: float


class ClientRandomizer:
    """RAIN's local randomizer for one dense floating-point update."""

    def __init__(
        self,
        *,
        clip_norm: float,
        noise_multiplier: float,
        seed: int | None = None,
    ) -> None:
        if not np.isfinite(clip_norm) or clip_norm <= 0:
            raise ValueError("clip_norm must be finite and positive")
        if not np.isfinite(noise_multiplier) or noise_multiplier < 0:
            raise ValueError("noise_multiplier must be finite and non-negative")
        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self._rng = np.random.default_rng(seed)

    def randomize(self, update: np.ndarray) -> RandomizedUpdate:
        started = time.perf_counter()
        vector = np.asarray(update, dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            raise ValueError("update must be a non-empty 1-D vector")
        if not np.all(np.isfinite(vector)):
            raise ValueError("update must contain only finite values")

        norm = float(np.linalg.norm(vector))
        scale = min(1.0, self.clip_norm / max(norm, np.finfo(np.float64).tiny))
        clipped = vector * scale
        if self.noise_multiplier:
            standard_deviation = self.noise_multiplier * self.clip_norm
            clipped = clipped + self._rng.normal(
                0.0, standard_deviation, vector.shape
            ).astype(np.float32)
        bits = encode_signs(clipped)
        return RandomizedUpdate(bits=bits, elapsed_seconds=time.perf_counter() - started)

    def randomize_batch(self, updates: np.ndarray) -> RandomizedUpdate:
        matrix = np.asarray(updates, dtype=np.float32)
        if matrix.ndim != 2 or min(matrix.shape) <= 0:
            raise ValueError("updates must be a non-empty 2-D matrix")
        started = time.perf_counter()
        rows = [self.randomize(row).bits for row in matrix]
        return RandomizedUpdate(
            bits=np.ascontiguousarray(np.stack(rows, axis=0)),
            elapsed_seconds=time.perf_counter() - started,
        )
