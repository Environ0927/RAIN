"""Plaintext aggregation rules for utility-only convergence experiments.

This module intentionally contains no shuffling, secret sharing, or transport
simulation.  It lets all baselines consume the same client update matrix so a
convergence comparison measures the aggregation rule rather than the secure
execution backend.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np


PLAINTEXT_AGGREGATIONS = frozenset({"rain", "signsgd", "fedavg", "flod"})


@dataclass(frozen=True, slots=True)
class PlaintextAggregationMetrics:
    aggregation_seconds: float
    preprocessing_seconds: float
    accepted_clients: int | None
    weight_sum: float | None
    threshold_count: int | None
    mismatch_mean: float | None
    mismatch_min: float | None
    mismatch_max: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PlaintextAggregationResult:
    update: np.ndarray
    metrics: PlaintextAggregationMetrics


def _validate_updates(updates: np.ndarray) -> np.ndarray:
    matrix = np.asarray(updates, dtype=np.float32)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("updates must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("updates must contain only finite values")
    return matrix


def _clip_and_noise(
    updates: np.ndarray,
    *,
    preprocessing: str,
    clip_norm: float,
    noise_multiplier: float,
    seed: int,
) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    if preprocessing == "none":
        if noise_multiplier != 0:
            raise ValueError("noise_multiplier must be zero when preprocessing='none'")
        return updates, time.perf_counter() - started
    if preprocessing != "clip_noise":
        raise ValueError("preprocessing must be 'none' or 'clip_noise'")
    if not np.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    if not np.isfinite(noise_multiplier) or noise_multiplier < 0:
        raise ValueError("noise_multiplier must be finite and non-negative")
    processed = np.empty_like(updates)
    rng = np.random.default_rng(seed)
    for index, row in enumerate(updates):
        norm = float(np.linalg.norm(row))
        scale = min(1.0, clip_norm / max(norm, np.finfo(np.float64).tiny))
        processed[index] = row * scale
        if noise_multiplier:
            processed[index] += rng.normal(
                0.0, noise_multiplier * clip_norm, size=row.shape
            ).astype(np.float32)
    return processed, time.perf_counter() - started


def _sign_bits(updates: np.ndarray) -> np.ndarray:
    # RAIN maps an exact zero to +1.  The same convention is used for FLOD;
    # SignSGD handles ties after voting and therefore keeps zero updates at 0.
    return np.ascontiguousarray((updates >= 0).astype(np.uint8))


def _reference_weights(
    bits: np.ndarray, reference_bits: np.ndarray, tau: float
) -> tuple[np.ndarray, int]:
    reference = np.asarray(reference_bits, dtype=np.uint8).reshape(-1)
    if reference.shape != (bits.shape[1],) or np.any(reference > 1):
        raise ValueError("reference_bits must be a bit vector matching the update dimension")
    if not np.isfinite(tau) or not 0 < tau <= 1:
        raise ValueError("tau must lie in (0, 1]")
    threshold = int(np.floor(bits.shape[1] * tau))
    if threshold < 1:
        raise ValueError("tau is too small for the update dimension")
    distances = np.asarray(
        [np.count_nonzero(row != reference) for row in bits], dtype=np.int64
    )
    return np.maximum(0, threshold - distances), threshold


def aggregate_plaintext(
    updates: np.ndarray,
    *,
    method: str,
    reference_bits: np.ndarray | None = None,
    tau: float | None = None,
    preprocessing: str = "none",
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.0,
    seed: int = 0,
    client_weights: np.ndarray | None = None,
) -> PlaintextAggregationResult:
    """Aggregate one client-update matrix without a secure execution layer.

    ``rain`` returns the final coordinate sign from Algorithm 3. ``flod`` uses
    the same reference-distance weights but preserves the normalized weighted
    sign magnitude, matching the repository's retained FLOD implementation.
    ``signsgd`` performs the standard coordinate majority vote with zero for a
    tie. ``fedavg`` averages full-precision client gradients.
    """

    normalized = method.lower().replace("-", "")
    if normalized == "signsgd":
        method = "signsgd"
    else:
        method = method.lower()
    if method not in PLAINTEXT_AGGREGATIONS:
        raise ValueError(f"unsupported plaintext aggregation: {method}")
    matrix = _validate_updates(updates)
    processed, preprocessing_seconds = _clip_and_noise(
        matrix, preprocessing=preprocessing, clip_norm=clip_norm,
        noise_multiplier=noise_multiplier, seed=seed,
    )
    started = time.perf_counter()
    accepted_clients: int | None = None
    weight_sum: float | None = None
    threshold: int | None = None
    mismatch_mean: float | None = None
    mismatch_min: float | None = None
    mismatch_max: float | None = None

    if method == "fedavg":
        if client_weights is None:
            update = np.mean(processed, axis=0, dtype=np.float64).astype(np.float32)
        else:
            weights = np.asarray(client_weights, dtype=np.float64).reshape(-1)
            if weights.shape != (processed.shape[0],) or np.any(weights < 0):
                raise ValueError("client_weights must be non-negative and match the clients")
            if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
                raise ValueError("client_weights must have a finite positive sum")
            update = np.average(processed, axis=0, weights=weights).astype(np.float32)
    elif method == "signsgd":
        votes = np.sum(np.sign(processed), axis=0, dtype=np.float32)
        update = np.sign(votes).astype(np.float32)
    else:
        if reference_bits is None or tau is None:
            raise ValueError(f"{method} requires reference_bits and tau")
        bits = _sign_bits(processed)
        weights, threshold = _reference_weights(bits, reference_bits, float(tau))
        reference = np.asarray(reference_bits, dtype=np.uint8).reshape(-1)
        mismatch = np.asarray(
            [np.count_nonzero(row != reference) / bits.shape[1] for row in bits],
            dtype=np.float64,
        )
        mismatch_mean = float(np.mean(mismatch))
        mismatch_min = float(np.min(mismatch))
        mismatch_max = float(np.max(mismatch))
        accepted_clients = int(np.count_nonzero(weights))
        weight_sum = float(weights.sum())
        accumulator = np.zeros(bits.shape[1], dtype=np.float64)
        for weight, row in zip(weights, bits):
            if weight:
                accumulator += float(weight) * (row.astype(np.float32) * 2.0 - 1.0)
        if method == "rain":
            update = np.where(accumulator >= 0, 1.0, -1.0).astype(np.float32)
        elif weight_sum:
            update = (accumulator / weight_sum).astype(np.float32)
        else:
            update = np.zeros(bits.shape[1], dtype=np.float32)

    return PlaintextAggregationResult(
        update=np.ascontiguousarray(update),
        metrics=PlaintextAggregationMetrics(
            aggregation_seconds=time.perf_counter() - started,
            preprocessing_seconds=preprocessing_seconds,
            accepted_clients=accepted_clients,
            weight_sum=weight_sum,
            threshold_count=threshold,
            mismatch_mean=mismatch_mean,
            mismatch_min=mismatch_min,
            mismatch_max=mismatch_max,
        ),
    )
