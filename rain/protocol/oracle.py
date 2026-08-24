"""Plaintext correctness oracle; never called by the protocol path."""

from __future__ import annotations

import numpy as np

from .boolean_sharing import validate_bits
from .preprocessing import validate_permutation


def apply_permutation(bits: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    matrix = validate_bits(bits, name="bits")
    if matrix.ndim != 2:
        raise ValueError("bits must be a 2-D matrix")
    order = validate_permutation(permutation, matrix.shape[0], name="permutation")
    return np.ascontiguousarray(matrix[order])


def plaintext_shuffle_oracle(
    bits: np.ndarray, permutation0: np.ndarray, permutation1: np.ndarray
) -> np.ndarray:
    """Return pi1(pi0(bits)) for tests and reviewer demos only."""

    return apply_permutation(apply_permutation(bits, permutation0), permutation1)


def rain_aggregate_oracle(
    bits: np.ndarray, reference_bits: np.ndarray, *, tau: float
) -> np.ndarray:
    """Plaintext Algorithm 1/3 direction oracle for tests only."""

    matrix = validate_bits(bits, name="bits")
    if matrix.ndim != 2:
        raise ValueError("bits must be a 2-D matrix")
    reference = validate_bits(reference_bits, name="reference_bits").reshape(-1)
    if reference.shape != (matrix.shape[1],):
        raise ValueError("reference dimension does not match bits")
    if not np.isfinite(tau) or not 0 < tau < 0.5:
        raise ValueError("tau must be in (0, 0.5)")
    threshold = int(np.floor(matrix.shape[1] * tau))
    if threshold <= 0:
        raise ValueError("tau is too small for this dimension")
    distances = np.count_nonzero(matrix != reference[None, :], axis=1)
    weights = np.maximum(0, threshold - distances).astype(np.int64)
    signs = matrix.astype(np.int64) * 2 - 1
    weighted = np.sum(weights[:, None] * signs, axis=0)
    return np.ascontiguousarray((weighted >= 0).astype(np.uint8).reshape(1, -1))
