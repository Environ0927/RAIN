"""Boolean XOR secret sharing."""

from __future__ import annotations

import numpy as np

from rain.errors import ShareValidationError


def validate_bits(
    value: np.ndarray,
    *,
    name: str = "bits",
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ShareValidationError(f"{name} must be a numpy array")
    if value.dtype != np.uint8:
        raise ShareValidationError(f"{name} must have dtype uint8")
    if value.size == 0 or any(size <= 0 for size in value.shape):
        raise ShareValidationError(f"{name} must not be empty")
    if expected_shape is not None and value.shape != expected_shape:
        raise ShareValidationError(
            f"{name} has shape {value.shape}; expected {expected_shape}"
        )
    if np.any(value > 1):
        raise ShareValidationError(f"{name} must contain only bits 0 or 1")
    return np.ascontiguousarray(value)


def share_bits(
    bits: np.ndarray, *, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    plaintext = validate_bits(bits, name="plaintext bits")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    share0 = rng.integers(0, 2, size=plaintext.shape, dtype=np.uint8)
    share1 = np.bitwise_xor(plaintext, share0)
    return np.ascontiguousarray(share0), np.ascontiguousarray(share1)


def reconstruct_bits(share0: np.ndarray, share1: np.ndarray) -> np.ndarray:
    left = validate_bits(share0, name="share0")
    right = validate_bits(share1, name="share1", expected_shape=left.shape)
    return np.ascontiguousarray(np.bitwise_xor(left, right))
