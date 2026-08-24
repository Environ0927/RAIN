"""Arithmetic sharing over the 64-bit ring Z/(2**64)Z.

Signed values use standard two's-complement encoding.  The ring was selected so
that local addition/subtraction is native and arithmetic-to-Boolean conversion
can use a fixed 64-bit GMW adder.  Protocol configurations must keep all true
intermediate magnitudes below 2**63; :func:`validate_aggregation_bounds`
enforces the RAIN-specific bound before a round starts.
"""

from __future__ import annotations

import numpy as np

from rain.errors import ShareValidationError


RING_BITS = 64
MAX_SIGNED = (1 << 63) - 1


def _as_u64(value: np.ndarray, *, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint64:
        raise ShareValidationError(f"{name} must be a numpy uint64 array")
    if value.size == 0:
        raise ShareValidationError(f"{name} must not be empty")
    return np.ascontiguousarray(value)


def encode_signed(value: np.ndarray | list[int] | int) -> np.ndarray:
    signed = np.asarray(value, dtype=np.int64)
    return np.ascontiguousarray(signed.view(np.uint64))


def decode_signed(value: np.ndarray) -> np.ndarray:
    return _as_u64(value, name="encoded value").view(np.int64).copy()


def share_arithmetic(
    value: np.ndarray, *, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    secret = _as_u64(value, name="secret")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be numpy.random.Generator")
    share0 = rng.integers(
        0, np.iinfo(np.uint64).max, size=secret.shape, endpoint=True, dtype=np.uint64
    )
    share1 = np.subtract(secret, share0, dtype=np.uint64)
    return np.ascontiguousarray(share0), np.ascontiguousarray(share1)


def reconstruct_arithmetic(share0: np.ndarray, share1: np.ndarray) -> np.ndarray:
    left = _as_u64(share0, name="share0")
    right = _as_u64(share1, name="share1")
    if left.shape != right.shape:
        raise ShareValidationError("arithmetic share shapes must match")
    return np.add(left, right, dtype=np.uint64)


def validate_aggregation_bounds(*, clients: int, dimension: int) -> None:
    if clients <= 0 or dimension <= 0:
        raise ValueError("clients and dimension must be positive")
    # 0 <= weight <= T <= d and |weighted sum| <= clients*d.
    if clients * dimension > MAX_SIGNED:
        raise OverflowError("clients * dimension exceeds signed Z_2^64 range")
