"""Dense sign encoding used by RAIN clients."""

from __future__ import annotations

import numpy as np


def encode_signs(values: np.ndarray) -> np.ndarray:
    """Encode negative values as 0 and non-negative values as 1.

    Zero is deterministically mapped to +1, matching the paper's binary sign
    representation and avoiding a third symbol.
    """

    array = np.asarray(values)
    if array.size == 0:
        raise ValueError("values must not be empty")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("values must have a numeric dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError("values must be finite")
    return np.ascontiguousarray((array >= 0).astype(np.uint8))
