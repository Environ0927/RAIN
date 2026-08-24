"""Client helpers for splitting a batch across the two servers."""

from __future__ import annotations

import numpy as np

from rain.protocol.boolean_sharing import share_bits


def share_client_batch(
    bits: np.ndarray, *, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Boolean-share a client-by-coordinate bit matrix."""

    return share_bits(bits, rng=np.random.default_rng(seed))
