"""Reproducible mapping from the paper's local randomization index to noise."""

from __future__ import annotations

import math


def noise_multiplier_from_epsilon0(
    epsilon0: float,
    *,
    kappa: float = 1.0,
    sensitivity_multiplier: float = 2.0,
) -> float:
    """Return ``sigma / C`` for ``sigma = kappa * Delta_2 / epsilon0``.

    RAIN uses replacement adjacency, so ``Delta_2 <= 2 C`` and the default
    sensitivity multiplier is two.  ``epsilon0`` is a local randomization
    index; the formal end-to-end guarantee is still produced by the RDP
    accountant.
    """

    values = (float(epsilon0), float(kappa), float(sensitivity_multiplier))
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("epsilon0, kappa, and sensitivity_multiplier must be positive and finite")
    return values[1] * values[2] / values[0]
