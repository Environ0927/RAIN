"""Analytical Approx+Shuffling bound used by the paper comparison.

This module implements Theorem III.1 of "Encode, Shuffle, Analyze Privacy
Revisited" (arXiv:2001.03618).  The bound is for removal DP of
attribute-fragmented k-RAPPOR and is deliberately exposed with its theorem
domain.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ApproxShufflingReport:
    clients: int
    local_epsilon: float
    delta: float
    central_epsilon: float
    adjacency: str = "removal"
    mechanism: str = "attribute-fragmented-k-rappor"
    theorem: str = "Theorem III.1, arXiv:2001.03618"

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def theorem_domain(clients: int, local_epsilon: float, delta: float) -> tuple[bool, str]:
    if clients < 2:
        return False, "clients must be at least 2"
    if not math.isfinite(local_epsilon) or local_epsilon < 1.0:
        return False, "local_epsilon must be finite and at least 1"
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        return False, "delta must lie strictly between 0 and 1"
    minimum_delta = math.exp(-(math.log(clients) ** 2))
    if delta < minimum_delta:
        return False, f"delta must be at least n^(-log(n))={minimum_delta:.12g}"
    maximum_local = math.log(clients) - math.log(14.0 * math.log(4.0 / delta))
    if local_epsilon > maximum_local:
        return False, f"local_epsilon exceeds theorem maximum {maximum_local:.12g}"
    return True, "valid"


def approximate_shuffling_epsilon(
    *, clients: int, local_epsilon: float, delta: float,
) -> ApproxShufflingReport:
    """Return the cited central removal-DP epsilon without extrapolation."""

    valid, reason = theorem_domain(clients, local_epsilon, delta)
    if not valid:
        raise ValueError(f"parameters lie outside Theorem III.1: {reason}")
    central = math.sqrt(
        64.0 * math.exp(local_epsilon) * math.log(4.0 / delta) / clients
    )
    return ApproxShufflingReport(
        clients=int(clients), local_epsilon=float(local_epsilon),
        delta=float(delta), central_epsilon=central,
    )
