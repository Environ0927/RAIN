"""Analytical shuffled-Gaussian RDP accountant used by the RAIN artifact.

The implementation evaluates the multinomial upper bound of the shuffle
Gaussian mechanism (Liu et al., 2022) in log space.  It intentionally takes
the minimum with the ordinary local Gaussian RDP bound, so amplification can
never make the reported guarantee worse than releasing every local message.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


def _logaddexp(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def _log_convolve(a: list[float], b: list[float], degree: int) -> list[float]:
    out = [-math.inf] * (degree + 1)
    for i, av in enumerate(a):
        if av == -math.inf:
            continue
        upper = min(len(b) - 1, degree - i)
        for j in range(upper + 1):
            if b[j] != -math.inf:
                out[i + j] = _logaddexp(out[i + j], av + b[j])
    return out


def _log_polynomial_power(base: list[float], exponent: int, degree: int) -> list[float]:
    result = [0.0] + [-math.inf] * degree
    factor = base
    power = exponent
    while power:
        if power & 1:
            result = _log_convolve(result, factor, degree)
        power >>= 1
        if power:
            factor = _log_convolve(factor, factor, degree)
    return result


@dataclass(frozen=True, slots=True)
class PrivacyReport:
    epsilon: float
    delta: float
    optimal_order: int
    rounds: int
    participating_clients: int
    clip_norm: float
    noise_multiplier: float
    sensitivity: float
    orders: tuple[int, ...]
    per_round_rdp: tuple[float, ...]
    composed_rdp: tuple[float, ...]
    method: str = "analytical-shuffle-gaussian-rdp"

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["orders"] = list(self.orders)
        value["per_round_rdp"] = list(self.per_round_rdp)
        value["composed_rdp"] = list(self.composed_rdp)
        return value


def account_privacy(
    *,
    participating_clients: int,
    rounds: int,
    clip_norm: float,
    noise_multiplier: float,
    delta: float = 1e-5,
    orders: Iterable[int] = range(2, 129),
    sensitivity_factor: float = 2.0,
) -> PrivacyReport:
    """Compose the per-round shuffled Gaussian RDP upper bound.

    ``noise_multiplier`` is the standard deviation divided by ``clip_norm``.
    Replacement adjacency has sensitivity ``sensitivity_factor * clip_norm``;
    the conservative default is therefore ``2C``.
    """

    if not isinstance(participating_clients, int) or participating_clients < 1:
        raise ValueError("participating_clients must be a positive integer")
    if not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be finite and positive")
    if not math.isfinite(delta) or not 0 < delta < 1:
        raise ValueError("delta must lie strictly between zero and one")
    if not math.isfinite(sensitivity_factor) or sensitivity_factor <= 0:
        raise ValueError("sensitivity_factor must be finite and positive")

    order_tuple = tuple(int(x) for x in orders)
    if not order_tuple or any(x < 2 for x in order_tuple) or len(set(order_tuple)) != len(order_tuple):
        raise ValueError("orders must be unique integers of at least two")
    max_order = max(order_tuple)

    noise_std = noise_multiplier * clip_norm
    sensitivity = sensitivity_factor * clip_norm
    effective_sigma = noise_std / sensitivity
    inv_two_sigma_sq = 1.0 / (2.0 * effective_sigma * effective_sigma)

    # alpha! [x^alpha] (sum_k exp(k^2/(2 sigma^2))/k! x^k)^N
    log_base = [
        k * k * inv_two_sigma_sq - math.lgamma(k + 1)
        for k in range(max_order + 1)
    ]
    log_coeff = _log_polynomial_power(log_base, participating_clients, max_order)
    per_round: list[float] = []
    for alpha in order_tuple:
        log_multinomial_sum = math.lgamma(alpha + 1) + log_coeff[alpha]
        shuffled = (
            -alpha * inv_two_sigma_sq
            - alpha * math.log(participating_clients)
            + log_multinomial_sum
        ) / (alpha - 1)
        local = alpha * inv_two_sigma_sq
        per_round.append(max(0.0, min(local, shuffled)))

    composed = tuple(rounds * value for value in per_round)
    epsilons = tuple(
        rho + math.log(1.0 / delta) / (alpha - 1)
        for alpha, rho in zip(order_tuple, composed)
    )
    best_index = min(range(len(epsilons)), key=epsilons.__getitem__)
    return PrivacyReport(
        epsilon=epsilons[best_index],
        delta=delta,
        optimal_order=order_tuple[best_index],
        rounds=rounds,
        participating_clients=participating_clients,
        clip_norm=float(clip_norm),
        noise_multiplier=float(noise_multiplier),
        sensitivity=sensitivity,
        orders=order_tuple,
        per_round_rdp=tuple(per_round),
        composed_rdp=composed,
    )
