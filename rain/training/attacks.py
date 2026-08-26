"""Unified client-update attack interface.

Attacks run before randomization and sharing.  They cannot access protocol
shares, filtering masks, trust weights, or the aggregate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class AttackContext:
    malicious_clients: int
    seed: int
    target_direction: np.ndarray | None = None
    hamming_budget: int | None = None
    scale: float = 10.0
    noise_std: float = 0.0
    adversarial_direction: np.ndarray | None = None
    subspace_basis: np.ndarray | None = None


class UpdateAttack(Protocol):
    name: str
    def apply(self, updates: np.ndarray, context: AttackContext) -> np.ndarray: ...


def _inputs(updates: np.ndarray, context: AttackContext) -> tuple[np.ndarray, np.random.Generator]:
    value = np.asarray(updates)
    if value.dtype.kind != "f":
        value = value.astype(np.float32)
    if value.ndim != 2 or min(value.shape) < 1 or not np.all(np.isfinite(value)):
        raise ValueError("updates must be a non-empty finite matrix")
    if not 0 <= context.malicious_clients < value.shape[0]:
        raise ValueError("malicious_clients must be in [0, client_count)")
    return value.copy(), np.random.default_rng(context.seed)


class KrumAttack:
    name = "krum"
    def apply(self, updates: np.ndarray, context: AttackContext) -> np.ndarray:
        value, _ = _inputs(updates, context)
        if context.malicious_clients == 0:
            return value
        benign = value[context.malicious_clients:]
        centre = benign.mean(axis=0)
        direction = np.sign(centre); direction[direction == 0] = 1
        radius = np.min(np.linalg.norm(benign - centre, axis=1))
        value[:context.malicious_clients] = centre - direction * radius / np.sqrt(value.shape[1])
        return value


class MinMaxAttack:
    name = "min-max"
    def apply(self, updates: np.ndarray, context: AttackContext) -> np.ndarray:
        value, _ = _inputs(updates, context)
        f = context.malicious_clients
        if f == 0:
            return value
        benign = value[f:]
        centre = benign.mean(axis=0)
        deviation = centre / max(np.linalg.norm(centre), np.finfo(float).tiny)
        maximum = max(np.linalg.norm(x - y) for x in benign for y in benign)
        low, high = 0.0, 100.0
        for _ in range(60):
            mid = (low + high) / 2
            candidate = centre - mid * deviation
            if max(np.linalg.norm(candidate - row) for row in benign) <= maximum:
                low = mid
            else:
                high = mid
        value[:f] = centre - low * deviation
        return value


class ScalingBackdoorAttack:
    name = "scaling"
    def apply(self, updates: np.ndarray, context: AttackContext) -> np.ndarray:
        value, _ = _inputs(updates, context)
        value[:context.malicious_clients] *= context.scale
        return value


class AttackDPFL:
    name = "attack-dpfl"
    def apply(self, updates: np.ndarray, context: AttackContext) -> np.ndarray:
        value, rng = _inputs(updates, context)
        f = context.malicious_clients
        if f:
            benign = value[f:]
            benign_mean = benign.mean(axis=0)
            coordinate_std = (
                np.full(value.shape[1], context.noise_std, dtype=np.float64)
                if context.noise_std > 0
                else np.maximum(benign.std(axis=0), 1e-12)
            )
            poisoned = -benign_mean
            for row in range(f):
                su = poisoned + rng.normal(0, coordinate_std, size=value.shape[1])
                source_norm = max(float(np.linalg.norm(value[row])), np.finfo(float).tiny)
                value[row] *= float(np.linalg.norm(su)) / source_norm
        return value


def _target(context: AttackContext, dimension: int) -> np.ndarray:
    if context.target_direction is None:
        raise ValueError("this attack requires target_direction")
    target = np.asarray(context.target_direction, dtype=np.int8).reshape(-1)
    if target.shape != (dimension,) or np.any(~np.isin(target, (-1, 1))):
        raise ValueError("target_direction must contain -1/+1 and match dimension")
    return target


def _adversarial_signal(
    value: np.ndarray, context: AttackContext, reference: np.ndarray
) -> np.ndarray:
    if context.adversarial_direction is None:
        benign = value[context.malicious_clients:]
        signal = -benign.mean(axis=0)
    else:
        signal = np.asarray(context.adversarial_direction, dtype=np.float64).reshape(-1)
        if signal.shape != (value.shape[1],) or not np.all(np.isfinite(signal)):
            raise ValueError("adversarial_direction must be finite and match dimension")
    if context.subspace_basis is not None:
        basis = np.asarray(context.subspace_basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[0] != value.shape[1] or not np.all(np.isfinite(basis)):
            raise ValueError("subspace_basis must have shape (dimension, rank)")
        signal = basis @ (basis.T @ signal)
    return np.asarray(signal, dtype=np.float64)


def _coordinated_flip_set(
    signal: np.ndarray, reference: np.ndarray, budget: int
) -> np.ndarray:
    if budget == 0:
        return np.empty(0, dtype=np.int64)
    desired = np.where(signal >= 0, 1, -1)
    conflicting = np.flatnonzero((desired != reference) & (np.abs(signal) > 0))
    order = conflicting[np.argsort(-np.abs(signal[conflicting]), kind="stable")]
    if len(order) >= budget:
        return order[:budget]
    remaining = np.setdiff1d(np.arange(signal.size), order, assume_unique=False)
    fill = remaining[np.argsort(-np.abs(signal[remaining]), kind="stable")]
    return np.concatenate((order, fill[: budget - len(order)]))


class RAA:
    """Reference-aligned attack with an explicit sign Hamming budget."""
    name = "raa"
    def apply(self, updates: np.ndarray, context: AttackContext) -> np.ndarray:
        value, _ = _inputs(updates, context)
        reference = _target(context, value.shape[1])
        budget = context.hamming_budget
        if budget is None or not 0 <= budget <= value.shape[1]:
            raise ValueError("hamming_budget must lie in [0, dimension]")
        signal = _adversarial_signal(value, context, reference)
        candidates = _coordinated_flip_set(signal, reference, budget)
        for row in range(context.malicious_clients):
            keep = np.roll(candidates, row) if len(candidates) else candidates
            crafted = reference.copy()
            crafted[keep] *= -1
            value[row] = crafted * np.maximum(np.abs(value[row]), 1e-6)
        return value


class RSCA:
    """Randomized sign-constrained attack with an exact Hamming budget."""
    name = "rsca"
    def apply(self, updates: np.ndarray, context: AttackContext) -> np.ndarray:
        value, _ = _inputs(updates, context)
        reference = _target(context, value.shape[1])
        budget = context.hamming_budget
        if budget is None or not 0 <= budget <= value.shape[1]:
            raise ValueError("hamming_budget must lie in [0, dimension]")
        projected = _adversarial_signal(value, context, reference)
        coordinated = _coordinated_flip_set(projected, reference, budget)
        crafted = reference.copy()
        crafted[coordinated] *= -1
        for row in range(context.malicious_clients):
            value[row] = crafted * np.maximum(np.abs(value[row]), 1e-6)
        return value


class WOAA:
    """Weight-optimized adaptive attack from the current RAIN draft.

    For every Hamming radius ``k`` up to ``floor(d * tau)``, the optimal
    radius-k message flips the coordinates with the smallest
    ``reference[j] * adversarial_direction[j]``.  We evaluate those candidates
    incrementally and coordinate all compromised clients on the maximizer of

        tau - k / d + <s, v> / ||v||_1.

    ``hamming_budget`` is intentionally unused: unlike RAA, WOAA chooses its
    own radius.  ``AttackContext.scale`` carries the public RAIN threshold
    ``tau`` for this attack so the common attack interface remains compact.
    """

    name = "woaa"

    def apply(self, updates: np.ndarray, context: AttackContext) -> np.ndarray:
        value, _ = _inputs(updates, context)
        f = context.malicious_clients
        if f == 0:
            return value
        reference = _target(context, value.shape[1])
        tau = float(context.scale)
        if not np.isfinite(tau) or not 0 < tau <= 1:
            raise ValueError("WOAA requires tau in (0, 1] through AttackContext.scale")
        signal = _adversarial_signal(value, context, reference)
        norm_one = float(np.linalg.norm(signal, ord=1))
        if norm_one <= np.finfo(float).tiny:
            crafted = reference.copy()
        else:
            products = reference.astype(np.float64) * signal
            flip_order = np.argsort(products, kind="stable")
            max_radius = min(value.shape[1], int(np.floor(value.shape[1] * tau)))
            # At radius k, the directional dot product is the reference score
            # minus twice the sum of the first k sorted products.
            reference_score = float(reference.astype(np.float64) @ signal)
            cumulative = np.concatenate((
                np.array([0.0], dtype=np.float64),
                np.cumsum(products[flip_order[:max_radius]], dtype=np.float64),
            ))
            radii = np.arange(max_radius + 1, dtype=np.float64)
            objectives = (
                tau - radii / value.shape[1]
                + (reference_score - 2.0 * cumulative) / norm_one
            )
            best_radius = int(np.argmax(objectives))
            crafted = reference.copy()
            crafted[flip_order[:best_radius]] *= -1
        for row in range(f):
            value[row] = crafted * np.maximum(np.abs(value[row]), 1e-6)
        return value


ATTACKS = {attack.name: attack for attack in (
    KrumAttack(), MinMaxAttack(), ScalingBackdoorAttack(), AttackDPFL(), RAA(), WOAA(),
    # RSCA belongs to the preceding draft and remains available so existing
    # result manifests stay reproducible.
    RSCA(),
)}


def apply_attack(name: str, updates: np.ndarray, context: AttackContext) -> np.ndarray:
    try:
        attack = ATTACKS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown attack: {name}") from exc
    return attack.apply(updates, context)
