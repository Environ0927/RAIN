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
            benign_mean = value[f:].mean(axis=0)
            std = np.maximum(value[f:].std(axis=0), 1e-12)
            value[:f] = -benign_mean + rng.normal(0, std, size=(f, value.shape[1]))
        return value


def _target(context: AttackContext, dimension: int) -> np.ndarray:
    if context.target_direction is None:
        raise ValueError("this attack requires target_direction")
    target = np.asarray(context.target_direction, dtype=np.int8).reshape(-1)
    if target.shape != (dimension,) or np.any(~np.isin(target, (-1, 1))):
        raise ValueError("target_direction must contain -1/+1 and match dimension")
    return target


class RAA:
    """Reference-aligned attack with an explicit sign Hamming budget."""
    name = "raa"
    def apply(self, updates: np.ndarray, context: AttackContext) -> np.ndarray:
        value, _ = _inputs(updates, context)
        target = _target(context, value.shape[1])
        budget = context.hamming_budget
        if budget is None or not 0 <= budget <= value.shape[1]:
            raise ValueError("hamming_budget must lie in [0, dimension]")
        for row in range(context.malicious_clients):
            current = np.where(value[row] >= 0, 1, -1)
            differing = np.flatnonzero(current != target)
            keep = differing[:budget]
            crafted = target.copy()
            crafted[keep] *= -1
            value[row] = crafted * np.maximum(np.abs(value[row]), 1e-6)
        return value


class RSCA:
    """Randomized sign-constrained attack with an exact Hamming budget."""
    name = "rsca"
    def apply(self, updates: np.ndarray, context: AttackContext) -> np.ndarray:
        value, rng = _inputs(updates, context)
        target = _target(context, value.shape[1])
        budget = context.hamming_budget
        if budget is None or not 0 <= budget <= value.shape[1]:
            raise ValueError("hamming_budget must lie in [0, dimension]")
        for row in range(context.malicious_clients):
            crafted = target.copy()
            if budget:
                crafted[rng.choice(value.shape[1], budget, replace=False)] *= -1
            value[row] = crafted * np.maximum(np.abs(value[row]), 1e-6)
        return value


ATTACKS = {attack.name: attack for attack in (
    KrumAttack(), MinMaxAttack(), ScalingBackdoorAttack(), AttackDPFL(), RAA(), RSCA()
)}


def apply_attack(name: str, updates: np.ndarray, context: AttackContext) -> np.ndarray:
    try:
        attack = ATTACKS[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unknown attack: {name}") from exc
    return attack.apply(updates, context)
