"""Offline shuffle-correlation setup for the Phase 1 simulator.

The paper requires the correlation

    delta = pi1(pi0(a1) XOR a0) XOR c0

while neither online server learns the other party's masks or permutation.  The
underlying Clarion construction realizes this setup with OT/PRG-based share
translation.  Phase 1 models that expensive setup as an explicit ideal
functionality: it receives no client data, returns only each party's prescribed
local view, and is never called by the online protocol as a fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .party import Server0, Server0State, Server1, Server1State


def validate_permutation(permutation: np.ndarray, size: int, *, name: str) -> np.ndarray:
    if not isinstance(permutation, np.ndarray) or permutation.ndim != 1:
        raise ValueError(f"{name} must be a 1-D numpy array")
    if permutation.dtype.kind not in "iu":
        raise ValueError(f"{name} must have an integer dtype")
    if permutation.shape != (size,) or not np.array_equal(np.sort(permutation), np.arange(size)):
        raise ValueError(f"{name} must be a permutation of range({size})")
    return np.ascontiguousarray(permutation, dtype=np.int64)


def _positive_shape(batch_size: int, dimension: int) -> tuple[int, int]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    return batch_size, dimension


def _valid_round_id(round_id: int) -> int:
    if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id < 0:
        raise ValueError("round_id must be a non-negative integer")
    return round_id


def _random_bits(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    return rng.integers(0, 2, size=shape, dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class PreprocessedParties:
    server0: Server0
    server1: Server1
    elapsed_seconds: float


class IdealShufflePreprocessor:
    """Explicit ideal functionality for offline-only correlated randomness.

    This object is part of the protocol simulator's setup model, not a third
    online server.  Replacing it with an OT/PRG backend does not change the
    ``Server0``/``Server1`` or online shuffle interfaces.
    """

    def prepare(
        self,
        *,
        batch_size: int,
        dimension: int,
        round_id: int,
        batch_id: int = 0,
        server0_seed: int | None = None,
        server1_seed: int | None = None,
        permutation0: np.ndarray | None = None,
        permutation1: np.ndarray | None = None,
    ) -> PreprocessedParties:
        started = time.perf_counter()
        shape = _positive_shape(batch_size, dimension)
        round_id = _valid_round_id(round_id)
        batch_id = _valid_round_id(batch_id)
        rng0 = np.random.default_rng(server0_seed)
        rng1 = np.random.default_rng(server1_seed)
        pi0 = (
            rng0.permutation(batch_size).astype(np.int64)
            if permutation0 is None
            else validate_permutation(permutation0, batch_size, name="S0 permutation")
        )
        pi1 = (
            rng1.permutation(batch_size).astype(np.int64)
            if permutation1 is None
            else validate_permutation(permutation1, batch_size, name="S1 permutation")
        )
        a0 = _random_bits(rng0, shape)
        c0 = _random_bits(rng0, shape)
        a1 = _random_bits(rng1, shape)
        delta = np.bitwise_xor(np.bitwise_xor(a1[pi0], a0)[pi1], c0)

        server0 = Server0(
            Server0State(
                round_id=round_id,
                batch_id=batch_id,
                shape=shape,
                permutation=pi0,
                a0=a0,
                c0=c0,
            )
        )
        server1 = Server1(
            Server1State(
                round_id=round_id,
                batch_id=batch_id,
                shape=shape,
                permutation=pi1,
                a1=a1,
                delta=np.ascontiguousarray(delta),
            )
        )
        return PreprocessedParties(
            server0=server0,
            server1=server1,
            elapsed_seconds=time.perf_counter() - started,
        )


class ShufflePreprocessingCache:
    """One-time cache for preprocessing rounds generated before online use."""

    def __init__(self, backend: IdealShufflePreprocessor | None = None) -> None:
        self._backend = backend or IdealShufflePreprocessor()
        self._items: dict[tuple[int, int], PreprocessedParties] = {}

    def prepare(self, **kwargs) -> None:
        key = (int(kwargs["round_id"]), int(kwargs.get("batch_id", 0)))
        if key in self._items:
            raise ValueError(f"preprocessing already cached for round/batch {key}")
        self._items[key] = self._backend.prepare(**kwargs)

    def consume(self, *, round_id: int, batch_id: int = 0) -> PreprocessedParties:
        key = (round_id, batch_id)
        try:
            return self._items.pop(key)
        except KeyError as exc:
            raise ValueError(f"no cached preprocessing for round/batch {key}") from exc

    def __len__(self) -> int:
        return len(self._items)
