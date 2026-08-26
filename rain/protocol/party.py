"""Strictly separated local state machines for servers S0 and S1."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np

from rain.errors import ProtocolError
from rain.serialization import messages_to_array
from rain.types import Endpoint, Message, MessageKind, ProtocolPhase

from .boolean_sharing import validate_bits


_T = TypeVar("_T")


@dataclass(slots=True)
class Server0State:
    """Private local view of S0.  It never contains S1's permutation or shares."""

    round_id: int
    batch_id: int
    shape: tuple[int, int]
    permutation: np.ndarray
    a0: np.ndarray
    c0: np.ndarray
    input_share: np.ndarray | None = None
    output_share: np.ndarray | None = None
    offline_step: str = "complete"
    online_step: str = "ready"
    offline_seconds: float = 0.0
    online_seconds: float = 0.0


@dataclass(slots=True)
class Server1State:
    """Private local view of S1.  It never contains S0's permutation or shares."""

    round_id: int
    batch_id: int
    shape: tuple[int, int]
    permutation: np.ndarray
    a1: np.ndarray
    delta: np.ndarray
    input_share: np.ndarray | None = None
    output_share: np.ndarray | None = None
    offline_step: str = "complete"
    online_step: str = "ready"
    offline_seconds: float = 0.0
    online_seconds: float = 0.0


class _Party:
    __slots__ = ()
    endpoint: Endpoint

    @staticmethod
    def _measure(callback: Callable[[], _T]) -> tuple[_T, float]:
        started = time.perf_counter()
        result = callback()
        return result, time.perf_counter() - started


class Server0(_Party):
    """S0 endpoint.  Methods can inspect only a :class:`Server0State`."""

    endpoint = Endpoint.S0
    __slots__ = ("__state",)

    def __init__(self, state: Server0State) -> None:
        if not isinstance(state, Server0State):
            raise TypeError("Server0 requires Server0State")
        self.__state = state

    @property
    def round_id(self) -> int:
        return self.__state.round_id

    @property
    def batch_id(self) -> int:
        return self.__state.batch_id

    @property
    def shape(self) -> tuple[int, int]:
        return self.__state.shape

    @property
    def offline_seconds(self) -> float:
        return self.__state.offline_seconds

    @property
    def online_seconds(self) -> float:
        return self.__state.online_seconds

    def accept_client_share(self, messages: list[Message]) -> None:
        state = self.__state
        if state.input_share is not None:
            raise ProtocolError("S0 already accepted a client share for this round")
        state.input_share = messages_to_array(
            messages,
            expected_kind=MessageKind.CLIENT_SHARE,
            expected_phase=ProtocolPhase.CLIENT,
            expected_sender=Endpoint.CLIENT,
            expected_receiver=Endpoint.S0,
            expected_round_id=state.round_id,
            expected_batch_id=state.batch_id,
            expected_shape=state.shape,
            require_bits=True,
        )

    def receive_online_z1(self, messages: list[Message]) -> np.ndarray:
        state = self.__state
        if state.offline_step != "complete":
            raise ProtocolError("S0 cannot start online shuffle before preprocessing")
        if state.input_share is None:
            raise ProtocolError("S0 has no client input share")
        if state.online_step != "ready":
            raise ProtocolError(f"S0 online step out of order: {state.online_step}")

        def compute() -> np.ndarray:
            z1 = messages_to_array(
                messages,
                expected_kind=MessageKind.ONLINE_Z1,
                expected_phase=ProtocolPhase.ONLINE,
                expected_sender=Endpoint.S1,
                expected_receiver=Endpoint.S0,
                expected_round_id=state.round_id,
                expected_batch_id=state.batch_id,
                expected_shape=state.shape,
                require_bits=True,
            )
            combined = np.bitwise_xor(z1, state.input_share)
            z0 = np.bitwise_xor(combined[state.permutation], state.a0)
            state.output_share = state.c0.copy()
            return np.ascontiguousarray(z0)

        value, elapsed = self._measure(compute)
        state.online_seconds += elapsed
        state.online_step = "complete"
        return value

    def output_share(self) -> np.ndarray:
        if self.__state.output_share is None:
            raise ProtocolError("S0 output share is not available")
        return self.__state.output_share.copy()


class Server1(_Party):
    """S1 endpoint.  Methods can inspect only a :class:`Server1State`."""

    endpoint = Endpoint.S1
    __slots__ = ("__state",)

    def __init__(self, state: Server1State) -> None:
        if not isinstance(state, Server1State):
            raise TypeError("Server1 requires Server1State")
        self.__state = state

    @property
    def round_id(self) -> int:
        return self.__state.round_id

    @property
    def batch_id(self) -> int:
        return self.__state.batch_id

    @property
    def shape(self) -> tuple[int, int]:
        return self.__state.shape

    @property
    def offline_seconds(self) -> float:
        return self.__state.offline_seconds

    @property
    def online_seconds(self) -> float:
        return self.__state.online_seconds

    def accept_client_share(self, messages: list[Message]) -> None:
        state = self.__state
        if state.input_share is not None:
            raise ProtocolError("S1 already accepted a client share for this round")
        state.input_share = messages_to_array(
            messages,
            expected_kind=MessageKind.CLIENT_SHARE,
            expected_phase=ProtocolPhase.CLIENT,
            expected_sender=Endpoint.CLIENT,
            expected_receiver=Endpoint.S1,
            expected_round_id=state.round_id,
            expected_batch_id=state.batch_id,
            expected_shape=state.shape,
            require_bits=True,
        )

    def start_online(self) -> np.ndarray:
        state = self.__state
        if state.offline_step != "complete":
            raise ProtocolError("S1 cannot start online shuffle before preprocessing")
        if state.input_share is None:
            raise ProtocolError("S1 has no client input share")
        if state.online_step != "ready":
            raise ProtocolError(f"S1 online step out of order: {state.online_step}")

        def compute() -> np.ndarray:
            return np.ascontiguousarray(np.bitwise_xor(state.input_share, state.a1))

        value, elapsed = self._measure(compute)
        state.online_seconds += elapsed
        state.online_step = "sent_z1"
        return value

    def receive_online_z0(self, messages: list[Message]) -> None:
        state = self.__state
        if state.online_step != "sent_z1":
            raise ProtocolError(f"S1 online step out of order: {state.online_step}")

        def compute() -> np.ndarray:
            z0 = messages_to_array(
                messages,
                expected_kind=MessageKind.ONLINE_Z0,
                expected_phase=ProtocolPhase.ONLINE,
                expected_sender=Endpoint.S0,
                expected_receiver=Endpoint.S1,
                expected_round_id=state.round_id,
                expected_batch_id=state.batch_id,
                expected_shape=state.shape,
                require_bits=True,
            )
            return np.ascontiguousarray(
                np.bitwise_xor(z0[state.permutation], state.delta)
            )

        state.output_share, elapsed = self._measure(compute)
        state.online_seconds += elapsed
        state.online_step = "complete"

    def output_share(self) -> np.ndarray:
        if self.__state.output_share is None:
            raise ProtocolError("S1 output share is not available")
        return self.__state.output_share.copy()
