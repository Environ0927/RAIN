"""Communication and local-computation metrics for the simulator."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .types import ProtocolPhase


@dataclass(frozen=True, slots=True)
class CommunicationSnapshot:
    client_to_server_bytes: int
    server_to_server_bytes: int
    offline_bytes: int
    online_bytes: int
    message_count: int
    integrity_bytes: int = 0
    integrity_message_count: int = 0

    @property
    def total_bytes(self) -> int:
        return self.client_to_server_bytes + self.server_to_server_bytes

    def as_dict(self) -> dict[str, int]:
        result = asdict(self)
        result["total_bytes"] = self.total_bytes
        return result


class CommunicationMetrics:
    """Mutable byte counters updated only from serialized wire bytes."""

    def __init__(self) -> None:
        self._client_to_server_bytes = 0
        self._server_to_server_bytes = 0
        self._offline_bytes = 0
        self._online_bytes = 0
        self._message_count = 0
        self._integrity_bytes = 0
        self._integrity_message_count = 0

    def record(
        self, *, wire_bytes: int, phase: ProtocolPhase,
        integrity_bytes: int = 0, integrity_messages: int = 0,
    ) -> None:
        if wire_bytes <= 0:
            raise ValueError("wire_bytes must be positive")
        if integrity_bytes < 0 or integrity_bytes > wire_bytes:
            raise ValueError("integrity_bytes must lie in [0, wire_bytes]")
        if integrity_messages < 0:
            raise ValueError("integrity_messages must be non-negative")
        if phase is ProtocolPhase.CLIENT:
            self._client_to_server_bytes += wire_bytes
        else:
            self._server_to_server_bytes += wire_bytes
            if phase is ProtocolPhase.OFFLINE:
                self._offline_bytes += wire_bytes
            elif phase is ProtocolPhase.ONLINE:
                self._online_bytes += wire_bytes
        self._message_count += 1
        self._integrity_bytes += integrity_bytes
        self._integrity_message_count += integrity_messages

    def record_control(
        self, *, wire_bytes: int, phase: ProtocolPhase, messages: int = 1
    ) -> None:
        if messages < 1:
            raise ValueError("messages must be positive")
        self.record(
            wire_bytes=wire_bytes, phase=phase,
            integrity_bytes=wire_bytes, integrity_messages=messages,
        )
        # record() counts one logical envelope; control records may represent
        # multiple fixed-size tags/digests.
        self._message_count += messages - 1

    def snapshot(self) -> CommunicationSnapshot:
        return CommunicationSnapshot(
            client_to_server_bytes=self._client_to_server_bytes,
            server_to_server_bytes=self._server_to_server_bytes,
            offline_bytes=self._offline_bytes,
            online_bytes=self._online_bytes,
            message_count=self._message_count,
            integrity_bytes=self._integrity_bytes,
            integrity_message_count=self._integrity_message_count,
        )


@dataclass(frozen=True, slots=True)
class ComputationSnapshot:
    client_seconds: float
    s0_seconds: float
    s1_seconds: float
    offline_seconds: float
    online_seconds: float

    @property
    def server_sum_seconds(self) -> float:
        return self.s0_seconds + self.s1_seconds

    @property
    def server_critical_seconds(self) -> float:
        """Parallel-party critical-path estimate from local party totals."""

        return max(self.s0_seconds, self.s1_seconds)

    def as_dict(self) -> dict[str, float]:
        result = asdict(self)
        result["server_sum_seconds"] = self.server_sum_seconds
        result["server_critical_seconds"] = self.server_critical_seconds
        return result
