"""Wire-level types shared by the protocol simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


PROTOCOL_VERSION = 1


class Endpoint(IntEnum):
    CLIENT = 0
    S0 = 1
    S1 = 2


class MessageKind(IntEnum):
    CLIENT_SHARE = 1
    ONLINE_Z1 = 2
    ONLINE_Z0 = 3
    BOOLEAN_OPEN = 10
    ARITHMETIC_OPEN = 11
    BOOLEAN_INPUT_SHARE = 12


class ProtocolPhase(IntEnum):
    CLIENT = 0
    OFFLINE = 1
    ONLINE = 2


@dataclass(frozen=True, slots=True)
class Message:
    """One deterministic wire message containing a 2-D array chunk."""

    version: int
    kind: MessageKind
    phase: ProtocolPhase
    sender: Endpoint
    receiver: Endpoint
    round_id: int
    batch_id: int
    chunk_id: int
    total_chunks: int
    rows: int
    cols: int
    dtype: str
    payload: bytes
