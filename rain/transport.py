"""In-process logical transport with exact serialized-byte accounting."""

from __future__ import annotations

import numpy as np

from .metrics import CommunicationMetrics, CommunicationSnapshot
from .serialization import array_to_messages, deserialize_message, serialize_message
from .types import Endpoint, Message, MessageKind, ProtocolPhase


class LogicalTransport:
    """Serialize every cross-party message and immediately deliver a decoded copy.

    This transport intentionally models neither sockets nor network latency.  Byte
    counts include the deterministic RAIN message header and payload, but no
    TCP/IP framing.
    """

    def __init__(self) -> None:
        self._metrics = CommunicationMetrics()

    def send_array(
        self,
        array: np.ndarray,
        *,
        kind: MessageKind,
        phase: ProtocolPhase,
        sender: Endpoint,
        receiver: Endpoint,
        round_id: int,
        chunk_size: int,
        batch_id: int = 0,
        pack_bits: bool = False,
    ) -> list[Message]:
        outbound = array_to_messages(
            array,
            kind=kind,
            phase=phase,
            sender=sender,
            receiver=receiver,
            round_id=round_id,
            chunk_size=chunk_size,
            batch_id=batch_id,
            pack_bits=pack_bits,
        )
        delivered: list[Message] = []
        for message in outbound:
            wire = serialize_message(message)
            self._metrics.record(wire_bytes=len(wire), phase=phase)
            delivered.append(deserialize_message(wire))
        return delivered

    @property
    def metrics(self) -> CommunicationSnapshot:
        return self._metrics.snapshot()
