"""In-process logical transport with exact serialized-byte accounting."""

from __future__ import annotations

import numpy as np

from typing import TYPE_CHECKING

from .metrics import CommunicationMetrics, CommunicationSnapshot
from .serialization import array_to_messages, deserialize_message, serialize_message
from .types import Endpoint, Message, MessageKind, ProtocolPhase

if TYPE_CHECKING:
    from .integrity import TranscriptAuthenticator


class LogicalTransport:
    """Serialize every cross-party message and immediately deliver a decoded copy.

    This transport intentionally models neither sockets nor network latency.  Byte
    counts include the deterministic RAIN message header and payload, but no
    TCP/IP framing.
    """

    def __init__(self, *, transcript: "TranscriptAuthenticator | None" = None) -> None:
        self._metrics = CommunicationMetrics()
        self._transcript = transcript

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
            integrity_bytes = 0
            if (
                self._transcript is not None
                and sender in (Endpoint.S0, Endpoint.S1)
                and receiver in (Endpoint.S0, Endpoint.S1)
            ):
                integrity_bytes = self._transcript.protect_and_verify(
                    wire, sender=sender, receiver=receiver, kind=kind, phase=phase
                )
            self._metrics.record(
                wire_bytes=len(wire) + integrity_bytes,
                phase=phase,
                integrity_bytes=integrity_bytes,
                integrity_messages=1 if integrity_bytes else 0,
            )
            delivered.append(deserialize_message(wire))
        return delivered

    def record_integrity_control(
        self, *, wire_bytes: int, phase: ProtocolPhase, messages: int
    ) -> None:
        self._metrics.record_control(
            wire_bytes=wire_bytes, phase=phase, messages=messages
        )

    @property
    def metrics(self) -> CommunicationSnapshot:
        return self._metrics.snapshot()
