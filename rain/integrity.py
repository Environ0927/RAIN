"""RAIN-MAC aggregation-input integrity for the two-server simulator.

The layer implements the boundary described in Section 5.4 of the paper.  It
authenticates every client share in its session/round/batch/server context,
rejects duplicate tickets, requires both servers to accept the same canonical
ticket set, and authenticates the ordered server-to-server transcript.

This is intentionally not malicious-secure MPC: a server that owns a valid
session key can still authenticate a semantically incorrect internal value.
"""

from __future__ import annotations

import hashlib
import hmac
import struct
from dataclasses import dataclass

import numpy as np

from rain.errors import IntegrityError
from rain.types import Endpoint, MessageKind, ProtocolPhase


MAC_BYTES = 32
TICKET_BYTES = 16
_ZERO_DIGEST = bytes(MAC_BYTES)


def _field(value: bytes) -> bytes:
    return struct.pack("<I", len(value)) + value


def _u64(value: int) -> bytes:
    if value < 0:
        raise ValueError("integrity context integers must be non-negative")
    return struct.pack("<Q", value)


def _derive(master: bytes, label: bytes) -> bytes:
    return hmac.new(master, b"RAIN-KDF-v1" + _field(label), hashlib.sha256).digest()


@dataclass(frozen=True, slots=True)
class IntegrityContext:
    session_id: str
    round_id: int
    batch_id: int

    def encode(self) -> bytes:
        if not self.session_id:
            raise ValueError("session_id must not be empty")
        return b"RAIN-CTX-v1" + _field(self.session_id.encode("utf-8")) + _u64(
            self.round_id
        ) + _u64(self.batch_id)


@dataclass(frozen=True, slots=True)
class AuthenticatedShareBatch:
    server: Endpoint
    shares: np.ndarray
    tickets: tuple[bytes, ...]
    tags: tuple[bytes, ...]


@dataclass(frozen=True, slots=True)
class VerifiedClientBatch:
    server0_shares: np.ndarray
    server1_shares: np.ndarray
    sorted_tickets: tuple[bytes, ...]
    digest: bytes
    client_auth_bytes: int
    batch_exchange_bytes: int


@dataclass(frozen=True, slots=True)
class TranscriptEnvelope:
    sender: Endpoint
    receiver: Endpoint
    kind: MessageKind
    phase: ProtocolPhase
    sequence: int
    previous_digest: bytes
    tag: bytes


class TranscriptAuthenticator:
    """Maintain identical authenticated transcript views for S0 and S1."""

    def __init__(self, *, context: IntegrityContext, key: bytes) -> None:
        if len(key) < 16:
            raise ValueError("transcript key must contain at least 16 bytes")
        self._context = context.encode()
        self._key = bytes(key)
        self._digests = {Endpoint.S0: _ZERO_DIGEST, Endpoint.S1: _ZERO_DIGEST}
        self._sequence = 0
        self._message_count = 0
        self._wire_overhead = 0

    def _message(
        self,
        wire: bytes,
        *,
        sender: Endpoint,
        receiver: Endpoint,
        kind: MessageKind,
        phase: ProtocolPhase,
        sequence: int,
        previous: bytes,
    ) -> bytes:
        return (
            b"RAIN-TRANSCRIPT-v1"
            + self._context
            + bytes((int(sender), int(receiver), int(kind), int(phase)))
            + _u64(sequence)
            + previous
            + _field(wire)
        )

    def create_envelope(
        self,
        wire: bytes,
        *,
        sender: Endpoint,
        receiver: Endpoint,
        kind: MessageKind,
        phase: ProtocolPhase,
    ) -> TranscriptEnvelope:
        if {sender, receiver} != {Endpoint.S0, Endpoint.S1}:
            raise IntegrityError("transcript authentication is only for S0/S1 traffic")
        previous = self._digests[sender]
        message = self._message(
            wire, sender=sender, receiver=receiver, kind=kind, phase=phase,
            sequence=self._sequence, previous=previous,
        )
        return TranscriptEnvelope(
            sender=sender,
            receiver=receiver,
            kind=kind,
            phase=phase,
            sequence=self._sequence,
            previous_digest=previous,
            tag=hmac.new(self._key, message, hashlib.sha256).digest(),
        )

    def verify_and_commit(self, wire: bytes, envelope: TranscriptEnvelope) -> int:
        if envelope.sequence != self._sequence:
            raise IntegrityError("transcript sequence mismatch")
        sender_digest = self._digests[envelope.sender]
        receiver_digest = self._digests[envelope.receiver]
        if envelope.previous_digest != sender_digest or sender_digest != receiver_digest:
            raise IntegrityError("transcript previous-hash mismatch")
        message = self._message(
            wire,
            sender=envelope.sender,
            receiver=envelope.receiver,
            kind=envelope.kind,
            phase=envelope.phase,
            sequence=envelope.sequence,
            previous=envelope.previous_digest,
        )
        expected = hmac.new(self._key, message, hashlib.sha256).digest()
        if not hmac.compare_digest(expected, envelope.tag):
            raise IntegrityError("transcript MAC verification failed")
        digest = hashlib.sha256(
            envelope.previous_digest + message + envelope.tag
        ).digest()
        self._digests[Endpoint.S0] = digest
        self._digests[Endpoint.S1] = digest
        self._sequence += 1
        self._message_count += 1
        # seq || previous digest || HMAC tag are additional to the base wire.
        overhead = 8 + MAC_BYTES + MAC_BYTES
        self._wire_overhead += overhead
        return overhead

    def protect_and_verify(
        self,
        wire: bytes,
        *,
        sender: Endpoint,
        receiver: Endpoint,
        kind: MessageKind,
        phase: ProtocolPhase,
    ) -> int:
        envelope = self.create_envelope(
            wire, sender=sender, receiver=receiver, kind=kind, phase=phase
        )
        return self.verify_and_commit(wire, envelope)

    @property
    def final_digest(self) -> bytes:
        if self._digests[Endpoint.S0] != self._digests[Endpoint.S1]:
            raise IntegrityError("server transcript digests do not match")
        return self._digests[Endpoint.S0]

    @property
    def message_count(self) -> int:
        return self._message_count

    @property
    def wire_overhead(self) -> int:
        return self._wire_overhead


class RainMacSession:
    """Create and verify one round's client-share and transcript MACs."""

    def __init__(self, *, context: IntegrityContext, seed: int) -> None:
        self.context = context
        master = hashlib.sha256(b"RAIN-MAC-SEED-v1" + _u64(int(seed))).digest()
        self._master = master
        self.transcript = TranscriptAuthenticator(
            context=context, key=_derive(master, b"server-transcript")
        )

    def _client_key(self, client: int, server: Endpoint) -> bytes:
        return _derive(
            self._master,
            b"client-share" + _u64(client) + bytes((int(server),)),
        )

    def _share_message(
        self, *, client: int, server: Endpoint, ticket: bytes, share: np.ndarray
    ) -> bytes:
        row = np.asarray(share, dtype=np.uint8)
        if row.ndim != 1 or row.size < 1 or np.any(row > 1):
            raise IntegrityError("authenticated client share must be a non-empty bit vector")
        if len(ticket) != TICKET_BYTES:
            raise IntegrityError("client ticket has an invalid length")
        return (
            b"RAIN-SHARE-v1"
            + self.context.encode()
            + _u64(client)
            + bytes((int(server),))
            + ticket
            + _u64(row.size)
            + row.tobytes(order="C")
        )

    def authenticate_client_shares(
        self, share0: np.ndarray, share1: np.ndarray, *, seed: int
    ) -> tuple[AuthenticatedShareBatch, AuthenticatedShareBatch]:
        left = np.asarray(share0, dtype=np.uint8)
        right = np.asarray(share1, dtype=np.uint8)
        if left.ndim != 2 or left.shape != right.shape or min(left.shape) < 1:
            raise IntegrityError("the two client-share matrices must have the same non-empty shape")
        if np.any(left > 1) or np.any(right > 1):
            raise IntegrityError("client shares must contain only bits")
        rng = np.random.default_rng(seed)
        tickets = tuple(
            rng.integers(0, 256, size=TICKET_BYTES, dtype=np.uint8).tobytes()
            for _ in range(left.shape[0])
        )
        if len(set(tickets)) != len(tickets):
            raise IntegrityError("ticket generator produced a duplicate ticket")
        batches = []
        for server, shares in ((Endpoint.S0, left), (Endpoint.S1, right)):
            tags = tuple(
                hmac.new(
                    self._client_key(client, server),
                    self._share_message(
                        client=client, server=server, ticket=tickets[client], share=shares[client]
                    ),
                    hashlib.sha256,
                ).digest()
                for client in range(shares.shape[0])
            )
            batches.append(
                AuthenticatedShareBatch(
                    server=server,
                    shares=np.ascontiguousarray(shares.copy()),
                    tickets=tickets,
                    tags=tags,
                )
            )
        return batches[0], batches[1]

    def _verify_server_batch(self, batch: AuthenticatedShareBatch) -> tuple[np.ndarray, tuple[bytes, ...]]:
        shares = np.asarray(batch.shares, dtype=np.uint8)
        if batch.server not in (Endpoint.S0, Endpoint.S1):
            raise IntegrityError("authenticated share batch has an invalid server role")
        if shares.ndim != 2 or min(shares.shape) < 1 or np.any(shares > 1):
            raise IntegrityError("authenticated share batch has invalid shares")
        if len(batch.tickets) != shares.shape[0] or len(batch.tags) != shares.shape[0]:
            raise IntegrityError("authenticated share metadata count mismatch")
        if len(set(batch.tickets)) != len(batch.tickets):
            raise IntegrityError("duplicate client ticket")
        for client, (ticket, tag, share) in enumerate(
            zip(batch.tickets, batch.tags, shares, strict=True)
        ):
            expected = hmac.new(
                self._client_key(client, batch.server),
                self._share_message(
                    client=client, server=batch.server, ticket=ticket, share=share
                ),
                hashlib.sha256,
            ).digest()
            if len(tag) != MAC_BYTES or not hmac.compare_digest(expected, tag):
                raise IntegrityError("client-share MAC verification failed")
        order = sorted(range(shares.shape[0]), key=lambda index: batch.tickets[index])
        return np.ascontiguousarray(shares[order]), tuple(batch.tickets[index] for index in order)

    def _batch_digest(self, tickets: tuple[bytes, ...]) -> bytes:
        return hashlib.sha256(
            b"RAIN-BATCH-v1" + self.context.encode() + b"".join(tickets)
        ).digest()

    def verify_client_batches(
        self, server0: AuthenticatedShareBatch, server1: AuthenticatedShareBatch
    ) -> VerifiedClientBatch:
        if server0.server is not Endpoint.S0 or server1.server is not Endpoint.S1:
            raise IntegrityError("client-share batches are bound to the wrong server")
        share0, tickets0 = self._verify_server_batch(server0)
        share1, tickets1 = self._verify_server_batch(server1)
        digest0 = self._batch_digest(tickets0)
        digest1 = self._batch_digest(tickets1)
        if tickets0 != tickets1 or not hmac.compare_digest(digest0, digest1):
            raise IntegrityError("servers accepted different client ticket sets")
        clients = len(tickets0)
        return VerifiedClientBatch(
            server0_shares=share0,
            server1_shares=share1,
            sorted_tickets=tickets0,
            digest=digest0,
            client_auth_bytes=2 * clients * (TICKET_BYTES + MAC_BYTES),
            batch_exchange_bytes=2 * (8 + MAC_BYTES),
        )
