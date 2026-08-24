"""Coordinator for Algorithm 2 without access to either party's state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rain.metrics import CommunicationSnapshot, ComputationSnapshot
from rain.transport import LogicalTransport
from rain.types import Endpoint, MessageKind, ProtocolPhase

from .boolean_sharing import validate_bits
from .party import Server0, Server1


@dataclass(frozen=True, slots=True)
class ShuffleResult:
    server0_share: np.ndarray
    server1_share: np.ndarray
    communication: CommunicationSnapshot
    computation: ComputationSnapshot


def _send(
    transport: LogicalTransport,
    array: np.ndarray,
    *,
    kind: MessageKind,
    phase: ProtocolPhase,
    sender: Endpoint,
    receiver: Endpoint,
    round_id: int,
    chunk_size: int,
    batch_id: int,
):
    return transport.send_array(
        array,
        kind=kind,
        phase=phase,
        sender=sender,
        receiver=receiver,
        round_id=round_id,
        chunk_size=chunk_size,
        batch_id=batch_id,
        pack_bits=True,
    )


def run_secret_shared_shuffle(
    *,
    server0: Server0,
    server1: Server1,
    client_share0: np.ndarray,
    client_share1: np.ndarray,
    transport: LogicalTransport,
    chunk_size: int,
    client_seconds: float = 0.0,
    preprocessing_seconds: float = 0.0,
) -> ShuffleResult:
    """Execute offline preprocessing and Algorithm 2's online phase.

    The coordinator calls only endpoint methods and cannot access either
    ``PartyState``.  It never reconstructs plaintext and never calls the oracle.
    """

    if not isinstance(server0, Server0) or not isinstance(server1, Server1):
        raise TypeError("server0 and server1 must be their matching endpoint types")
    if server0.round_id != server1.round_id:
        raise ValueError("server round IDs must match")
    if server0.batch_id != server1.batch_id:
        raise ValueError("server batch IDs must match")
    if server0.shape != server1.shape:
        raise ValueError("server shapes must match")
    shape = server0.shape
    validate_bits(client_share0, name="client_share0", expected_shape=shape)
    validate_bits(client_share1, name="client_share1", expected_shape=shape)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    round_id = server0.round_id
    batch_id = server0.batch_id

    # Client-to-party delivery is serialized and counted like every other edge.
    server0.accept_client_share(
        _send(
            transport,
            client_share0,
            kind=MessageKind.CLIENT_SHARE,
            phase=ProtocolPhase.CLIENT,
            sender=Endpoint.CLIENT,
            receiver=Endpoint.S0,
            round_id=round_id,
            chunk_size=chunk_size,
            batch_id=batch_id,
        )
    )
    server1.accept_client_share(
        _send(
            transport,
            client_share1,
            kind=MessageKind.CLIENT_SHARE,
            phase=ProtocolPhase.CLIENT,
            sender=Endpoint.CLIENT,
            receiver=Endpoint.S1,
            round_id=round_id,
            chunk_size=chunk_size,
            batch_id=batch_id,
        )
    )

    # Paper Algorithm 2 online phase: S1 -> S0 (z1), then S0 -> S1 (z0).
    z1_messages = _send(
        transport,
        server1.start_online(),
        kind=MessageKind.ONLINE_Z1,
        phase=ProtocolPhase.ONLINE,
        sender=Endpoint.S1,
        receiver=Endpoint.S0,
        round_id=round_id,
        chunk_size=chunk_size,
        batch_id=batch_id,
    )
    z0_messages = _send(
        transport,
        server0.receive_online_z1(z1_messages),
        kind=MessageKind.ONLINE_Z0,
        phase=ProtocolPhase.ONLINE,
        sender=Endpoint.S0,
        receiver=Endpoint.S1,
        round_id=round_id,
        chunk_size=chunk_size,
        batch_id=batch_id,
    )
    server1.receive_online_z0(z0_messages)

    s0 = server0.output_share()
    s1 = server1.output_share()
    return ShuffleResult(
        server0_share=s0,
        server1_share=s1,
        communication=transport.metrics,
        computation=ComputationSnapshot(
            client_seconds=float(client_seconds),
            s0_seconds=server0.offline_seconds + server0.online_seconds,
            s1_seconds=server1.offline_seconds + server1.online_seconds,
            offline_seconds=(
                float(preprocessing_seconds)
                + server0.offline_seconds
                + server1.offline_seconds
            ),
            online_seconds=server0.online_seconds + server1.online_seconds,
        ),
    )
