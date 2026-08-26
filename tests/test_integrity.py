from dataclasses import replace

import numpy as np
import pytest

from rain.errors import IntegrityError
from rain.integrity import IntegrityContext, RainMacSession
from rain.types import Endpoint, MessageKind, ProtocolPhase


def _session(seed=7):
    return RainMacSession(
        context=IntegrityContext(session_id="test", round_id=3, batch_id=9),
        seed=seed,
    )


def _batches():
    rng = np.random.default_rng(4)
    share0 = rng.integers(0, 2, size=(5, 17), dtype=np.uint8)
    share1 = rng.integers(0, 2, size=(5, 17), dtype=np.uint8)
    return _session(), share0, share1


def test_client_share_macs_verify_and_align_same_ticket_set():
    session, share0, share1 = _batches()
    batch0, batch1 = session.authenticate_client_shares(share0, share1, seed=10)
    verified = session.verify_client_batches(batch0, batch1)
    assert verified.server0_shares.shape == share0.shape
    assert verified.server1_shares.shape == share1.shape
    assert verified.client_auth_bytes == 2 * 5 * (16 + 32)
    assert len(verified.digest) == 32


def test_modified_share_and_tag_are_rejected():
    session, share0, share1 = _batches()
    batch0, batch1 = session.authenticate_client_shares(share0, share1, seed=10)
    changed = batch0.shares.copy()
    changed[0, 0] ^= 1
    with pytest.raises(IntegrityError, match="MAC"):
        session.verify_client_batches(replace(batch0, shares=changed), batch1)
    bad_tags = (bytes(32),) + batch0.tags[1:]
    with pytest.raises(IntegrityError, match="MAC"):
        session.verify_client_batches(replace(batch0, tags=bad_tags), batch1)


def test_duplicate_replayed_and_cross_server_tickets_are_rejected():
    session, share0, share1 = _batches()
    batch0, batch1 = session.authenticate_client_shares(share0, share1, seed=10)
    tickets = (batch0.tickets[0], batch0.tickets[0]) + batch0.tickets[2:]
    with pytest.raises(IntegrityError, match="duplicate"):
        session.verify_client_batches(replace(batch0, tickets=tickets), batch1)
    with pytest.raises(IntegrityError, match="wrong server"):
        session.verify_client_batches(replace(batch0, server=Endpoint.S1), batch1)


def test_round_context_prevents_cross_round_replay():
    session, share0, share1 = _batches()
    batch0, batch1 = session.authenticate_client_shares(share0, share1, seed=10)
    other = RainMacSession(
        context=IntegrityContext(session_id="test", round_id=4, batch_id=9), seed=7
    )
    with pytest.raises(IntegrityError, match="MAC"):
        other.verify_client_batches(batch0, batch1)


def test_transcript_rejects_tamper_reorder_and_stale_previous_hash():
    transcript = _session().transcript
    wire = b"serialized-message"
    envelope = transcript.create_envelope(
        wire,
        sender=Endpoint.S0,
        receiver=Endpoint.S1,
        kind=MessageKind.ONLINE_Z0,
        phase=ProtocolPhase.ONLINE,
    )
    with pytest.raises(IntegrityError, match="MAC"):
        transcript.verify_and_commit(wire + b"tamper", envelope)
    assert transcript.verify_and_commit(wire, envelope) == 72
    with pytest.raises(IntegrityError, match="sequence"):
        transcript.verify_and_commit(wire, envelope)


def test_coordinator_reports_verified_integrity_and_digests():
    from rain.training.coordinator import RainRoundCoordinator

    updates = np.random.default_rng(3).normal(size=(4, 13))
    reference = np.ones(13, dtype=np.uint8)
    result = RainRoundCoordinator(
        clip_norm=100, noise_multiplier=0, tau=.4, seed=2, chunk_size=5
    ).run_round(updates, reference, round_id=1)
    assert result.metrics.integrity_verified is True
    assert result.metrics.integrity_bytes > 0
    assert result.metrics.integrity_message_count > 2 * len(updates)
    assert len(result.metrics.batch_digest) == 64
    assert len(result.metrics.transcript_digest) == 64
