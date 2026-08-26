"""Run deterministic RAIN-MAC tamper-detection checks."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np

from rain.errors import IntegrityError
from rain.integrity import IntegrityContext, RainMacSession
from rain.types import Endpoint, MessageKind, ProtocolPhase


def _detected(action) -> bool:
    try:
        action()
    except IntegrityError:
        return True
    return False


def run_checks(*, clients: int, dimension: int, seed: int) -> dict[str, object]:
    if clients < 2 or dimension < 1:
        raise ValueError("clients must be at least two and dimension must be positive")
    rng = np.random.default_rng(seed)
    share0 = rng.integers(0, 2, size=(clients, dimension), dtype=np.uint8)
    share1 = rng.integers(0, 2, size=(clients, dimension), dtype=np.uint8)
    context = IntegrityContext(session_id="integrity-demo", round_id=2, batch_id=3)
    session = RainMacSession(context=context, seed=seed + 1)
    batch0, batch1 = session.authenticate_client_shares(share0, share1, seed=seed + 2)
    accepted = session.verify_client_batches(batch0, batch1)

    modified = batch0.shares.copy(); modified[0, 0] ^= 1
    duplicate_tickets = (batch0.tickets[0], batch0.tickets[0]) + batch0.tickets[2:]
    other_round = RainMacSession(
        context=IntegrityContext(session_id="integrity-demo", round_id=4, batch_id=3),
        seed=seed + 1,
    )
    cases = {
        "modified_share": _detected(
            lambda: session.verify_client_batches(replace(batch0, shares=modified), batch1)
        ),
        "forged_tag": _detected(
            lambda: session.verify_client_batches(
                replace(batch0, tags=(bytes(32),) + batch0.tags[1:]), batch1
            )
        ),
        "duplicate_ticket": _detected(
            lambda: session.verify_client_batches(
                replace(batch0, tickets=duplicate_tickets), batch1
            )
        ),
        "cross_round_replay": _detected(
            lambda: other_round.verify_client_batches(batch0, batch1)
        ),
    }

    transcript = RainMacSession(context=context, seed=seed + 3).transcript
    wire = b"demo-server-message"
    envelope = transcript.create_envelope(
        wire, sender=Endpoint.S0, receiver=Endpoint.S1,
        kind=MessageKind.ONLINE_Z0, phase=ProtocolPhase.ONLINE,
    )
    cases["transcript_payload_tamper"] = _detected(
        lambda: transcript.verify_and_commit(wire + b"!", envelope)
    )
    transcript.verify_and_commit(wire, envelope)
    cases["transcript_replay"] = _detected(
        lambda: transcript.verify_and_commit(wire, envelope)
    )
    return {
        "schema_version": 1,
        "accepted_clean_batch": True,
        "clients": clients,
        "dimension": dimension,
        "batch_digest": accepted.digest.hex(),
        "cases": cases,
        "all_tampering_detected": all(cases.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(run_checks(clients=args.clients, dimension=args.dimension, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
