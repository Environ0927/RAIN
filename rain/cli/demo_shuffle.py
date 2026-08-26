"""One-command correctness demo for RAIN's Phase 1 shuffle."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from rain.client.randomizer import ClientRandomizer
from rain.protocol.boolean_sharing import reconstruct_bits, share_bits
from rain.protocol.oracle import plaintext_shuffle_oracle
from rain.protocol.preprocessing import IdealShufflePreprocessor
from rain.protocol.shuffle import run_secret_shared_shuffle
from rain.transport import LogicalTransport


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a protocol-level two-server RAIN shuffle correctness demo."
    )
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--dimension", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--round-id", type=int, default=1)
    parser.add_argument("--batch-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=0.2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.clients <= 0 or args.dimension <= 0 or args.chunk_size <= 0:
        raise ValueError("clients, dimension, and chunk-size must be positive")

    source_rng = np.random.default_rng(args.seed)
    updates = source_rng.normal(size=(args.clients, args.dimension))
    randomizer = ClientRandomizer(
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        seed=args.seed + 1,
    )
    started = time.perf_counter()
    randomized = randomizer.randomize_batch(updates)
    share0, share1 = share_bits(
        randomized.bits, rng=np.random.default_rng(args.seed + 2)
    )
    client_seconds = time.perf_counter() - started

    permutation0 = np.random.default_rng(args.seed + 3).permutation(args.clients)
    permutation1 = np.random.default_rng(args.seed + 4).permutation(args.clients)
    preprocessed = IdealShufflePreprocessor().prepare(
        batch_size=args.clients,
        dimension=args.dimension,
        round_id=args.round_id,
        batch_id=args.batch_id,
        server0_seed=args.seed + 5,
        server1_seed=args.seed + 6,
        permutation0=permutation0,
        permutation1=permutation1,
    )
    result = run_secret_shared_shuffle(
        server0=preprocessed.server0,
        server1=preprocessed.server1,
        client_share0=share0,
        client_share1=share1,
        transport=LogicalTransport(),
        chunk_size=args.chunk_size,
        client_seconds=client_seconds,
        preprocessing_seconds=preprocessed.elapsed_seconds,
    )
    reconstructed = reconstruct_bits(result.server0_share, result.server1_share)
    expected = plaintext_shuffle_oracle(
        randomized.bits, permutation0, permutation1
    )
    oracle_match = bool(np.array_equal(reconstructed, expected))
    before = sorted(map(tuple, randomized.bits.tolist()))
    after = sorted(map(tuple, reconstructed.tolist()))
    multiset_preserved = before == after
    if not oracle_match or not multiset_preserved:
        raise RuntimeError("shuffle correctness check failed; no oracle fallback was used")

    report = {
        "configuration": {
            "clients": args.clients,
            "dimension": args.dimension,
            "chunk_size": args.chunk_size,
            "round_id": args.round_id,
            "batch_id": args.batch_id,
            "seed": args.seed,
        },
        "correctness": {
            "oracle_match": oracle_match,
            "multiset_preserved": multiset_preserved,
        },
        "communication_bytes": result.communication.as_dict(),
        "computation_seconds": result.computation.as_dict(),
        "measurement_scope": (
            "Protocol-level simulator; communication is exact serialized bytes "
            "without TCP/IP headers, and computation excludes network latency."
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
