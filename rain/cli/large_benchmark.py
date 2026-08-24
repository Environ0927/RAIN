"""Resumable exact-message benchmark for multi-million-coordinate RAIN rounds."""
from __future__ import annotations

import argparse, hashlib, json, time, tracemalloc
from pathlib import Path
import numpy as np

from rain.protocol.aggregation import secure_rain_aggregate
from rain.protocol.boolean_sharing import reconstruct_bits, share_bits
from rain.protocol.preprocessing import IdealShufflePreprocessor
from rain.protocol.shuffle import run_secret_shared_shuffle
from rain.transport import LogicalTransport


def _integers(value: str) -> list[int]:
    result = [int(item) for item in value.split(",")]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=_integers, default=[2, 5, 10])
    parser.add_argument("--dimensions", type=_integers, default=[11173962, 21328292, 23910152])
    parser.add_argument("--chunk-sizes", type=_integers, default=[262144])
    parser.add_argument("--tau", type=float, default=.4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="outputs/large-protocol/rows.jsonl")
    args = parser.parse_args()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line); completed.add((row["clients"], row["dimension"], row["chunk_size"]))

    for clients in args.clients:
        for dimension in args.dimensions:
            seed_sequence = np.random.SeedSequence([args.seed, clients, dimension])
            seeds = [int(value) for value in seed_sequence.generate_state(8, dtype=np.uint64)]
            rng = np.random.default_rng(seeds[0])
            bits = rng.integers(0, 2, size=(clients, dimension), dtype=np.uint8)
            reference = rng.integers(0, 2, size=dimension, dtype=np.uint8)
            for chunk_size in args.chunk_sizes:
                key = (clients, dimension, chunk_size)
                if key in completed:
                    continue
                started = time.perf_counter(); tracemalloc.start()
                share0, share1 = share_bits(bits, rng=np.random.default_rng(seeds[1]))
                preprocessing = IdealShufflePreprocessor().prepare(
                    batch_size=clients, dimension=dimension, round_id=0, batch_id=0,
                    server0_seed=seeds[2], server1_seed=seeds[3],
                )
                transport = LogicalTransport()
                shuffled = run_secret_shared_shuffle(
                    server0=preprocessing.server0, server1=preprocessing.server1,
                    client_share0=share0, client_share1=share1, transport=transport,
                    chunk_size=chunk_size, preprocessing_seconds=preprocessing.elapsed_seconds,
                )
                aggregate = secure_rain_aggregate(
                    shuffled_share0=shuffled.server0_share,
                    shuffled_share1=shuffled.server1_share,
                    reference_bits=reference, tau=args.tau, round_id=0, batch_id=0,
                    seed=seeds[4], transport=transport, chunk_size=chunk_size,
                )
                direction = reconstruct_bits(
                    aggregate.server0_direction_share, aggregate.server1_direction_share
                )
                _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
                row = {
                    "schema_version": 1, "clients": clients, "dimension": dimension,
                    "chunk_size": chunk_size, "tau": args.tau, "seed": args.seed,
                    "wall_seconds": time.perf_counter() - started,
                    "peak_python_bytes": peak,
                    "direction_sha256": hashlib.sha256(direction.tobytes()).hexdigest(),
                    "threshold_count": aggregate.threshold_count,
                    "communication": transport.metrics.as_dict(),
                    "primitives": {
                        "boolean_and_gates": aggregate.primitives.boolean_and_gates,
                        "arithmetic_multiplications": aggregate.primitives.arithmetic_multiplications,
                        "dabits_consumed": aggregate.primitives.dabits_consumed,
                    },
                    "offline_seconds": shuffled.computation.offline_seconds + aggregate.offline_seconds,
                    "online_seconds": shuffled.computation.online_seconds + aggregate.elapsed_seconds - aggregate.offline_seconds,
                }
                with output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True)); completed.add(key)


if __name__ == "__main__": main()
