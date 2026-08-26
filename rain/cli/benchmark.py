"""Protocol communication/chunk benchmark; no network latency is modeled."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from rain.training.coordinator import RainRoundCoordinator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--dimension", type=int, default=1000)
    parser.add_argument("--chunks", default="256,1024,4096")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    updates = rng.normal(size=(args.clients, args.dimension))
    reference = rng.integers(0, 2, size=args.dimension, dtype=np.uint8)
    rows = []
    expected = None
    for chunk in [int(x) for x in args.chunks.split(",")]:
        started = time.perf_counter()
        result = RainRoundCoordinator(
            clip_norm=100, noise_multiplier=0, tau=.4, chunk_size=chunk, seed=args.seed
        ).run_round(updates, reference, round_id=0)
        if expected is None:
            expected = result.direction_bits
        elif not np.array_equal(expected, result.direction_bits):
            raise RuntimeError("chunk sizes produced different directions")
        rows.append({"chunk_size": chunk, "wall_seconds": time.perf_counter() - started, **result.metrics.as_dict()})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
