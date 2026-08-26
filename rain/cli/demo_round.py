"""Small complete client -> shuffle -> secure aggregation demonstration."""

from __future__ import annotations

import argparse
import json

import numpy as np

from rain.training.coordinator import RainRoundCoordinator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=31)
    parser.add_argument("--tau", type=float, default=.4)
    parser.add_argument("--noise-multiplier", type=float, default=0.0)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    updates = rng.normal(size=(args.clients, args.dimension))
    reference = rng.integers(0, 2, size=args.dimension, dtype=np.uint8)
    result = RainRoundCoordinator(
        clip_norm=1.0,
        noise_multiplier=args.noise_multiplier,
        tau=args.tau,
        chunk_size=args.chunk_size,
        seed=args.seed,
    ).run_round(updates, reference, round_id=0)
    print(json.dumps({
        "direction_shape": list(result.direction_bits.shape),
        "positive_fraction": float(result.direction_bits.mean()),
        "threshold_count": result.threshold_count,
        "metrics": result.metrics.as_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
