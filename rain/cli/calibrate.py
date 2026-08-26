"""Calibrate tau from independent NumPy arrays."""

from __future__ import annotations

import argparse
import json

import numpy as np

from rain.privacy import calibrate_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", required=True, help=".npy matrix of benign calibration updates")
    parser.add_argument("--reference", required=True, help=".npy bit vector")
    parser.add_argument("--output", required=True)
    parser.add_argument("--clip-norm", type=float, required=True)
    parser.add_argument("--noise-multiplier", type=float, required=True)
    parser.add_argument("--quantile", type=float, default=.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    record = calibrate_threshold(
        np.load(args.updates, allow_pickle=False),
        np.load(args.reference, allow_pickle=False),
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        quantile=args.quantile,
        seed=args.seed,
    )
    record.save(args.output)
    print(json.dumps(record.as_dict(), indent=2))


if __name__ == "__main__":
    main()
