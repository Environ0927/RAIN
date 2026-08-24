"""Command-line shuffled Gaussian privacy accountant."""

from __future__ import annotations

import argparse
import json

from rain.privacy import account_privacy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, required=True)
    parser.add_argument("--delta", type=float, default=1e-5)
    args = parser.parse_args()
    report = account_privacy(
        participating_clients=args.clients,
        rounds=args.rounds,
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        delta=args.delta,
    )
    print(json.dumps(report.as_dict(), indent=2))


if __name__ == "__main__":
    main()
