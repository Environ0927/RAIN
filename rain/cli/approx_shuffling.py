"""Evaluate the cited Approx+Shuffling privacy bound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rain.privacy.approx_shuffling import approximate_shuffling_epsilon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--local-epsilon", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = approximate_shuffling_epsilon(
        clients=args.clients, local_epsilon=args.local_epsilon, delta=args.delta
    ).to_dict()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
