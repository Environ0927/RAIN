"""Plot comparable convergence curves from raw per-run JSONL logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("inputs must use LABEL=path")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("inputs must use a non-empty LABEL=path")
    return label.strip(), Path(raw_path)


def load_curve(path: Path, metric: str = "accuracy") -> tuple[list[int], list[float]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    evaluated = [row for row in rows if row.get(metric) is not None]
    if not evaluated:
        raise ValueError(f"{path} contains no evaluated rounds")
    return (
        [int(row["round"]) + 1 for row in evaluated],
        [float(row[metric]) for row in evaluated],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=_parse_spec)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Plaintext aggregation convergence")
    parser.add_argument("--metric", choices=("accuracy", "validation_accuracy"), default="accuracy")
    args = parser.parse_args()
    import matplotlib.pyplot as pyplot

    figure, axis = pyplot.subplots(figsize=(6.4, 4.2))
    for label, path in args.inputs:
        rounds, accuracy = load_curve(path, args.metric)
        axis.plot(rounds, accuracy, label=label, linewidth=1.8)
    ylabel = "Validation accuracy" if args.metric == "validation_accuracy" else "Test accuracy"
    axis.set(xlabel="Communication round", ylabel=ylabel, title=args.title)
    axis.grid(alpha=.25); axis.legend(); figure.tight_layout()
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180); pyplot.close(figure)


if __name__ == "__main__":
    main()
