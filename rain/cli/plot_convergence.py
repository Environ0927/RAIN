"""Plot comparable convergence curves from raw per-run JSONL logs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


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


def aggregate_curves(
    paths: list[Path], metric: str = "accuracy",
) -> tuple[list[int], np.ndarray, np.ndarray]:
    curves = [load_curve(path, metric) for path in paths]
    rounds = curves[0][0]
    if any(value[0] != rounds for value in curves[1:]):
        raise ValueError("seed curves must use the same evaluated rounds")
    matrix = np.asarray([value[1] for value in curves], dtype=np.float64)
    return rounds, matrix.mean(axis=0), matrix.std(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, type=_parse_spec)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Plaintext aggregation convergence")
    parser.add_argument("--metric", choices=("accuracy", "validation_accuracy"), default="accuracy")
    args = parser.parse_args()
    import matplotlib.pyplot as pyplot

    figure, axis = pyplot.subplots(figsize=(6.4, 4.2))
    grouped: dict[str, list[Path]] = defaultdict(list)
    for label, path in args.inputs:
        grouped[label].append(path)
    for label, paths in grouped.items():
        rounds, mean, std = aggregate_curves(paths, args.metric)
        line = axis.plot(rounds, mean, label=label, linewidth=1.8)[0]
        if len(paths) > 1:
            axis.fill_between(
                rounds, mean - std, mean + std,
                color=line.get_color(), alpha=.16, linewidth=0,
            )
    ylabel = "Validation accuracy" if args.metric == "validation_accuracy" else "Test accuracy"
    axis.set(xlabel="Communication round", ylabel=ylabel, title=args.title)
    axis.grid(alpha=.25); axis.legend(); figure.tight_layout()
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=180); pyplot.close(figure)


if __name__ == "__main__":
    main()
