"""Plot CIFAR-10 and FEMNIST convergence in a compact paper-style figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHOD_ORDER = ("fedavg", "rain", "signsgd", "flod")
METHOD_LABELS = {
    "fedavg": "FedAvg",
    "rain": "RAIN",
    "signsgd": "SignSGD",
    "flod": "FLOD",
}
METHOD_STYLES = {
    "fedavg": {"color": "#7fb2d3", "linestyle": "-."},
    "rain": {"color": "#fb8073", "linestyle": "-"},
    "signsgd": {"color": "#e9aa60", "linestyle": "--"},
    "flod": {"color": "#8c6bb1", "linestyle": ":"},
}


def parse_method_path(value: str) -> tuple[str, Path]:
    """Parse a METHOD=PATH command-line input."""
    if "=" not in value:
        raise argparse.ArgumentTypeError("curve inputs must use METHOD=PATH")
    raw_method, raw_path = value.split("=", 1)
    method = raw_method.strip().lower()
    if method not in METHOD_LABELS:
        choices = ", ".join(METHOD_ORDER)
        raise argparse.ArgumentTypeError(f"unknown method {raw_method!r}; choose from {choices}")
    if not raw_path.strip():
        raise argparse.ArgumentTypeError("curve input path cannot be empty")
    return method, Path(raw_path.strip())


def load_curve(
    path: Path, metric: str = "validation_accuracy", round_limit: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load evaluated rounds directly from a training JSONL file.

    No interpolation, smoothing, rescaling, or synthetic noise is applied.
    """
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path} at line {line_number}") from error
        one_based_round = int(row["round"]) + 1
        if row.get(metric) is not None and (
            round_limit is None or one_based_round <= round_limit
        ):
            rows.append(row)
    if not rows:
        raise ValueError(f"{path} contains no values for {metric}")
    rounds = np.asarray([int(row["round"]) + 1 for row in rows], dtype=np.int64)
    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
    return rounds, values


def smooth_values(values: np.ndarray, window: int) -> np.ndarray:
    """Apply an edge-padded centered moving average for display only."""
    if window < 1 or window % 2 == 0:
        raise ValueError("smoothing window must be a positive odd integer")
    if window == 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def _as_panel(specs: list[tuple[str, Path]]) -> dict[str, Path]:
    panel: dict[str, Path] = {}
    for method, path in specs:
        if method in panel:
            raise ValueError(f"duplicate curve for {method}")
        panel[method] = path
    if not panel:
        raise ValueError("each panel requires at least one curve")
    return panel


def _draw_panel(
    axis, panel: dict[str, Path], metric: str, round_limit: int | None = None,
    smoothing_window: int = 1,
) -> None:
    maximum_round = 1
    for method in METHOD_ORDER:
        path = panel.get(method)
        if path is None:
            continue
        rounds, values = load_curve(path, metric, round_limit)
        values = smooth_values(values, smoothing_window)
        maximum_round = max(maximum_round, int(rounds[-1]))
        axis.plot(
            rounds,
            values,
            label=METHOD_LABELS[method],
            linewidth=1.5,
            alpha=1.0,
            **METHOD_STYLES[method],
        )
    axis.set_xlim(0, maximum_round)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Communication rounds")
    axis.grid(True, linestyle="--", alpha=0.45)
    axis.legend(loc="lower right", frameon=True, fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar10", nargs="+", required=True, type=parse_method_path)
    parser.add_argument("--femnist", nargs="+", required=True, type=parse_method_path)
    parser.add_argument(
        "--metric", choices=("accuracy", "validation_accuracy"),
        default="validation_accuracy",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--pdf-output")
    parser.add_argument("--cifar10-round-limit", type=int)
    parser.add_argument("--femnist-round-limit", type=int)
    parser.add_argument(
        "--smoothing-window", type=int, default=1,
        help="Centered moving-average window over evaluated points; 1 keeps raw values.",
    )
    parser.add_argument(
        "--caption-suffix", default="Seed 1",
        help="Short provenance label appended to both panel captions.",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as pyplot

    cifar10 = _as_panel(args.cifar10)
    femnist = _as_panel(args.femnist)
    figure, axes = pyplot.subplots(1, 2, figsize=(10.2, 3.55), sharey=True)
    _draw_panel(
        axes[0], cifar10, args.metric, args.cifar10_round_limit, args.smoothing_window,
    )
    _draw_panel(
        axes[1], femnist, args.metric, args.femnist_round_limit, args.smoothing_window,
    )
    axes[0].set_ylabel(
        "Validation accuracy" if args.metric == "validation_accuracy" else "Test accuracy"
    )
    suffix = f"; {args.caption_suffix}" if args.caption_suffix else ""
    axes[0].set_title(f"(a) CIFAR-10{suffix}", y=-0.34, fontfamily="serif", fontsize=12)
    axes[1].set_title(f"(b) FEMNIST{suffix}", y=-0.34, fontfamily="serif", fontsize=12)
    figure.subplots_adjust(left=0.08, right=0.985, top=0.97, bottom=0.27, wspace=0.20)

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=300, bbox_inches="tight")
    if args.pdf_output:
        pdf_target = Path(args.pdf_output)
        pdf_target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(pdf_target, bbox_inches="tight")
    pyplot.close(figure)


if __name__ == "__main__":
    main()
