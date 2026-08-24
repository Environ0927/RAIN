"""Generate training curves exclusively from raw JSONL results."""
from __future__ import annotations

import argparse, json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("input result file is empty")
    required = {"round", "accuracy", "loss"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("each row must contain round, accuracy, and loss")
    rows = [row for row in rows if row["accuracy"] is not None and row["loss"] is not None]
    if not rows:
        raise ValueError("input contains no evaluated rounds")
    import matplotlib.pyplot as pyplot
    figure, left = pyplot.subplots()
    right = left.twinx()
    left.plot([row["round"] for row in rows], [row["accuracy"] for row in rows], label="accuracy")
    right.plot([row["round"] for row in rows], [row["loss"] for row in rows], color="tab:orange", label="loss")
    left.set(xlabel="round", ylabel="accuracy"); right.set_ylabel("loss")
    figure.tight_layout(); figure.savefig(args.output, dpi=160); pyplot.close(figure)


if __name__ == "__main__": main()
