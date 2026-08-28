"""Select learning rates from validation-only pilot logs."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def select_candidates(
    paths: list[Path], *, metric: str = "validation_accuracy",
    expected_candidates: int | None = None, minimum_rounds: int = 1,
    required_methods: set[str] | None = None,
) -> dict[str, object]:
    candidates: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in paths:
        rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(rows) < minimum_rounds:
            raise ValueError(f"{path} has only {len(rows)} rounds; expected at least {minimum_rounds}")
        values = [(int(row["round"]), float(row[metric])) for row in rows if row.get(metric) is not None]
        if not values:
            raise ValueError(f"{path} has no {metric} values")
        config = json.loads((path.parent / "config.json").read_text(encoding="utf-8"))
        method = str(config["protocol"]["aggregation"]).lower()
        learning_rate = float(config["training"]["learning_rate"])
        best_round, score = max(values, key=lambda item: item[1])
        candidates[method].append({
            "learning_rate": learning_rate,
            "score": score,
            "best_round": best_round + 1,
            "path": str(path),
        })
    if required_methods is not None and set(candidates) != required_methods:
        raise ValueError(
            f"methods are {sorted(candidates)}; expected {sorted(required_methods)}"
        )
    selected: dict[str, dict[str, object]] = {}
    for method, values in sorted(candidates.items()):
        unique_rates = {float(item["learning_rate"]) for item in values}
        if len(unique_rates) != len(values):
            raise ValueError(f"{method} contains duplicate learning-rate candidates")
        if expected_candidates is not None and len(values) != expected_candidates:
            raise ValueError(
                f"{method} has {len(values)} candidates; expected {expected_candidates}"
            )
        values.sort(key=lambda item: float(item["learning_rate"]))
        selected[method] = max(
            values, key=lambda item: (float(item["score"]), -float(item["learning_rate"]))
        )
    return {
        "schema_version": 1,
        "selection_metric": metric,
        "selection_rule": "highest observed pilot validation metric; smaller learning rate breaks ties",
        "minimum_rounds": minimum_rounds,
        "candidates": dict(sorted(candidates.items())),
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", default="validation_accuracy")
    parser.add_argument("--expected-candidates", type=int)
    parser.add_argument("--minimum-rounds", type=int, default=1)
    parser.add_argument("--methods", default="")
    args = parser.parse_args()
    result = select_candidates(
        [Path(value) for value in args.inputs], metric=args.metric,
        expected_candidates=args.expected_candidates, minimum_rounds=args.minimum_rounds,
        required_methods={value.strip().lower() for value in args.methods.split(",") if value.strip()} or None,
    )
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
