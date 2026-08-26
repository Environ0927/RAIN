"""Validate normalized measurements produced by a pinned external artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


COMMITS = {
    "camel": "19f5aced24e86872d8864a6fd7c38b576050970f",
    "flguard": "537c5919004eda8fa7fa2339c5842b390ab05b93",
}
ALLOWED_DATASETS = {"mnist", "fmnist", "femnist", "cifar10", "tinyimagenet"}


def validate_document(value: object, baseline: str) -> list[dict[str, object]]:
    if isinstance(value, dict) and "records" in value:
        records = value["records"]
    else:
        records = value
    if not isinstance(records, list) or not records:
        raise ValueError("external result must contain a non-empty records list")
    normalized = []
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"record {index} is not an object")
        if item.get("baseline") != baseline:
            raise ValueError(f"record {index} baseline does not match {baseline}")
        if item.get("source_commit") != COMMITS[baseline]:
            raise ValueError(f"record {index} does not cite the pinned source commit")
        if item.get("dataset") not in ALLOWED_DATASETS:
            raise ValueError(f"record {index} has an unsupported dataset")
        if not isinstance(item.get("seed"), int):
            raise ValueError(f"record {index} must contain an integer seed")
        metrics = item.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"record {index} must contain measured metrics")
        numeric = {
            key: value for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if not numeric:
            raise ValueError(f"record {index} contains no numeric metric")
        normalized.append({**item, "source_kind": "external-pinned"})
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, choices=sorted(COMMITS))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    records = validate_document(value, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"imported {len(records)} {args.baseline} record(s) into {args.output}")


if __name__ == "__main__":
    main()
