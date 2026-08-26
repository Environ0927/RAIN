"""Aggregate final evaluated rows across seeds without hard-coded values."""
from __future__ import annotations

import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np


def summarize(paths: list[Path]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        evaluated = [row for row in rows if row.get("accuracy") is not None]
        if not evaluated:
            raise ValueError(f"{path} has no evaluated round")
        config_path = path.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        attack = config.get("attack", {"name": "none", "malicious_clients": 0})
        protocol = config.get("protocol", {"aggregation": "unknown", "backend": "unknown"})
        key = (config["data"]["name"], config["model"]["name"],
               protocol["aggregation"], protocol.get("backend", "secure"),
               attack.get("name", "none"), int(attack.get("malicious_clients", 0)))
        grouped[key].append(evaluated[-1])
    result = []
    metrics = ("accuracy", "balanced_accuracy", "asr", "wall_seconds",
               "client_to_server_bytes", "server_to_server_bytes", "epsilon")
    for key, rows in sorted(grouped.items()):
        item = {"dataset": key[0], "model": key[1], "aggregation": key[2],
                "backend": key[3], "attack": key[4],
                "malicious_clients": key[5], "runs": len(rows)}
        for metric in metrics:
            values = [float(row[metric]) for row in rows if row.get(metric) is not None]
            item[f"{metric}_mean"] = float(np.mean(values)) if values else None
            item[f"{metric}_std"] = float(np.std(values)) if values else None
        result.append(item)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = sorted({Path(match) for pattern in args.inputs for match in glob.glob(pattern)})
    if not paths:
        raise ValueError("no result files matched --inputs")
    result = summarize(paths)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__": main()
