"""Run a pinned upstream baseline while preserving provenance and raw output."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

from install_external_baselines import BASELINES, verify_checkout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, choices=sorted(BASELINES))
    parser.add_argument("--root", type=Path, default=Path("external"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("an upstream command is required after --")

    spec = BASELINES[args.name]
    checkout = args.root / str(spec["directory"])
    verification = verify_checkout(checkout, str(spec["commit"]))
    if not verification.get("ready"):
        raise SystemExit(f"upstream checkout is not ready: {verification}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "upstream.log"
    manifest_path = args.output_dir / "run_manifest.json"
    started = dt.datetime.now(dt.timezone.utc)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=checkout, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="")
        return_code = process.wait()
    finished = dt.datetime.now(dt.timezone.utc)
    manifest = {
        "schema_version": 1,
        "baseline": args.name,
        "repository": spec["repository"],
        "source_commit": spec["commit"],
        "checkout": str(checkout.resolve()),
        "command": command,
        "python": sys.version,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "wall_seconds": (finished - started).total_seconds(),
        "return_code": return_code,
        "raw_log": log_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if return_code:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()
