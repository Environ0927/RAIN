"""Download, verify, and cache the official writer-partitioned FEMNIST data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rain.training.large_data import (
    FEMNISTDataset,
    download_femnist,
    prepare_femnist_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./data")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="do not download; require fed_emnist_train.h5 and fed_emnist_test.h5",
    )
    args = parser.parse_args()
    root = Path(args.root)
    download_femnist(root, allow_download=not args.offline)
    rows = {}
    for split in ("train", "test"):
        dataset = FEMNISTDataset(prepare_femnist_cache(root, split))
        rows[split] = {
            "samples": len(dataset),
            "writers": len(dataset.writer_ids),
            "classes": int(dataset.targets.max()) + 1,
            "cache": str(dataset.cache_dir),
        }
    print(json.dumps({"schema_version": 1, "dataset": "femnist", "splits": rows}, indent=2))


if __name__ == "__main__":
    main()
