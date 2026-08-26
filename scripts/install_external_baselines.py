"""Install and verify exact upstream revisions used by paper comparisons."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


BASELINES = {
    "camel": {
        "repository": "https://github.com/Shuangqing-Xu/Camel",
        "commit": "19f5aced24e86872d8864a6fd7c38b576050970f",
        "directory": "Camel",
    },
    "flguard": {
        "repository": "https://github.com/201younghanlee/FLGuard",
        "commit": "537c5919004eda8fa7fa2339c5842b390ab05b93",
        "directory": "FLGuard",
    },
    "rflpa": {
        "repository": "https://github.com/NusIoraPrivacy/RFLPA",
        "commit": "dcd71183a907d70d0a385bd2bbf6385078b4c01f",
        "directory": "RFLPA",
    },
    "safefl": {
        "repository": "https://github.com/encryptogroup/SAFEFL",
        "commit": "31773ec23e25e620b6662ae284b507f1fffcd6e7",
        "directory": "SAFEFL",
    },
    "foolsgold": {
        "repository": "https://github.com/DistributedML/FoolsGold",
        "commit": "0aa55114296a2d3c2bcb6f544a6fae31e8e7b8b4",
        "directory": "FoolsGold",
    },
    "flpoison": {
        "repository": "https://github.com/vio1etus/FLPoison",
        "commit": "2e9a9d41e9cbe991b88505324fada5e8ac47fad3",
        "directory": "FLPoison",
    },
}


def _git(*args: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def verify_checkout(path: Path, expected_commit: str) -> dict[str, object]:
    if not (path / ".git").is_dir():
        return {"ready": False, "reason": "missing Git checkout"}
    try:
        head = _git("rev-parse", "HEAD", cwd=path)
        dirty = bool(_git("status", "--porcelain", cwd=path))
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"ready": False, "reason": f"Git verification failed: {exc}"}
    return {
        "ready": head == expected_commit and not dirty,
        "head": head,
        "expected_commit": expected_commit,
        "clean": not dirty,
    }


def install_checkout(root: Path, name: str) -> dict[str, object]:
    spec = BASELINES[name]
    path = root / str(spec["directory"])
    if path.exists() and not (path / ".git").is_dir():
        raise RuntimeError(f"refusing to overwrite non-Git path: {path}")
    if not path.exists():
        root.mkdir(parents=True, exist_ok=True)
        _git("clone", "--filter=blob:none", str(spec["repository"]), str(path))
    _git("fetch", "--force", "origin", str(spec["commit"]), cwd=path)
    _git("checkout", "--detach", str(spec["commit"]), cwd=path)
    report = verify_checkout(path, str(spec["commit"]))
    if not report["ready"]:
        raise RuntimeError(f"checkout verification failed for {name}: {report}")
    return {"name": name, "path": str(path), "repository": spec["repository"], **report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("external"))
    parser.add_argument("--only", choices=sorted(BASELINES), action="append")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    names = args.only or sorted(BASELINES)
    reports = []
    for name in names:
        spec = BASELINES[name]
        path = args.root / str(spec["directory"])
        if args.verify_only:
            reports.append({"name": name, "path": str(path), **verify_checkout(path, str(spec["commit"]))})
        else:
            reports.append(install_checkout(args.root, name))
    document = {"schema_version": 1, "checkouts": reports}
    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not all(item.get("ready") is True for item in reports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
