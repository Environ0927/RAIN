"""Checks used to prepare an anonymous open-science review package."""

from __future__ import annotations

import json
import re
from pathlib import Path


REQUIRED_OPEN_SCIENCE_FILES = (
    "docs/OPEN_SCIENCE.md",
    "paper/open_science_appendix.tex",
    "paper/open_science.json",
    "paper/artifact_inventory.json",
    "scripts/build_anonymous_artifact.py",
    "RELEASE_CONTENTS.md",
)

TEXT_SUFFIXES = {
    "", ".cfg", ".csv", ".ini", ".json", ".jsonl", ".md", ".py",
    ".rst", ".sbatch", ".sh", ".tex", ".toml", ".txt", ".yaml", ".yml",
}


def _identity_patterns() -> dict[str, re.Pattern[str]]:
    windows_prefix = "[A-Za-z]:" + re.escape("\\" + "Users" + "\\")
    posix_prefix = re.escape("/" + "home" + "/")
    return {
        "windows_user_path": re.compile(windows_prefix + r"[^\\/\s]+", re.IGNORECASE),
        "posix_home_path": re.compile(posix_prefix + r"[^/\s]+", re.IGNORECASE),
        "private_ipv4": re.compile(
            r"(?<!\d)(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|"
            r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?!\d)"
        ),
        "email_address": re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
        ),
    }


def scan_anonymity(root: str | Path) -> dict[str, object]:
    """Return identity leaks in text files and forbidden repository metadata."""

    base = Path(root).resolve()
    findings: list[dict[str, object]] = []
    patterns = _identity_patterns()
    for path in sorted(base.rglob("*")):
        if not path.is_file() or ".git" in path.parts or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative = path.relative_to(base).as_posix()
        for line_number, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    findings.append({
                        "path": relative,
                        "line": line_number,
                        "kind": kind,
                        "match": match.group(0),
                    })
    git_metadata_present = (base / ".git").exists()
    return {
        "anonymous": not findings and not git_metadata_present,
        "git_metadata_present": git_metadata_present,
        "findings": findings,
    }


def check_open_science(root: str | Path) -> dict[str, object]:
    """Check submission-package requirements that can be verified offline."""

    base = Path(root).resolve()
    files = {name: (base / name).is_file() for name in REQUIRED_OPEN_SCIENCE_FILES}
    metadata_valid = False
    anonymous_url_configured = False
    inventory_valid = False
    try:
        metadata = json.loads((base / "paper/open_science.json").read_text(encoding="utf-8"))
        url = metadata.get("anonymous_url")
        anonymous_url_configured = (
            isinstance(url, str)
            and url.startswith("https://")
            and "REQUIRED" not in url.upper()
            and "PLACEHOLDER" not in url.upper()
        )
        metadata_valid = metadata.get("schema_version") == 1
    except (OSError, json.JSONDecodeError):
        metadata_valid = False
    try:
        inventory = json.loads(
            (base / "paper/artifact_inventory.json").read_text(encoding="utf-8")
        )
        kinds = {row["kind"] for row in inventory["artifacts"]}
        inventory_valid = (
            inventory.get("schema_version") == 1
            and {"source", "configuration", "data", "results", "external-baseline"}
            <= kinds
            and all("availability" in row for row in inventory["artifacts"])
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        inventory_valid = False
    anonymity = scan_anonymity(base)
    package_ready = (
        all(files.values()) and metadata_valid and inventory_valid
        and anonymity["anonymous"] is True
    )
    return {
        "required_files": files,
        "metadata_valid": metadata_valid,
        "inventory_valid": inventory_valid,
        "anonymous_url_configured": anonymous_url_configured,
        "anonymity": anonymity,
        "anonymous_package_ready": package_ready,
        "open_science_submission_ready": package_ready and anonymous_url_configured,
    }
