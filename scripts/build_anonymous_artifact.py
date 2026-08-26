"""Build a source-only anonymous review package from an explicit allowlist."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from rain.artifact_policy import check_open_science


ROOT_FILES = {
    ".gitattributes", ".gitignore", "aggregation_rules.py", "attacks.py",
    "data_loaders.py", "Dockerfile", "LICENSE", "main.py", "pyproject.toml",
    "README.md", "RELEASE_CONTENTS.md", "requirements-cpu.txt",
    "requirements-gpu.txt", "trust_sign.py", "utils.py",
}
ROOT_DIRECTORIES = {
    "calibration", "configs", "docs", "models", "paper", "rain", "tests", "util"
}
SCRIPT_FILES = {
    ".github/workflows/ci.yml",
    "scripts/build_anonymous_artifact.py",
    "scripts/install_external_baselines.py",
    "scripts/run_external_baseline.py",
}
SCRIPT_DIRECTORIES = {"scripts/slurm"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDED_NAMES = {
    "check_cyclic_wave.py", "summarize_seed_cases.py",
    "export_selected_curve_data.py", "plot_selected_curve_bundle.py",
    "plot_configured_curves.py", "test_check_cyclic_wave.py",
    "test_summarize_seed_cases.py", "test_selected_curve_bundle.py",
    "test_plot_configured_curves.py",
}


def _allowed(relative: Path) -> bool:
    posix = relative.as_posix()
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if relative.name in EXCLUDED_NAMES or "stress" in relative.name.lower():
        return False
    if len(relative.parts) == 1:
        return relative.name in ROOT_FILES
    if relative.parts[0] in ROOT_DIRECTORIES:
        return True
    if posix in SCRIPT_FILES:
        return True
    return any(posix == directory or posix.startswith(directory + "/") for directory in SCRIPT_DIRECTORIES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(source: Path, output: Path, artifact_url: str | None) -> dict[str, object]:
    source = source.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    if source == output or source in output.parents:
        raise ValueError("output must not be inside the source repository")
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if not _allowed(relative):
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

    if artifact_url:
        if not artifact_url.startswith("https://"):
            raise ValueError("artifact URL must use HTTPS")
        metadata_path = output / "paper/open_science.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["anonymous_url"] = artifact_url
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        tex_path = output / "paper/open_science_appendix.tex"
        tex = tex_path.read_text(encoding="utf-8").replace(
            "\\texttt{[ANONYMOUS-URL-REQUIRED]}", f"\\url{{{artifact_url}}}"
        )
        tex_path.write_text(tex, encoding="utf-8")

    report = check_open_science(output)
    (output / "OPEN_SCIENCE_AUDIT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    checksum_path = output / "ARTIFACT_MANIFEST.sha256"
    rows = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path != checksum_path:
            rows.append(f"{_sha256(path)}  {path.relative_to(output).as_posix()}")
    checksum_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-url")
    args = parser.parse_args()
    report = build(args.source, args.output, args.artifact_url)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["anonymous_package_ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
