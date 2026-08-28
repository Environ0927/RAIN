"""Machine-checkable paper-to-artifact implementation coverage audit."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path

from rain.artifact_policy import check_open_science
from rain.config import SUPPORTED_AGGREGATIONS, load_config
from rain.training.attacks import ATTACKS
from rain.training.plaintext_aggregation import PLAINTEXT_AGGREGATIONS
from rain.training.baseline_adapter import ADAPTER_AGGREGATIONS


DATASETS = {"femnist", "cifar10", "tinyimagenet"}
ATTACK_NAMES = {"krum", "min-max", "scaling", "attack-dpfl", "raa", "woaa"}
COMPATIBILITY_ATTACK_NAMES = {"rsca"}
PRIMARY_BASELINES = {
    "rain", "signsgd", "fedavg", "flod", "krum", "trim-mean", "median",
    "fltrust", "foundationfl", "rflpa",
}
ADAPTER_BASELINE_FUNCTIONS = {
    "shieldfl", "signguard", "foolsgold", "divide_and_conquer", "contra",
    "romoa", "flare",
}
PROTOCOL_FILES = {
    "client_randomizer": "rain/client/randomizer.py",
    "boolean_sharing": "rain/protocol/boolean_sharing.py",
    "arithmetic_sharing": "rain/protocol/arithmetic_sharing.py",
    "secret_shuffle": "rain/protocol/shuffle.py",
    "b2a": "rain/protocol/b2a.py",
    "beaver": "rain/protocol/beaver.py",
    "comparison": "rain/protocol/comparison.py",
    "secure_aggregation": "rain/protocol/aggregation.py",
    "rain_mac": "rain/integrity.py",
    "privacy_accountant": "rain/privacy/rdp_accountant.py",
    "calibration": "rain/privacy/calibration.py",
    "training": "rain/cli/train.py",
    "metrics": "rain/metrics.py",
}
PAPER_MATRIX = "paper/experiment_matrix.json"
EXTERNAL_BASELINE_MANIFEST = "paper/external_baselines.json"
BASELINE_REGISTRY = "paper/baseline_registry.json"
RESULTS_MANIFEST = "paper/results_manifest.json"
CALIBRATION_FILES = {
    "cifar10": "calibration/paper/figure4_cifar10_e10_seed1.json",
    "femnist": "calibration/paper/figure4_femnist_e10_seed1.json",
    "tinyimagenet": "calibration/paper/tinyimagenet_e10_seed1.json",
}
REQUIRED_REGISTERED_BASELINES = {
    "fedavg", "signsgd", "krum", "trim-mean", "median", "foundationfl",
    "flod", "rflpa", "shieldfl", "signguard", "fltrust", "foolsgold",
    "divide-and-conquer", "contra", "romoa", "flare", "approx-shuffling",
    "camel", "flguard",
}


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _function_hashes(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: hashlib.sha256(
            ast.dump(node, include_attributes=False).encode("utf-8")
        ).hexdigest()
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _git_checkout_status(path: Path, expected_commit: str) -> dict[str, object]:
    if not (path / ".git").is_dir():
        return {"ready": False, "reason": "missing Git checkout"}
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=path, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"ready": False, "reason": f"Git verification failed: {exc}"}
    return {
        "ready": head == expected_commit and not dirty,
        "head": head, "expected_commit": expected_commit, "clean": not dirty,
    }


def run_audit(root: str | Path) -> dict[str, object]:
    base = Path(root).resolve()
    open_science = check_open_science(base)
    calibration_files: dict[str, dict[str, object]] = {}
    for dataset, relative in CALIBRATION_FILES.items():
        path = base / relative
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            valid = (
                row.get("schema_version") == 1
                and float(row["tau"]) >= 0.0
                and float(row["tau"]) <= 1.0
                and int(row["dimension"]) > 0
                and int(row["sample_count"]) > 0
            )
            calibration_files[dataset] = {
                "present": True, "valid": valid, "path": relative,
                "tau": row.get("tau"), "sample_count": row.get("sample_count"),
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            calibration_files[dataset] = {
                "present": False, "valid": False, "path": relative,
            }
    protocol = {name: (base / relative).is_file() for name, relative in PROTOCOL_FILES.items()}
    comparison_path = base / "aggregation_rules.py"
    comparison_functions = _function_names(comparison_path) if comparison_path.is_file() else set()
    comparison_hashes = _function_hashes(comparison_path) if comparison_path.is_file() else {}
    comparison_modules = {
        name: name in comparison_functions for name in sorted(ADAPTER_BASELINE_FUNCTIONS)
    }

    configs: dict[str, bool] = {}
    for dataset in sorted(DATASETS):
        quick_name = {
            "cifar10": "cifar10_resnet18.json",
            "femnist": "femnist_cnn.json",
            "tinyimagenet": "tinyimagenet_resnet18.json",
        }[dataset]
        formal_name = quick_name
        paths = [base / "configs" / "quick" / quick_name, base / "configs" / "large" / formal_name]
        paths.extend(
            base / "configs" / "convergence" / f"{dataset}_{method}.json"
            for method in ("fedavg", "rain", "signsgd", "flod")
        )
        for path in paths:
            key = path.relative_to(base).as_posix()
            try:
                configs[key] = path.is_file() and load_config(path)["data"]["name"] == dataset
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                configs[key] = False

    try:
        matrix = json.loads((base / PAPER_MATRIX).read_text(encoding="utf-8"))
        matrix_ok = (
            set(matrix["datasets"]) == DATASETS
            and set(matrix["robustness"]["attacks"]) == ATTACK_NAMES
            and {"convergence", "privacy", "robustness", "efficiency", "ablation"}
            <= set(matrix["experiments"])
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        matrix = {}
        matrix_ok = False
    try:
        external_manifest = json.loads(
            (base / EXTERNAL_BASELINE_MANIFEST).read_text(encoding="utf-8")
        )
        external_declared = {item["name"] for item in external_manifest["baselines"]}
        external_manifest_ok = {
            "approx-shuffling", "camel", "flguard"
        }.issubset(external_declared)
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        external_manifest = {}
        external_manifest_ok = False
    try:
        registry = json.loads((base / BASELINE_REGISTRY).read_text(encoding="utf-8"))
        registry_rows = {row["name"]: row for row in registry["baselines"]}
        registered_sources = {
            name: (
                not row.get("source") or (base / str(row["source"])).is_file()
            )
            for name, row in registry_rows.items()
        }
        verified_hashes = {}
        for name in ADAPTER_AGGREGATIONS:
            function_name = "divide_and_conquer" if name == "divide-and-conquer" else name
            expected = registry_rows.get(name, {}).get("function_sha256")
            verified_hashes[name] = (
                bool(expected) and comparison_hashes.get(function_name) == expected
            )
        registry_ok = (
            REQUIRED_REGISTERED_BASELINES.issubset(registry_rows)
            and all(registered_sources.get(name, False) for name in REQUIRED_REGISTERED_BASELINES)
            and all(registry_rows[name].get("mode") for name in REQUIRED_REGISTERED_BASELINES)
            and all(verified_hashes.values())
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        registry_rows = {}
        registered_sources = {}
        verified_hashes = {}
        registry_ok = False

    checks = {
        "datasets": DATASETS == {"femnist", "cifar10", "tinyimagenet"},
        "attacks": ATTACK_NAMES.issubset(ATTACKS),
        "compatibility_attack_available": COMPATIBILITY_ATTACK_NAMES.issubset(ATTACKS),
        "primary_aggregations": PRIMARY_BASELINES.issubset(
            SUPPORTED_AGGREGATIONS & PLAINTEXT_AGGREGATIONS
        ),
        "unified_robustness_baselines": ADAPTER_AGGREGATIONS.issubset(
            SUPPORTED_AGGREGATIONS
        ),
        "comparison_modules": all(comparison_modules.values()),
        "protocol_modules": all(protocol.values()),
        "experiment_configs": all(configs.values()),
        "paper_experiment_matrix": matrix_ok,
        "external_baselines_declared": external_manifest_ok,
        "baseline_registry": registry_ok,
    }
    external_installations: dict[str, dict[str, object]] = {}
    for item in external_manifest.get("baselines", []):
        name = str(item["name"])
        if item.get("mode") == "in-tree-analytical":
            source = base / str(item.get("source", ""))
            external_installations[name] = {
                "ready": source.is_file(), "mode": "in-tree-analytical",
                "source": str(item.get("source", "")),
            }
        elif item.get("expected_path") and item.get("commit"):
            external_installations[name] = _git_checkout_status(
                base / str(item["expected_path"]), str(item["commit"])
            )
        else:
            external_installations[name] = {
                "ready": False, "reason": "manifest lacks exact path or commit"
            }
    external_ready = bool(external_installations) and all(
        item.get("ready") is True for item in external_installations.values()
    )
    try:
        results_manifest = json.loads((base / RESULTS_MANIFEST).read_text(encoding="utf-8"))
        expected_groups = set(results_manifest["expected_groups"])
        completed_groups = set(results_manifest["completed_groups"])
        result_files = [base / str(row["path"]) for row in results_manifest["artifacts"]]
        results_manifest_valid = (
            results_manifest.get("schema_version") == 1
            and expected_groups == {"convergence", "privacy", "robustness", "efficiency", "ablation"}
            and completed_groups.issubset(expected_groups)
        )
        formal_results_complete = (
            results_manifest_valid and completed_groups == expected_groups
            and bool(result_files) and all(path.is_file() for path in result_files)
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        results_manifest_valid = False
        formal_results_complete = False
    return {
        "schema_version": 1,
        "scope": (
            "in-tree RAIN implementation and experiment coverage; external baseline "
            "installations and completed GPU results are reported separately"
        ),
        "datasets": sorted(DATASETS),
        "attacks": {name: name in ATTACKS for name in sorted(ATTACK_NAMES)},
        "primary_aggregations": {
            name: name in SUPPORTED_AGGREGATIONS and name in PLAINTEXT_AGGREGATIONS
            for name in sorted(PRIMARY_BASELINES)
        },
        "unified_comparison_aggregations": {
            name: name in SUPPORTED_AGGREGATIONS for name in sorted(ADAPTER_AGGREGATIONS)
        },
        "comparison_modules": comparison_modules,
        "protocol_modules": protocol,
        "experiment_configs": configs,
        "paper_experiment_matrix": matrix_ok,
        "external_baselines": {
            "manifest_valid": external_manifest_ok,
            "installations": external_installations,
        },
        "baseline_registry": {
            "valid": registry_ok,
            "registered": sorted(registry_rows),
            "sources": registered_sources,
            "function_hashes_verified": verified_hashes,
        },
        "checks": checks,
        "implementation_complete": all(checks.values()),
        "external_reproduction_ready": external_ready,
        "reproduction_environment_ready": all(checks.values()) and external_ready,
        "results_manifest_valid": results_manifest_valid,
        "formal_results_complete": formal_results_complete,
        "calibration_files": calibration_files,
        "locked_calibration_ready": all(
            row["valid"] is True for row in calibration_files.values()
        ),
        "open_science": open_science,
        "anonymous_package_ready": open_science["anonymous_package_ready"],
        "open_science_submission_ready": open_science["open_science_submission_ready"],
        "full_paper_reproduction_ready": (
            all(checks.values()) and external_ready and formal_results_complete
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output")
    parser.add_argument(
        "--strict-external", action="store_true",
        help="also fail unless declared third-party baseline checkouts are installed",
    )
    parser.add_argument(
        "--strict-results", action="store_true",
        help="also fail unless every locked formal result is present in the results manifest",
    )
    parser.add_argument(
        "--strict-open-science", action="store_true",
        help="also fail unless the package is anonymous and its review URL is configured",
    )
    args = parser.parse_args()
    report = run_audit(args.root)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not report["implementation_complete"]:
        raise SystemExit(1)
    if args.strict_external and not report["external_reproduction_ready"]:
        raise SystemExit(2)
    if args.strict_results and not report["formal_results_complete"]:
        raise SystemExit(3)
    if args.strict_open_science and not report["open_science_submission_ready"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
