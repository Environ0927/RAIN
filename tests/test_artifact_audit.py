from pathlib import Path

from rain.cli.audit_artifact import run_audit


def test_paper_implementation_coverage_is_complete():
    report = run_audit(Path(__file__).parents[1])
    assert report["implementation_complete"] is True
    assert all(report["checks"].values())
    assert report["paper_experiment_matrix"] is True
    assert report["baseline_registry"]["valid"] is True
    assert all(report["baseline_registry"]["retained_function_hashes"].values())
    # Third-party checkouts are deliberately not vendored. Their installation
    # state must be separate from the in-tree implementation audit.
    assert "external_reproduction_ready" in report
    assert report["external_baselines"]["installations"]["approx-shuffling"]["ready"] is True
    assert report["results_manifest_valid"] is True
    assert report["formal_results_complete"] is False
    assert "open_science" in report
    assert report["open_science"]["metadata_valid"] is True
    assert report["open_science"]["inventory_valid"] is True
    assert report["calibration_files"]["cifar10"]["valid"] is True
    assert report["calibration_files"]["femnist"]["valid"] is True
    assert report["calibration_files"]["tinyimagenet"]["valid"] is False
    assert report["locked_calibration_ready"] is False
