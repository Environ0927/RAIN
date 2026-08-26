import json
from pathlib import Path

from rain.artifact_policy import check_open_science, scan_anonymity


def test_scan_anonymity_detects_repository_metadata(tmp_path):
    (tmp_path / ".git").mkdir()
    report = scan_anonymity(tmp_path)
    assert report["anonymous"] is False
    assert report["git_metadata_present"] is True


def test_open_science_files_are_machine_readable():
    root = Path(__file__).parents[1]
    report = check_open_science(root)
    assert all(report["required_files"].values())
    assert report["metadata_valid"] is True
    assert report["inventory_valid"] is True
    assert report["open_science_submission_ready"] is (
        report["anonymous_package_ready"] and report["anonymous_url_configured"]
    )
    metadata = json.loads((root / "paper/open_science.json").read_text(encoding="utf-8"))
    assert metadata["contact_information_in_review_artifact"] is False
