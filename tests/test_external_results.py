import pytest

from rain.cli.import_external_results import COMMITS, validate_document


def test_external_measurement_is_tagged_and_retains_provenance():
    rows = validate_document([{
        "baseline": "camel", "source_commit": COMMITS["camel"],
        "dataset": "fmnist", "seed": 1, "metrics": {"accuracy": 0.91},
    }], "camel")
    assert rows[0]["source_kind"] == "external-pinned"
    assert rows[0]["metrics"]["accuracy"] == 0.91


def test_external_measurement_rejects_wrong_commit():
    with pytest.raises(ValueError, match="pinned source commit"):
        validate_document([{
            "baseline": "flguard", "source_commit": "main",
            "dataset": "cifar10", "seed": 1, "metrics": {"accuracy": 0.5},
        }], "flguard")
