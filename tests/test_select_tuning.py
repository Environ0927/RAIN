import json

import pytest

from rain.cli.select_tuning import select_candidates


def _candidate(tmp_path, method, learning_rate, values):
    run = tmp_path / f"{method}-{learning_rate}"; run.mkdir()
    (run / "config.json").write_text(json.dumps({
        "protocol": {"aggregation": method},
        "training": {"learning_rate": learning_rate},
    }), encoding="utf-8")
    path = run / "rounds.jsonl"
    path.write_text("".join(json.dumps({
        "round": index, "validation_accuracy": value,
    }) + "\n" for index, value in enumerate(values)), encoding="utf-8")
    return path


def test_select_candidates_uses_validation_only_and_records_evidence(tmp_path):
    paths = [
        _candidate(tmp_path, "rain", .001, [.4, .6]),
        _candidate(tmp_path, "rain", .002, [.5, .55]),
        _candidate(tmp_path, "signsgd", .001, [.3, .45]),
        _candidate(tmp_path, "signsgd", .002, [.4, .5]),
    ]
    result = select_candidates(
        paths, expected_candidates=2, minimum_rounds=2,
        required_methods={"rain", "signsgd"},
    )
    assert result["selected"]["rain"]["learning_rate"] == .001
    assert result["selected"]["rain"]["best_round"] == 2
    assert result["selected"]["signsgd"]["learning_rate"] == .002


def test_select_candidates_rejects_incomplete_equal_budget(tmp_path):
    path = _candidate(tmp_path, "rain", .001, [.4])
    with pytest.raises(ValueError, match="expected 2"):
        select_candidates([path], expected_candidates=2)


def test_select_candidates_requires_declared_methods(tmp_path):
    path = _candidate(tmp_path, "rain", .001, [.4])
    with pytest.raises(ValueError, match="methods are"):
        select_candidates([path], required_methods={"rain", "signsgd"})
