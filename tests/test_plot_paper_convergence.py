import json

import numpy as np
import pytest

from rain.cli.plot_paper_convergence import (
    _as_panel, load_curve, parse_method_path, smooth_values,
)


def test_load_curve_preserves_raw_evaluated_points(tmp_path):
    path = tmp_path / "rounds.jsonl"
    rows = [
        {"round": 0, "validation_accuracy": 0.1},
        {"round": 1, "validation_accuracy": None},
        {"round": 2, "validation_accuracy": 0.3},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    rounds, values = load_curve(path)

    np.testing.assert_array_equal(rounds, [1, 3])
    np.testing.assert_allclose(values, [0.1, 0.3])


def test_load_curve_applies_one_based_round_limit(tmp_path):
    path = tmp_path / "rounds.jsonl"
    rows = [
        {"round": 0, "validation_accuracy": 0.1},
        {"round": 1, "validation_accuracy": 0.2},
        {"round": 2, "validation_accuracy": 0.3},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    rounds, values = load_curve(path, round_limit=2)

    np.testing.assert_array_equal(rounds, [1, 2])
    np.testing.assert_allclose(values, [0.1, 0.2])


def test_smooth_values_uses_edge_padded_centered_average():
    values = smooth_values(np.asarray([0.0, 0.6, 0.0]), 3)
    np.testing.assert_allclose(values, [0.2, 0.2, 0.2])


def test_smooth_values_rejects_even_window():
    with pytest.raises(ValueError, match="positive odd"):
        smooth_values(np.asarray([0.0, 1.0]), 2)


def test_parse_method_path_normalizes_method(tmp_path):
    method, path = parse_method_path(f"SignSGD={tmp_path / 'run.jsonl'}")
    assert method == "signsgd"
    assert path == tmp_path / "run.jsonl"


def test_panel_rejects_duplicate_methods(tmp_path):
    path = tmp_path / "run.jsonl"
    with pytest.raises(ValueError, match="duplicate curve"):
        _as_panel([("rain", path), ("rain", path)])
