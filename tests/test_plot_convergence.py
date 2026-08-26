import json

import numpy as np

from rain.cli.plot_convergence import aggregate_curves


def _curve(tmp_path, name, values):
    path = tmp_path / name
    path.write_text("".join(json.dumps({
        "round": 9 + index * 10, "validation_accuracy": value,
    }) + "\n" for index, value in enumerate(values)), encoding="utf-8")
    return path


def test_aggregate_curves_returns_seed_mean_and_std(tmp_path):
    paths = [_curve(tmp_path, "a.jsonl", [.4, .6]), _curve(tmp_path, "b.jsonl", [.6, .8])]
    rounds, mean, std = aggregate_curves(paths, "validation_accuracy")
    assert rounds == [10, 20]
    np.testing.assert_allclose(mean, [.5, .7])
    np.testing.assert_allclose(std, [.1, .1])
