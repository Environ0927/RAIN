import numpy as np
import pytest

from rain.training.plaintext_aggregation import aggregate_plaintext
from rain.cli.calibrate_large import _dirichlet_calibration_groups


def test_plaintext_baselines_have_expected_updates():
    updates = np.asarray([
        [2.0, 1.0, -3.0],
        [1.0, -4.0, -2.0],
        [-1.0, 3.0, 1.0],
    ], dtype=np.float32)
    fedavg = aggregate_plaintext(updates, method="fedavg")
    assert np.allclose(fedavg.update, updates.mean(axis=0))
    signsgd = aggregate_plaintext(updates, method="signsgd")
    assert np.array_equal(signsgd.update, np.asarray([1.0, 1.0, -1.0]))


def test_plaintext_rain_and_flod_share_reference_weights_but_not_magnitude():
    updates = np.asarray([
        [1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ], dtype=np.float32)
    reference = np.asarray([1, 1, 0], dtype=np.uint8)
    rain = aggregate_plaintext(
        updates, method="rain", reference_bits=reference, tau=1.0,
    )
    flod = aggregate_plaintext(
        updates, method="flod", reference_bits=reference, tau=1.0,
    )
    assert np.array_equal(rain.update, np.asarray([1.0, 1.0, -1.0]))
    assert np.array_equal(np.sign(flod.update), rain.update)
    assert np.any(np.abs(flod.update) < 1.0)
    assert rain.metrics.accepted_clients == 2
    assert rain.metrics.threshold_count == 3
    assert rain.metrics.mismatch_min == 0.0
    assert rain.metrics.mismatch_max == 1.0


def test_plaintext_preprocessing_is_deterministic_and_validated():
    updates = np.asarray([[3.0, 4.0], [-4.0, 3.0]], dtype=np.float32)
    kwargs = dict(
        method="fedavg", preprocessing="clip_noise", clip_norm=1.0,
        noise_multiplier=.1, seed=7,
    )
    left = aggregate_plaintext(updates, **kwargs)
    right = aggregate_plaintext(updates, **kwargs)
    assert np.array_equal(left.update, right.update)
    with pytest.raises(ValueError, match="must be zero"):
        aggregate_plaintext(updates, method="fedavg", noise_multiplier=.1)


def test_calibration_groups_are_deterministic_complete_and_nonempty():
    indices = np.arange(60, dtype=np.int64)
    targets = indices % 6
    left = _dirichlet_calibration_groups(indices, targets, clients=8, alpha=.5, seed=9)
    right = _dirichlet_calibration_groups(indices, targets, clients=8, alpha=.5, seed=9)
    assert [group.tolist() for group in left] == [group.tolist() for group in right]
    assert all(len(group) for group in left)
    assert sorted(np.concatenate(left).tolist()) == indices.tolist()
