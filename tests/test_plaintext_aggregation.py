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


def test_flod_preserves_exact_zero_sign_while_rain_encodes_a_bit():
    updates = np.asarray([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    reference = np.asarray([0, 1], dtype=np.uint8)
    flod = aggregate_plaintext(
        updates, method="flod", reference_bits=reference, tau=1.0,
    )
    rain = aggregate_plaintext(
        updates, method="rain", reference_bits=reference, tau=1.0,
    )
    assert flod.update[0] == 0.0
    assert rain.update[0] == 1.0
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


def test_geometry_baselines_are_available_in_unified_trainer():
    updates = np.asarray([
        [1.0, 1.0], [1.1, .9], [.9, 1.1], [1.0, 1.2], [20.0, -20.0],
    ], dtype=np.float32)
    krum = aggregate_plaintext(updates, method="krum", malicious_clients=1)
    assert np.linalg.norm(krum.update - np.asarray([1.0, 1.0])) < .25
    trimmed = aggregate_plaintext(updates, method="trim-mean", malicious_clients=1)
    assert np.allclose(trimmed.update, np.asarray([1.0333333, 1.0]), atol=1e-5)
    median = aggregate_plaintext(updates, method="median")
    assert np.allclose(median.update, np.asarray([1.0, 1.0]))


def test_foundationfl_median_uses_paper_extreme_distance_selection():
    updates = np.asarray([
        [0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [100.0, -100.0],
    ], dtype=np.float32)
    result = aggregate_plaintext(
        updates, method="foundationfl", synthetic_ratio=1.0
    )
    maximum, minimum = updates.max(axis=0), updates.min(axis=0)
    scores = np.minimum(
        np.linalg.norm(updates - maximum, axis=1),
        np.linalg.norm(updates - minimum, axis=1),
    )
    selected = updates[np.argmax(scores)]
    expected = np.median(
        np.concatenate((updates, np.repeat(selected[None, :], len(updates), axis=0))),
        axis=0,
    )
    assert np.array_equal(result.update, expected)


def test_fltrust_and_rflpa_share_reference_cosine_rule():
    updates = np.asarray([
        [2.0, 0.0], [1.0, 1.0], [-2.0, 0.0],
    ], dtype=np.float32)
    reference = np.asarray([1.0, 0.0], dtype=np.float32)
    fltrust = aggregate_plaintext(
        updates, method="fltrust", reference_update=reference
    )
    rflpa = aggregate_plaintext(
        updates, method="rflpa", reference_update=reference
    )
    assert np.allclose(fltrust.update, rflpa.update)
    assert fltrust.metrics.accepted_clients == 2


def test_calibration_groups_are_deterministic_complete_and_nonempty():
    indices = np.arange(60, dtype=np.int64)
    targets = indices % 6
    left = _dirichlet_calibration_groups(indices, targets, clients=8, alpha=.5, seed=9)
    right = _dirichlet_calibration_groups(indices, targets, clients=8, alpha=.5, seed=9)
    assert [group.tolist() for group in left] == [group.tolist() for group in right]
    assert all(len(group) for group in left)
    assert sorted(np.concatenate(left).tolist()) == indices.tolist()
