import numpy as np
import pytest

from rain.privacy import account_privacy, calibrate_threshold, load_calibration


def test_accountant_deterministic_and_monotone():
    base = account_privacy(participating_clients=16, rounds=5, clip_norm=1, noise_multiplier=4)
    assert base == account_privacy(participating_clients=16, rounds=5, clip_norm=1, noise_multiplier=4)
    assert account_privacy(participating_clients=16, rounds=5, clip_norm=1, noise_multiplier=6).epsilon <= base.epsilon
    assert account_privacy(participating_clients=32, rounds=5, clip_norm=1, noise_multiplier=4).epsilon <= base.epsilon
    assert account_privacy(participating_clients=16, rounds=10, clip_norm=1, noise_multiplier=4).epsilon >= base.epsilon


@pytest.mark.parametrize("kwargs", [
    {"participating_clients": 0}, {"rounds": 0}, {"clip_norm": 0},
    {"noise_multiplier": 0}, {"delta": 1}, {"orders": [1]},
])
def test_accountant_rejects_invalid(kwargs):
    valid = dict(participating_clients=4, rounds=2, clip_norm=1, noise_multiplier=2, delta=1e-5)
    valid.update(kwargs)
    with pytest.raises(ValueError):
        account_privacy(**valid)


def test_calibration_is_reproducible_and_versioned(tmp_path):
    updates = np.random.default_rng(2).normal(size=(20, 17))
    reference = (np.arange(17) % 2).astype(np.uint8)
    first = calibrate_threshold(updates, reference, clip_norm=2, noise_multiplier=.3, seed=9)
    second = calibrate_threshold(updates, reference, clip_norm=2, noise_multiplier=.3, seed=9)
    assert first == second
    path = tmp_path / "tau.json"
    first.save(path)
    assert load_calibration(path) == first
    assert 0 <= first.tau <= 1
