import numpy as np
import pytest

from rain.client.encoder import encode_signs
from rain.client.randomizer import ClientRandomizer


def test_dense_sign_encoding_maps_zero_to_positive():
    values = np.array([-2.0, -0.0, 0.0, 3.0])
    assert np.array_equal(encode_signs(values), np.array([0, 1, 1, 1], dtype=np.uint8))


def test_randomizer_is_deterministic_for_fixed_seed():
    updates = np.arange(24, dtype=np.float64).reshape(4, 6) - 10
    left = ClientRandomizer(clip_norm=2.0, noise_multiplier=0.4, seed=17)
    right = ClientRandomizer(clip_norm=2.0, noise_multiplier=0.4, seed=17)
    assert np.array_equal(left.randomize_batch(updates).bits, right.randomize_batch(updates).bits)


def test_randomizer_rejects_non_finite_update():
    randomizer = ClientRandomizer(clip_norm=1.0, noise_multiplier=0.0)
    with pytest.raises(ValueError, match="finite"):
        randomizer.randomize(np.array([0.0, np.nan]))
