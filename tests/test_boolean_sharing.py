import numpy as np
import pytest

from rain.errors import ShareValidationError
from rain.protocol.boolean_sharing import reconstruct_bits, share_bits, validate_bits


def test_share_reconstruction_and_local_view_hides_plaintext():
    plaintext = np.tile(np.array([[0, 1]], dtype=np.uint8), (16, 16))
    share0, share1 = share_bits(plaintext, rng=np.random.default_rng(1234))

    assert np.array_equal(reconstruct_bits(share0, share1), plaintext)
    assert not np.array_equal(share0, plaintext)
    assert not np.array_equal(share1, plaintext)


def test_fixed_seed_is_reproducible():
    bits = np.zeros((3, 11), dtype=np.uint8)
    first = share_bits(bits, rng=np.random.default_rng(99))
    second = share_bits(bits, rng=np.random.default_rng(99))
    assert all(np.array_equal(left, right) for left, right in zip(first, second))


@pytest.mark.parametrize(
    "bad",
    [
        np.array([[0, 2]], dtype=np.uint8),
        np.array([[0, 1]], dtype=np.int64),
        np.empty((0, 2), dtype=np.uint8),
    ],
)
def test_malformed_bits_are_rejected(bad):
    with pytest.raises(ShareValidationError):
        validate_bits(bad)


def test_share_shape_mismatch_is_rejected():
    with pytest.raises(ShareValidationError, match="expected"):
        reconstruct_bits(
            np.zeros((2, 3), dtype=np.uint8),
            np.zeros((2, 4), dtype=np.uint8),
        )
