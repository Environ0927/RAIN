import numpy as np

from rain.protocol.oracle import rain_aggregate_oracle
from rain.training.coordinator import RainRoundCoordinator


def test_complete_round_is_deterministic_and_matches_randomized_oracle():
    updates = np.random.default_rng(3).normal(size=(5, 19))
    reference = (np.arange(19) % 3 == 0).astype(np.uint8)
    coordinator = RainRoundCoordinator(clip_norm=100, noise_multiplier=0, tau=.4, chunk_size=7, seed=8)
    actual = coordinator.run_round(updates, reference, round_id=2)
    signs = (updates >= 0).astype(np.uint8)
    expected = rain_aggregate_oracle(signs, reference, tau=.4)
    assert np.array_equal(actual.direction_bits, expected.reshape(-1))
    repeated = coordinator.run_round(updates, reference, round_id=2)
    assert np.array_equal(actual.direction_bits, repeated.direction_bits)
    assert actual.threshold_count == repeated.threshold_count
    assert actual.metrics.client_to_server_bytes > 0
    assert actual.metrics.server_to_server_bytes > 0


def test_round_and_chunk_size_are_domain_separated():
    updates = np.random.default_rng(1).normal(size=(3, 11))
    reference = np.ones(11, dtype=np.uint8)
    small = RainRoundCoordinator(clip_norm=100, noise_multiplier=0, tau=.45, chunk_size=2, seed=4)
    large = RainRoundCoordinator(clip_norm=100, noise_multiplier=0, tau=.45, chunk_size=99, seed=4)
    assert np.array_equal(
        small.run_round(updates, reference, round_id=0).direction_bits,
        large.run_round(updates, reference, round_id=0).direction_bits,
    )
