from dataclasses import fields

import numpy as np
import pytest

from rain.protocol.aggregation import SecureAggregationResult, secure_rain_aggregate
from rain.protocol.boolean_sharing import reconstruct_bits, share_bits
from rain.protocol.oracle import rain_aggregate_oracle
from rain.transport import LogicalTransport


def _aggregate(bits, reference, *, tau=0.4, chunk_size=8, seed=100):
    share0, share1 = share_bits(bits, rng=np.random.default_rng(9))
    result = secure_rain_aggregate(
        shuffled_share0=share0,
        shuffled_share1=share1,
        reference_bits=reference,
        tau=tau,
        round_id=2,
        batch_id=7,
        seed=seed,
        transport=LogicalTransport(),
        chunk_size=chunk_size,
    )
    output = reconstruct_bits(
        result.server0_direction_share, result.server1_direction_share
    )
    return output, result


@pytest.mark.parametrize("clients,dimension,seed", [(3, 9, 1), (7, 17, 2), (9, 25, 77)])
def test_secure_aggregation_matches_plaintext_oracle(clients, dimension, seed):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(clients, dimension), dtype=np.uint8)
    reference = rng.integers(0, 2, size=dimension, dtype=np.uint8)
    output, result = _aggregate(bits, reference, tau=0.44, chunk_size=7, seed=seed + 50)
    assert np.array_equal(output, rain_aggregate_oracle(bits, reference, tau=0.44))
    assert result.communication.server_to_server_bytes > 0
    assert result.primitives.dabits_consumed > 0
    assert result.primitives.arithmetic_triples_consumed > 0


def test_all_accepted_all_rejected_and_ties_match_oracle():
    reference = np.array([0, 1] * 8, dtype=np.uint8)
    accepted = np.tile(reference, (4, 1))
    output, _ = _aggregate(accepted, reference, tau=0.4)
    assert np.array_equal(output.reshape(-1), reference)

    rejected = np.tile(1 - reference, (4, 1)).astype(np.uint8)
    output, _ = _aggregate(rejected, reference, tau=0.4)
    assert np.array_equal(output, np.ones((1, reference.size), dtype=np.uint8))

    # Equal accepted positive/negative contributions create coordinate ties.
    mixed = np.stack([reference, reference.copy()], axis=0)
    mixed[1, :2] ^= 1
    output, _ = _aggregate(mixed, reference, tau=0.4)
    assert np.array_equal(output, rain_aggregate_oracle(mixed, reference, tau=0.4))


def test_chunk_sizes_produce_identical_reconstructed_direction():
    rng = np.random.default_rng(123)
    bits = rng.integers(0, 2, size=(5, 19), dtype=np.uint8)
    reference = rng.integers(0, 2, size=19, dtype=np.uint8)
    outputs = [
        _aggregate(bits, reference, tau=0.42, chunk_size=size, seed=44)[0]
        for size in (1, 4, 19, 64)
    ]
    assert all(np.array_equal(outputs[0], output) for output in outputs[1:])


def test_result_does_not_expose_private_intermediates():
    names = {field.name for field in fields(SecureAggregationResult)}
    assert {"hamming", "weights", "pass_mask", "contributions"}.isdisjoint(names)


def test_invalid_threshold_is_rejected():
    bits = np.ones((2, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="tau"):
        _aggregate(bits, bits[0], tau=0.5)
