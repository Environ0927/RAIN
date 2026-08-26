from dataclasses import fields

import numpy as np
import pytest

from rain.errors import ProtocolError, ShareValidationError
from rain.protocol.boolean_sharing import reconstruct_bits, share_bits
from rain.protocol.oracle import plaintext_shuffle_oracle
from rain.protocol.party import Server0State, Server1State
from rain.protocol.preprocessing import IdealShufflePreprocessor, ShufflePreprocessingCache
from rain.protocol.shuffle import run_secret_shared_shuffle
from rain.transport import LogicalTransport


def _run(*, clients, dimension, seed, chunk_size):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(clients, dimension), dtype=np.uint8)
    share0, share1 = share_bits(bits, rng=np.random.default_rng(seed + 1))
    permutation0 = np.random.default_rng(seed + 2).permutation(clients)
    permutation1 = np.random.default_rng(seed + 3).permutation(clients)
    preprocessed = IdealShufflePreprocessor().prepare(
        batch_size=clients,
        dimension=dimension,
        round_id=9,
        server0_seed=seed + 4,
        server1_seed=seed + 5,
        permutation0=permutation0,
        permutation1=permutation1,
    )
    result = run_secret_shared_shuffle(
        server0=preprocessed.server0,
        server1=preprocessed.server1,
        client_share0=share0,
        client_share1=share1,
        transport=LogicalTransport(),
        chunk_size=chunk_size,
        preprocessing_seconds=preprocessed.elapsed_seconds,
    )
    return bits, permutation0, permutation1, result


@pytest.mark.parametrize(
    "clients,dimension,seed,chunk_size",
    [
        (1, 1, 0, 1),
        (2, 7, 1, 3),
        (5, 32, 42, 8),
        (13, 71, 999, 16),
        (8, 17, 23, 128),
    ],
)
def test_shuffle_matches_oracle_and_preserves_multiset(
    clients, dimension, seed, chunk_size
):
    bits, permutation0, permutation1, result = _run(
        clients=clients,
        dimension=dimension,
        seed=seed,
        chunk_size=chunk_size,
    )
    reconstructed = reconstruct_bits(result.server0_share, result.server1_share)
    expected = plaintext_shuffle_oracle(bits, permutation0, permutation1)
    assert np.array_equal(reconstructed, expected)
    assert sorted(map(tuple, reconstructed.tolist())) == sorted(map(tuple, bits.tolist()))
    if clients * dimension > 1:
        assert not np.array_equal(result.server0_share, expected)
        assert not np.array_equal(result.server1_share, expected)
    assert result.communication.client_to_server_bytes > 0
    assert result.computation.offline_seconds > 0
    assert result.communication.online_bytes > 0


def test_chunk_size_does_not_change_output():
    outputs = []
    byte_counts = []
    for chunk_size in (1, 4, 9, 64):
        _, _, _, result = _run(
            clients=6, dimension=19, seed=101, chunk_size=chunk_size
        )
        outputs.append(reconstruct_bits(result.server0_share, result.server1_share))
        byte_counts.append(result.communication.total_bytes)
    assert all(np.array_equal(outputs[0], output) for output in outputs[1:])
    assert byte_counts[0] > byte_counts[-1]


def test_complete_protocol_is_reproducible_for_fixed_seed():
    first = _run(clients=7, dimension=21, seed=88, chunk_size=5)[3]
    second = _run(clients=7, dimension=21, seed=88, chunk_size=5)[3]
    assert np.array_equal(first.server0_share, second.server0_share)
    assert np.array_equal(first.server1_share, second.server1_share)
    assert first.communication == second.communication


def test_party_state_types_do_not_contain_other_partys_secrets():
    s0_fields = {item.name for item in fields(Server0State)}
    s1_fields = {item.name for item in fields(Server1State)}
    assert {"a1", "delta"}.isdisjoint(s0_fields)
    assert {"a0", "c0"}.isdisjoint(s1_fields)
    preprocessed = IdealShufflePreprocessor().prepare(
        batch_size=2, dimension=3, round_id=1, server0_seed=2, server1_seed=3
    )
    assert not hasattr(preprocessed.server0, "__dict__")
    assert not hasattr(preprocessed.server1, "__dict__")


def test_online_without_client_input_raises_explicit_protocol_error():
    preprocessed = IdealShufflePreprocessor().prepare(
        batch_size=2, dimension=3, round_id=1, server0_seed=2, server1_seed=3
    )
    with pytest.raises(ProtocolError, match="no client input"):
        preprocessed.server1.start_online()


def test_malformed_client_share_is_rejected_without_fallback():
    preprocessed = IdealShufflePreprocessor().prepare(
        batch_size=2, dimension=3, round_id=1, server0_seed=2, server1_seed=3
    )
    with pytest.raises(ShareValidationError, match="bits"):
        run_secret_shared_shuffle(
            server0=preprocessed.server0,
            server1=preprocessed.server1,
            client_share0=np.array([[0, 1, 2], [1, 0, 1]], dtype=np.uint8),
            client_share1=np.zeros((2, 3), dtype=np.uint8),
            transport=LogicalTransport(),
            chunk_size=2,
        )


def test_preprocessing_cache_is_one_time():
    cache = ShufflePreprocessingCache()
    cache.prepare(batch_size=2, dimension=3, round_id=4, batch_id=2,
                  server0_seed=1, server1_seed=2)
    assert len(cache) == 1
    cache.consume(round_id=4, batch_id=2)
    assert len(cache) == 0
    with pytest.raises(ValueError, match="no cached"):
        cache.consume(round_id=4, batch_id=2)
