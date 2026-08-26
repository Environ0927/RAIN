import numpy as np
import pytest

from rain.errors import ProtocolError
from rain.protocol.arithmetic_sharing import (
    decode_signed,
    encode_signed,
    reconstruct_arithmetic,
    share_arithmetic,
    validate_aggregation_bounds,
)
from rain.protocol.boolean_sharing import reconstruct_bits, share_bits
from rain.protocol.secure_engine import PrimitiveSession
from rain.transport import LogicalTransport


def _session(seed=7):
    return PrimitiveSession(
        round_id=3,
        batch_id=4,
        seed=seed,
        transport=LogicalTransport(),
        chunk_size=16,
    )


def test_arithmetic_reconstruction_and_signed_encoding():
    signed = np.array([[-9], [0], [17]], dtype=np.int64)
    encoded = encode_signed(signed)
    left, right = share_arithmetic(encoded, rng=np.random.default_rng(1))
    assert not np.array_equal(left, encoded)
    assert np.array_equal(decode_signed(reconstruct_arithmetic(left, right)), signed)


def test_batched_dabit_b2a_does_not_open_target_bits():
    bits = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)
    b0, b1 = share_bits(bits, rng=np.random.default_rng(2))
    session = _session()
    session.load_boolean_share0("b", b0)
    session.load_boolean_share1("b", b1)
    session.b2a("b", "a")
    a0, a1 = session.final_arithmetic_shares("a")
    assert np.array_equal(reconstruct_arithmetic(a0, a1), bits.astype(np.uint64))
    assert session.transport.metrics.server_to_server_bytes > 0


def test_beaver_multiplication_and_triple_reuse_detection():
    x = encode_signed(np.array([[3], [-4], [9]], dtype=np.int64))
    y = encode_signed(np.array([[-2], [5], [7]], dtype=np.int64))
    x0, x1 = share_arithmetic(x, rng=np.random.default_rng(3))
    y0, y1 = share_arithmetic(y, rng=np.random.default_rng(4))
    session = _session()
    session.load_arithmetic_share0("x", x0)
    session.load_arithmetic_share1("x", x1)
    session.load_arithmetic_share0("y", y0)
    session.load_arithmetic_share1("y", y1)
    session.arithmetic_multiply("x", "y", "z", triple_id="one-time")
    z0, z1 = session.final_arithmetic_shares("z")
    expected = np.array([[-6], [-20], [63]], dtype=np.int64)
    assert np.array_equal(decode_signed(reconstruct_arithmetic(z0, z1)), expected)
    with pytest.raises(ProtocolError, match="reused"):
        session.arithmetic_multiply("x", "y", "z2", triple_id="one-time")


def test_secure_comparison_boundaries_and_signed_extraction():
    values = np.array([[0], [1], [4], [5], [6], [11]], dtype=np.int64)
    encoded = encode_signed(values)
    x0, x1 = share_arithmetic(encoded, rng=np.random.default_rng(10))
    session = _session(seed=11)
    session.load_arithmetic_share0("x", x0)
    session.load_arithmetic_share1("x", x1)
    session.compare_lt_public("x", 5, "lt")
    lt0, lt1 = session.final_boolean_shares("lt")
    assert np.array_equal(
        reconstruct_bits(lt0, lt1).reshape(-1),
        np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8),
    )

    signed_values = np.array([[-8], [-1], [0], [2]], dtype=np.int64)
    s0, s1 = share_arithmetic(encode_signed(signed_values), rng=np.random.default_rng(12))
    second = _session(seed=13)
    second.load_arithmetic_share0("z", s0)
    second.load_arithmetic_share1("z", s1)
    second.signed_nonnegative("z", "nonnegative")
    n0, n1 = second.final_boolean_shares("nonnegative")
    assert np.array_equal(
        reconstruct_bits(n0, n1).reshape(-1), np.array([0, 0, 1, 1], dtype=np.uint8)
    )


def test_bound_aware_comparison_uses_fewer_gates_with_same_result():
    values = np.array([[0], [4], [5], [11]], dtype=np.int64)
    encoded = encode_signed(values)
    shares = share_arithmetic(encoded, rng=np.random.default_rng(21))
    outputs = []
    gates = []
    for width in (4, 64):
        session = _session(seed=22)
        session.load_arithmetic_share0("x", shares[0])
        session.load_arithmetic_share1("x", shares[1])
        session.compare_lt_public("x", 5, "lt", width=width)
        outputs.append(reconstruct_bits(*session.final_boolean_shares("lt")))
        gates.append(session.metrics.boolean_and_gates)
    assert np.array_equal(outputs[0], outputs[1])
    assert gates[0] < gates[1] / 10


def test_overflow_bound_validation():
    validate_aggregation_bounds(clients=100, dimension=11_000_000)
    with pytest.raises(OverflowError):
        validate_aggregation_bounds(clients=1 << 62, dimension=3)


def test_primitive_party_registries_are_not_public():
    session = _session(seed=1)
    assert not hasattr(session, "party0")
    assert not hasattr(session, "party1")
