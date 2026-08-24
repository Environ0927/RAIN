"""Chunked secret-shared implementation of paper Algorithm 3."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from rain.metrics import CommunicationSnapshot
from rain.transport import LogicalTransport

from .arithmetic_sharing import validate_aggregation_bounds
from .boolean_sharing import validate_bits
from .secure_engine import PrimitiveMetrics, PrimitiveSession


@dataclass(frozen=True, slots=True)
class SecureAggregationResult:
    server0_direction_share: np.ndarray
    server1_direction_share: np.ndarray
    threshold_count: int
    communication: CommunicationSnapshot
    primitives: PrimitiveMetrics
    elapsed_seconds: float
    server0_seconds: float
    server1_seconds: float
    offline_seconds: float


def threshold_count(dimension: int, tau: float) -> int:
    if dimension <= 0:
        raise ValueError("dimension must be positive")
    if not np.isfinite(tau) or not 0 < tau < 0.5:
        raise ValueError("tau must be finite and in (0, 0.5)")
    count = math.floor(dimension * float(tau))
    if count <= 0:
        raise ValueError("tau is too small for this dimension (floor(d*tau) == 0)")
    return count


def secure_rain_aggregate(
    *,
    shuffled_share0: np.ndarray,
    shuffled_share1: np.ndarray,
    reference_bits: np.ndarray,
    tau: float,
    round_id: int,
    batch_id: int,
    seed: int,
    transport: LogicalTransport,
    chunk_size: int = 4096,
) -> SecureAggregationResult:
    """Run Algorithm 3 and expose only final direction shares.

    Hamming counts, acceptance masks, weights, and weighted contributions remain
    inside the two isolated primitive-party registries.  Weight normalization is
    omitted because division by the same positive denominator cannot change the
    final coordinate sign.  A zero weighted sum deterministically maps to +1.
    """

    started = time.perf_counter()
    left = validate_bits(shuffled_share0, name="shuffled_share0")
    right = validate_bits(
        shuffled_share1, name="shuffled_share1", expected_shape=left.shape
    )
    if left.ndim != 2:
        raise ValueError("shuffled shares must be client-by-coordinate matrices")
    clients, dimension = left.shape
    reference = validate_bits(reference_bits, name="reference_bits")
    if reference.shape not in ((dimension,), (1, dimension)):
        raise ValueError(f"reference_bits must have shape ({dimension},)")
    reference = reference.reshape(1, dimension)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    validate_aggregation_bounds(clients=clients, dimension=dimension)
    threshold = threshold_count(dimension, tau)
    hamming_width = max(1, dimension.bit_length())
    signed_accumulator_width = max(2, (clients * threshold).bit_length() + 1)
    session = PrimitiveSession(
        round_id=round_id,
        batch_id=batch_id,
        seed=seed,
        transport=transport,
        chunk_size=chunk_size,
    )

    # A1-A2: mismatch B2A and streaming Hamming accumulation.
    zero_hd = np.zeros((clients, 1), dtype=np.uint64)
    session.arithmetic_public("hd:0", zero_hd)
    hd_handle = "hd:0"
    for index, start in enumerate(range(0, dimension, chunk_size)):
        stop = min(start + chunk_size, dimension)
        prefix = f"hdchunk:{index}:"
        sign = prefix + "sign"
        session.load_boolean_share0(sign, left[:, start:stop])
        session.load_boolean_share1(sign, right[:, start:stop])
        mismatch = prefix + "mismatch"
        public_reference = np.broadcast_to(reference[:, start:stop], (clients, stop - start))
        session.xor_public(sign, public_reference, mismatch)
        mismatch_a = prefix + "mismatch_a"
        session.b2a(mismatch, mismatch_a)
        partial = prefix + "partial"
        session.arithmetic_sum(mismatch_a, partial, axis=1, keepdims=True)
        next_hd = f"hd:{index + 1}"
        session.arithmetic_add(hd_handle, partial, next_hd)
        session.drop_prefix(prefix)
        hd_handle = next_hd

    # A3: m_i = [hd_i < T], w_i = (T-hd_i)*m_i.
    mask_b = "weight:mask_b"
    session.compare_lt_public(hd_handle, threshold, mask_b, width=hamming_width)
    mask_a = "weight:mask_a"
    session.b2a(mask_b, mask_a)
    threshold_public = np.full((clients, 1), threshold, dtype=np.uint64)
    session.arithmetic_public("weight:T", threshold_public)
    session.arithmetic_sub("weight:T", hd_handle, "weight:delta")
    session.arithmetic_multiply("weight:delta", mask_a, "weight:value")
    session.drop_prefix("weight:mask_b:")

    # A2/A4/A5: convert signs, multiply by weights, sum, and extract sign,
    # processing model coordinates in bounded-memory chunks.
    output0: list[np.ndarray] = []
    output1: list[np.ndarray] = []
    for index, start in enumerate(range(0, dimension, chunk_size)):
        stop = min(start + chunk_size, dimension)
        width = stop - start
        prefix = f"aggchunk:{index}:"
        sign_b = prefix + "sign_b"
        session.load_boolean_share0(sign_b, left[:, start:stop])
        session.load_boolean_share1(sign_b, right[:, start:stop])
        sign_a = prefix + "sign_a"
        session.b2a(sign_b, sign_a)
        signed = prefix + "signed"
        session.arithmetic_linear(sign_a, signed, factor=2, offset=-1)
        weight_matrix = prefix + "weights"
        session.arithmetic_broadcast("weight:value", weight_matrix, (clients, width))
        weighted = prefix + "weighted"
        session.arithmetic_multiply(weight_matrix, signed, weighted)
        z_row = prefix + "zrow"
        session.arithmetic_sum(weighted, z_row, axis=0, keepdims=True)
        z_col = prefix + "zcol"
        session.arithmetic_transpose(z_row, z_col)
        direction = prefix + "direction"
        session.signed_nonnegative(
            z_col, direction, width=signed_accumulator_width
        )
        chunk0, chunk1 = session.final_boolean_shares(direction)
        output0.append(np.ascontiguousarray(chunk0.T))
        output1.append(np.ascontiguousarray(chunk1.T))
        session.drop_prefix(prefix)

    s0_seconds, s1_seconds = session.party_seconds
    return SecureAggregationResult(
        server0_direction_share=np.ascontiguousarray(np.concatenate(output0, axis=1)),
        server1_direction_share=np.ascontiguousarray(np.concatenate(output1, axis=1)),
        threshold_count=threshold,
        communication=transport.metrics,
        primitives=session.metrics,
        elapsed_seconds=time.perf_counter() - started,
        server0_seconds=s0_seconds,
        server1_seconds=s1_seconds,
        offline_seconds=session.offline_seconds,
    )
