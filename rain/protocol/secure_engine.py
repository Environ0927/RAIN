"""State-separated semi-honest two-party primitives used by Algorithm 3."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np

from rain.errors import ProtocolError, ShareValidationError
from rain.serialization import messages_to_array
from rain.transport import LogicalTransport
from rain.types import Endpoint, Message, MessageKind, ProtocolPhase

from .arithmetic_sharing import RING_BITS, share_arithmetic
from .boolean_sharing import share_bits, validate_bits


def _u64(value: np.ndarray, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.uint64 or value.ndim != 2:
        raise ShareValidationError(f"{name} must be a 2-D uint64 array")
    if value.size == 0:
        raise ShareValidationError(f"{name} must not be empty")
    return np.ascontiguousarray(value)


@dataclass(frozen=True, slots=True)
class PrimitiveMetrics:
    boolean_and_gates: int
    arithmetic_multiplications: int
    dabits_consumed: int
    boolean_triples_consumed: int
    arithmetic_triples_consumed: int


class PrimitiveParty:
    """One isolated party view for arithmetic and GMW Boolean computation."""

    __slots__ = (
        "role", "endpoint", "peer", "round_id", "batch_id", "_arith", "_bool",
        "_arith_triples", "_bool_triples", "_dabits", "_pending", "_consumed",
    )

    def __init__(self, role: int, *, round_id: int, batch_id: int) -> None:
        if role not in (0, 1):
            raise ValueError("role must be 0 or 1")
        self.role = role
        self.endpoint = Endpoint.S0 if role == 0 else Endpoint.S1
        self.peer = Endpoint.S1 if role == 0 else Endpoint.S0
        self.round_id = round_id
        self.batch_id = batch_id
        self._arith: dict[str, np.ndarray] = {}
        self._bool: dict[str, np.ndarray] = {}
        self._arith_triples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._bool_triples: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._dabits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._pending: dict[str, tuple] = {}
        self._consumed: set[str] = set()

    def put_arithmetic(self, handle: str, share: np.ndarray) -> None:
        if handle in self._arith:
            raise ProtocolError(f"arithmetic handle already exists: {handle}")
        self._arith[handle] = _u64(share, handle).copy()

    def put_boolean(self, handle: str, share: np.ndarray) -> None:
        if handle in self._bool:
            raise ProtocolError(f"Boolean handle already exists: {handle}")
        value = validate_bits(share, name=handle)
        if value.ndim != 2:
            raise ShareValidationError(f"{handle} must be a 2-D bit array")
        self._bool[handle] = value.copy()

    def boolean_shape(self, handle: str) -> tuple[int, int]:
        return self._bool[handle].shape

    def arithmetic_shape(self, handle: str) -> tuple[int, int]:
        return self._arith[handle].shape

    def local_arithmetic_add(self, left: str, right: str, target: str) -> None:
        self.put_arithmetic(target, np.add(self._arith[left], self._arith[right], dtype=np.uint64))

    def local_arithmetic_sub(self, left: str, right: str, target: str) -> None:
        self.put_arithmetic(target, np.subtract(self._arith[left], self._arith[right], dtype=np.uint64))

    def local_arithmetic_sum(self, source: str, target: str, *, axis: int, keepdims: bool = True) -> None:
        self.put_arithmetic(
            target,
            np.sum(self._arith[source], axis=axis, keepdims=keepdims, dtype=np.uint64),
        )

    def local_arithmetic_broadcast(self, source: str, target: str, shape: tuple[int, int]) -> None:
        self.put_arithmetic(target, np.broadcast_to(self._arith[source], shape).copy())

    def local_arithmetic_transpose(self, source: str, target: str) -> None:
        self.put_arithmetic(target, np.ascontiguousarray(self._arith[source].T))

    def drop_prefix(self, prefix: str, *, keep: set[str] | None = None) -> None:
        keep = keep or set()
        for registry in (self._arith, self._bool):
            for handle in [name for name in registry if name.startswith(prefix) and name not in keep]:
                del registry[handle]

    def local_arithmetic_public(self, target: str, value: np.ndarray) -> None:
        public = _u64(value, target)
        self.put_arithmetic(target, public if self.role == 0 else np.zeros_like(public))

    def local_arithmetic_linear(self, source: str, target: str, *, factor: int, offset: int = 0) -> None:
        source_value = self._arith[source]
        factor_u = np.uint64(factor % (1 << 64))
        result = np.multiply(source_value, factor_u, dtype=np.uint64)
        if self.role == 0 and offset:
            result = np.add(result, np.uint64(offset % (1 << 64)), dtype=np.uint64)
        self.put_arithmetic(target, result)

    def local_boolean_xor(self, left: str, right: str, target: str) -> None:
        self.put_boolean(target, np.bitwise_xor(self._bool[left], self._bool[right]))

    def local_boolean_xor_public(self, source: str, public: np.ndarray, target: str) -> None:
        bits = validate_bits(public, name="public bits", expected_shape=self._bool[source].shape)
        value = np.bitwise_xor(self._bool[source], bits) if self.role == 0 else self._bool[source].copy()
        self.put_boolean(target, value)

    def local_boolean_constant(self, target: str, value: int, shape: tuple[int, int]) -> None:
        if value not in (0, 1):
            raise ValueError("Boolean constant must be 0 or 1")
        local = np.full(shape, value if self.role == 0 else 0, dtype=np.uint8)
        self.put_boolean(target, local)

    def local_boolean_column(self, source: str, column: int, target: str) -> None:
        self.put_boolean(target, self._bool[source][:, column : column + 1])

    def local_boolean_stack(self, sources: list[str], target: str) -> None:
        self.put_boolean(target, np.concatenate([self._bool[source] for source in sources], axis=1))

    def provision_arithmetic_triple(
        self, triple_id: str, shares: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        if triple_id in self._arith_triples or triple_id in self._consumed:
            raise ProtocolError(f"arithmetic triple reused: {triple_id}")
        self._arith_triples[triple_id] = tuple(_u64(x, triple_id).copy() for x in shares)

    def provision_boolean_triple(
        self, triple_id: str, shares: tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> None:
        if triple_id in self._bool_triples or triple_id in self._consumed:
            raise ProtocolError(f"Boolean triple reused: {triple_id}")
        self._bool_triples[triple_id] = tuple(validate_bits(x, name=triple_id).copy() for x in shares)

    def provision_dabit(self, dabit_id: str, bool_share: np.ndarray, arith_share: np.ndarray) -> None:
        if dabit_id in self._dabits or dabit_id in self._consumed:
            raise ProtocolError(f"daBit reused: {dabit_id}")
        self._dabits[dabit_id] = (
            validate_bits(bool_share, name=dabit_id).copy(), _u64(arith_share, dabit_id).copy()
        )

    def prepare_arithmetic_multiply(self, op_id: str, left: str, right: str, triple_id: str) -> np.ndarray:
        if triple_id not in self._arith_triples:
            raise ProtocolError(f"missing or consumed arithmetic triple: {triple_id}")
        a, b, c = self._arith_triples.pop(triple_id)
        self._consumed.add(triple_id)
        x, y = self._arith[left], self._arith[right]
        if x.shape != y.shape or x.shape != a.shape:
            raise ShareValidationError("multiplication and triple shapes must match")
        d = np.subtract(x, a, dtype=np.uint64)
        e = np.subtract(y, b, dtype=np.uint64)
        self._pending[op_id] = ("amul", d, e, a, b, c)
        return np.ascontiguousarray(np.concatenate([d, e], axis=0))

    def finish_arithmetic_multiply(self, op_id: str, messages: list[Message], target: str) -> None:
        kind, d_local, e_local, a, b, c = self._pending.pop(op_id)
        if kind != "amul":
            raise ProtocolError("pending operation type mismatch")
        rows, cols = d_local.shape
        remote = messages_to_array(
            messages,
            expected_kind=MessageKind.ARITHMETIC_OPEN,
            expected_phase=ProtocolPhase.ONLINE,
            expected_sender=self.peer,
            expected_receiver=self.endpoint,
            expected_round_id=self.round_id,
            expected_batch_id=self.batch_id,
            expected_shape=(rows * 2, cols),
        )
        d = np.add(d_local, remote[:rows], dtype=np.uint64)
        e = np.add(e_local, remote[rows:], dtype=np.uint64)
        z = np.add(c, np.multiply(d, b, dtype=np.uint64), dtype=np.uint64)
        z = np.add(z, np.multiply(e, a, dtype=np.uint64), dtype=np.uint64)
        if self.role == 0:
            z = np.add(z, np.multiply(d, e, dtype=np.uint64), dtype=np.uint64)
        self.put_arithmetic(target, z)

    def prepare_boolean_and(self, op_id: str, left: str, right: str, triple_id: str) -> np.ndarray:
        if triple_id not in self._bool_triples:
            raise ProtocolError(f"missing or consumed Boolean triple: {triple_id}")
        a, b, c = self._bool_triples.pop(triple_id)
        self._consumed.add(triple_id)
        x, y = self._bool[left], self._bool[right]
        if x.shape != y.shape or x.shape != a.shape:
            raise ShareValidationError("AND inputs and triple shapes must match")
        d = np.bitwise_xor(x, a)
        e = np.bitwise_xor(y, b)
        self._pending[op_id] = ("band", d, e, a, b, c)
        return np.ascontiguousarray(np.concatenate([d, e], axis=0))

    def finish_boolean_and(self, op_id: str, messages: list[Message], target: str) -> None:
        kind, d_local, e_local, a, b, c = self._pending.pop(op_id)
        if kind != "band":
            raise ProtocolError("pending operation type mismatch")
        rows, cols = d_local.shape
        remote = messages_to_array(
            messages,
            expected_kind=MessageKind.BOOLEAN_OPEN,
            expected_phase=ProtocolPhase.ONLINE,
            expected_sender=self.peer,
            expected_receiver=self.endpoint,
            expected_round_id=self.round_id,
            expected_batch_id=self.batch_id,
            expected_shape=(rows * 2, cols),
            require_bits=True,
        )
        d = np.bitwise_xor(d_local, remote[:rows])
        e = np.bitwise_xor(e_local, remote[rows:])
        z = np.bitwise_xor(c, np.bitwise_and(d, b))
        z = np.bitwise_xor(z, np.bitwise_and(e, a))
        if self.role == 0:
            z = np.bitwise_xor(z, np.bitwise_and(d, e))
        self.put_boolean(target, z)

    def prepare_b2a(self, op_id: str, source: str, dabit_id: str) -> np.ndarray:
        if dabit_id not in self._dabits:
            raise ProtocolError(f"missing or consumed daBit: {dabit_id}")
        rb, ra = self._dabits.pop(dabit_id)
        self._consumed.add(dabit_id)
        value = self._bool[source]
        if value.shape != rb.shape:
            raise ShareValidationError("B2A input and daBit shapes must match")
        masked = np.bitwise_xor(value, rb)
        self._pending[op_id] = ("b2a", masked, ra)
        return masked

    def finish_b2a(self, op_id: str, messages: list[Message], target: str) -> None:
        kind, local_masked, ra = self._pending.pop(op_id)
        if kind != "b2a":
            raise ProtocolError("pending operation type mismatch")
        remote = messages_to_array(
            messages,
            expected_kind=MessageKind.BOOLEAN_OPEN,
            expected_phase=ProtocolPhase.ONLINE,
            expected_sender=self.peer,
            expected_receiver=self.endpoint,
            expected_round_id=self.round_id,
            expected_batch_id=self.batch_id,
            expected_shape=local_masked.shape,
            require_bits=True,
        )
        opened = np.bitwise_xor(local_masked, remote)
        if self.role == 0:
            inverted = np.subtract(np.uint64(1), ra, dtype=np.uint64)
        else:
            inverted = np.subtract(np.uint64(0), ra, dtype=np.uint64)
        result = np.where(opened == 0, ra, inverted).astype(np.uint64)
        self.put_arithmetic(target, result)

    def prepare_a2b_input(
        self, op_id: str, source: str, local_handle: str,
        rng: np.random.Generator, width: int = RING_BITS,
    ) -> np.ndarray:
        if not 1 <= width <= RING_BITS:
            raise ValueError("A2B width must be in [1, 64]")
        values = self._arith[source].reshape(-1)
        shifts = np.arange(width, dtype=np.uint64)
        bits = ((values[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.uint8)
        mask = rng.integers(0, 2, size=bits.shape, dtype=np.uint8)
        self.put_boolean(local_handle, mask)
        self._pending[op_id] = ("a2bin", bits.shape)
        return np.bitwise_xor(bits, mask)

    def finish_a2b_input(self, op_id: str, messages: list[Message], remote_handle: str) -> None:
        kind, shape = self._pending.pop(op_id)
        if kind != "a2bin":
            raise ProtocolError("pending operation type mismatch")
        remote_share = messages_to_array(
            messages,
            expected_kind=MessageKind.BOOLEAN_INPUT_SHARE,
            expected_phase=ProtocolPhase.ONLINE,
            expected_sender=self.peer,
            expected_receiver=self.endpoint,
            expected_round_id=self.round_id,
            expected_batch_id=self.batch_id,
            expected_shape=shape,
            require_bits=True,
        )
        self.put_boolean(remote_handle, remote_share)

    def export_final_boolean(self, handle: str) -> np.ndarray:
        return self._bool[handle].copy()

    def export_final_arithmetic(self, handle: str) -> np.ndarray:
        return self._arith[handle].copy()


class PrimitiveSession:
    """Coordinator that sees handles and masked wire payloads, never both secrets."""

    def __init__(
        self,
        *,
        round_id: int,
        batch_id: int,
        seed: int,
        transport: LogicalTransport,
        chunk_size: int = 4096,
    ) -> None:
        self.__party0 = PrimitiveParty(0, round_id=round_id, batch_id=batch_id)
        self.__party1 = PrimitiveParty(1, round_id=round_id, batch_id=batch_id)
        self.transport = transport
        self.round_id = round_id
        self.batch_id = batch_id
        self.chunk_size = chunk_size
        self._rng = np.random.default_rng(seed)
        self._counter = 0
        self._and_gates = 0
        self._amul = 0
        self._dabits_used = 0
        self._bool_triples_used = 0
        self._arith_triples_used = 0
        self._s0_seconds = 0.0
        self._s1_seconds = 0.0
        self._offline_seconds = 0.0

    def _p0(self, method: str, *args, **kwargs):
        started = time.perf_counter()
        value = getattr(self.__party0, method)(*args, **kwargs)
        self._s0_seconds += time.perf_counter() - started
        return value

    def _p1(self, method: str, *args, **kwargs):
        started = time.perf_counter()
        value = getattr(self.__party1, method)(*args, **kwargs)
        self._s1_seconds += time.perf_counter() - started
        return value

    def load_boolean_share0(self, handle: str, share: np.ndarray) -> None:
        self._p0("put_boolean", handle, share)

    def load_boolean_share1(self, handle: str, share: np.ndarray) -> None:
        self._p1("put_boolean", handle, share)

    def load_arithmetic_share0(self, handle: str, share: np.ndarray) -> None:
        self._p0("put_arithmetic", handle, share)

    def load_arithmetic_share1(self, handle: str, share: np.ndarray) -> None:
        self._p1("put_arithmetic", handle, share)

    def _id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}:{self._counter}"

    def _send_pair(self, left: np.ndarray, right: np.ndarray, *, kind: MessageKind, bits: bool):
        m01 = self.transport.send_array(
            left, kind=kind, phase=ProtocolPhase.ONLINE,
            sender=Endpoint.S0, receiver=Endpoint.S1,
            round_id=self.round_id, batch_id=self.batch_id,
            chunk_size=self.chunk_size, pack_bits=bits,
        )
        m10 = self.transport.send_array(
            right, kind=kind, phase=ProtocolPhase.ONLINE,
            sender=Endpoint.S1, receiver=Endpoint.S0,
            round_id=self.round_id, batch_id=self.batch_id,
            chunk_size=self.chunk_size, pack_bits=bits,
        )
        return m10, m01

    def xor(self, left: str, right: str, target: str) -> None:
        self._p0("local_boolean_xor", left, right, target)
        self._p1("local_boolean_xor", left, right, target)

    def xor_public(self, source: str, public: np.ndarray, target: str) -> None:
        self._p0("local_boolean_xor_public", source, public, target)
        self._p1("local_boolean_xor_public", source, public, target)

    def bool_constant(self, target: str, value: int, shape: tuple[int, int]) -> None:
        self._p0("local_boolean_constant", target, value, shape)
        self._p1("local_boolean_constant", target, value, shape)

    def bool_column(self, source: str, column: int, target: str) -> None:
        self._p0("local_boolean_column", source, column, target)
        self._p1("local_boolean_column", source, column, target)

    def boolean_and(self, left: str, right: str, target: str, *, triple_id: str | None = None) -> str:
        offline_started = time.perf_counter()
        shape = self.__party0.boolean_shape(left)
        if shape != self.__party1.boolean_shape(left):
            raise ShareValidationError("party Boolean shapes disagree")
        tid = triple_id or self._id("bt")
        a = self._rng.integers(0, 2, size=shape, dtype=np.uint8)
        b = self._rng.integers(0, 2, size=shape, dtype=np.uint8)
        c = np.bitwise_and(a, b)
        a0, a1 = share_bits(a, rng=self._rng)
        b0, b1 = share_bits(b, rng=self._rng)
        c0, c1 = share_bits(c, rng=self._rng)
        self._p0("provision_boolean_triple", tid, (a0, b0, c0))
        self._p1("provision_boolean_triple", tid, (a1, b1, c1))
        self._offline_seconds += time.perf_counter() - offline_started
        op = self._id("band")
        out0 = self._p0("prepare_boolean_and", op, left, right, tid)
        out1 = self._p1("prepare_boolean_and", op, left, right, tid)
        to0, to1 = self._send_pair(out0, out1, kind=MessageKind.BOOLEAN_OPEN, bits=True)
        self._p0("finish_boolean_and", op, to0, target)
        self._p1("finish_boolean_and", op, to1, target)
        self._and_gates += int(np.prod(shape))
        self._bool_triples_used += int(np.prod(shape))
        return tid

    def b2a(self, source: str, target: str, *, dabit_id: str | None = None) -> str:
        offline_started = time.perf_counter()
        shape = self.__party0.boolean_shape(source)
        did = dabit_id or self._id("db")
        r = self._rng.integers(0, 2, size=shape, dtype=np.uint8)
        rb0, rb1 = share_bits(r, rng=self._rng)
        ra0, ra1 = share_arithmetic(r.astype(np.uint64), rng=self._rng)
        self._p0("provision_dabit", did, rb0, ra0)
        self._p1("provision_dabit", did, rb1, ra1)
        self._offline_seconds += time.perf_counter() - offline_started
        op = self._id("b2a")
        out0 = self._p0("prepare_b2a", op, source, did)
        out1 = self._p1("prepare_b2a", op, source, did)
        to0, to1 = self._send_pair(out0, out1, kind=MessageKind.BOOLEAN_OPEN, bits=True)
        self._p0("finish_b2a", op, to0, target)
        self._p1("finish_b2a", op, to1, target)
        self._dabits_used += int(np.prod(shape))
        return did

    def arithmetic_multiply(
        self, left: str, right: str, target: str, *, triple_id: str | None = None
    ) -> str:
        offline_started = time.perf_counter()
        shape = self.__party0.arithmetic_shape(left)
        tid = triple_id or self._id("at")
        a = self._rng.integers(0, np.iinfo(np.uint64).max, size=shape, endpoint=True, dtype=np.uint64)
        b = self._rng.integers(0, np.iinfo(np.uint64).max, size=shape, endpoint=True, dtype=np.uint64)
        c = np.multiply(a, b, dtype=np.uint64)
        a0, a1 = share_arithmetic(a, rng=self._rng)
        b0, b1 = share_arithmetic(b, rng=self._rng)
        c0, c1 = share_arithmetic(c, rng=self._rng)
        self._p0("provision_arithmetic_triple", tid, (a0, b0, c0))
        self._p1("provision_arithmetic_triple", tid, (a1, b1, c1))
        self._offline_seconds += time.perf_counter() - offline_started
        op = self._id("amul")
        out0 = self._p0("prepare_arithmetic_multiply", op, left, right, tid)
        out1 = self._p1("prepare_arithmetic_multiply", op, left, right, tid)
        to0, to1 = self._send_pair(out0, out1, kind=MessageKind.ARITHMETIC_OPEN, bits=False)
        self._p0("finish_arithmetic_multiply", op, to0, target)
        self._p1("finish_arithmetic_multiply", op, to1, target)
        self._amul += int(np.prod(shape))
        self._arith_triples_used += int(np.prod(shape))
        return tid

    def arithmetic_public(self, target: str, value: np.ndarray) -> None:
        self._p0("local_arithmetic_public", target, value)
        self._p1("local_arithmetic_public", target, value)

    def arithmetic_add(self, left: str, right: str, target: str) -> None:
        self._p0("local_arithmetic_add", left, right, target)
        self._p1("local_arithmetic_add", left, right, target)

    def arithmetic_sub(self, left: str, right: str, target: str) -> None:
        self._p0("local_arithmetic_sub", left, right, target)
        self._p1("local_arithmetic_sub", left, right, target)

    def arithmetic_sum(self, source: str, target: str, *, axis: int, keepdims: bool = True) -> None:
        self._p0("local_arithmetic_sum", source, target, axis=axis, keepdims=keepdims)
        self._p1("local_arithmetic_sum", source, target, axis=axis, keepdims=keepdims)

    def arithmetic_broadcast(self, source: str, target: str, shape: tuple[int, int]) -> None:
        self._p0("local_arithmetic_broadcast", source, target, shape)
        self._p1("local_arithmetic_broadcast", source, target, shape)

    def arithmetic_transpose(self, source: str, target: str) -> None:
        self._p0("local_arithmetic_transpose", source, target)
        self._p1("local_arithmetic_transpose", source, target)

    def arithmetic_linear(self, source: str, target: str, *, factor: int, offset: int = 0) -> None:
        self._p0("local_arithmetic_linear", source, target, factor=factor, offset=offset)
        self._p1("local_arithmetic_linear", source, target, factor=factor, offset=offset)

    def a2b(self, source: str, prefix: str, *, width: int = RING_BITS) -> str:
        if not 1 <= width <= RING_BITS:
            raise ValueError("A2B width must be in [1, 64]")
        shape = self.__party0.arithmetic_shape(source)
        if shape[1] != 1:
            raise ShareValidationError("A2B currently expects a column vector")
        op0, op1 = self._id("a2bin0"), self._id("a2bin1")
        p0_local, p1_local = prefix + ":part0", prefix + ":part1"
        out0 = self._p0("prepare_a2b_input", op0, source, p0_local, self._rng, width)
        out1 = self._p1("prepare_a2b_input", op1, source, p1_local, self._rng, width)
        to0, to1 = self._send_pair(out0, out1, kind=MessageKind.BOOLEAN_INPUT_SHARE, bits=True)
        self._p0("finish_a2b_input", op0, to0, p1_local)
        self._p1("finish_a2b_input", op1, to1, p0_local)

        n = shape[0]
        carry = prefix + ":carry:0"
        self.bool_constant(carry, 0, (n, 1))
        sum_bits: list[str] = []
        for bit in range(width):
            a_bit, b_bit = f"{prefix}:a:{bit}", f"{prefix}:b:{bit}"
            self.bool_column(p0_local, bit, a_bit)
            self.bool_column(p1_local, bit, b_bit)
            xor_ab = f"{prefix}:xor:{bit}"
            self.xor(a_bit, b_bit, xor_ab)
            sum_bit = f"{prefix}:sum:{bit}"
            self.xor(xor_ab, carry, sum_bit)
            sum_bits.append(sum_bit)
            ab, ct = f"{prefix}:ab:{bit}", f"{prefix}:ct:{bit}"
            self.boolean_and(a_bit, b_bit, ab)
            self.boolean_and(carry, xor_ab, ct)
            next_carry = f"{prefix}:carry:{bit + 1}"
            self.xor(ab, ct, next_carry)
            carry = next_carry
        # Materialize the 64 output bit columns locally without revealing them.
        self._p0("local_boolean_stack", sum_bits, prefix)
        self._p1("local_boolean_stack", sum_bits, prefix)
        return prefix

    def compare_lt_public(
        self, source: str, threshold: int, target: str, *, width: int = RING_BITS,
    ) -> None:
        if threshold < 0 or threshold >= (1 << 63):
            raise ValueError("public threshold must be in [0, 2**63)")
        if threshold >= (1 << width):
            raise ValueError("public threshold does not fit comparison width")
        bits_handle = self.a2b(source, target + ":bits", width=width)
        n = self.__party0.boolean_shape(bits_handle)[0]
        eq, lt = f"{target}:eq:{width}", f"{target}:lt:{width}"
        self.bool_constant(eq, 1, (n, 1))
        self.bool_constant(lt, 0, (n, 1))
        for bit in range(width - 1, -1, -1):
            xb = f"{target}:x:{bit}"
            self.bool_column(bits_handle, bit, xb)
            not_x = f"{target}:notx:{bit}"
            self.xor_public(xb, np.ones((n, 1), dtype=np.uint8), not_x)
            if (threshold >> bit) & 1:
                term = f"{target}:term:{bit}"
                self.boolean_and(eq, not_x, term)
                next_lt = f"{target}:lt:{bit}"
                self.xor(lt, term, next_lt)
                next_eq = f"{target}:eq:{bit}"
                self.boolean_and(eq, xb, next_eq)
            else:
                next_lt = lt
                next_eq = f"{target}:eq:{bit}"
                self.boolean_and(eq, not_x, next_eq)
            lt, eq = next_lt, next_eq
        # Copy into the requested stable handle.
        zero = target + ":zero"
        self.bool_constant(zero, 0, (n, 1))
        self.xor(lt, zero, target)

    def signed_nonnegative(
        self, source: str, target: str, *, width: int = RING_BITS,
    ) -> None:
        if not 2 <= width <= RING_BITS:
            raise ValueError("signed comparison width must be in [2, 64]")
        bits = self.a2b(source, target + ":bits", width=width)
        msb = target + ":msb"
        self.bool_column(bits, width - 1, msb)
        self.xor_public(msb, np.ones(self.__party0.boolean_shape(msb), dtype=np.uint8), target)

    def final_boolean_shares(self, handle: str) -> tuple[np.ndarray, np.ndarray]:
        return self._p0("export_final_boolean", handle), self._p1("export_final_boolean", handle)

    def final_arithmetic_shares(self, handle: str) -> tuple[np.ndarray, np.ndarray]:
        return self._p0("export_final_arithmetic", handle), self._p1("export_final_arithmetic", handle)

    def drop_prefix(self, prefix: str, *, keep: set[str] | None = None) -> None:
        self._p0("drop_prefix", prefix, keep=keep)
        self._p1("drop_prefix", prefix, keep=keep)

    @property
    def party_seconds(self) -> tuple[float, float]:
        return self._s0_seconds, self._s1_seconds

    @property
    def offline_seconds(self) -> float:
        return self._offline_seconds

    @property
    def metrics(self) -> PrimitiveMetrics:
        return PrimitiveMetrics(
            boolean_and_gates=self._and_gates,
            arithmetic_multiplications=self._amul,
            dabits_consumed=self._dabits_used,
            boolean_triples_consumed=self._bool_triples_used,
            arithmetic_triples_consumed=self._arith_triples_used,
        )
