"""Deterministic binary serialization for logical protocol messages."""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

import numpy as np

from .errors import MessageValidationError
from .types import Endpoint, Message, MessageKind, PROTOCOL_VERSION, ProtocolPhase


_MAGIC = b"RAIN"
_HEADER = struct.Struct("<4sBBBBBqqIIIIBQ")
_DTYPE_TO_CODE = {"uint8": 1, "int64": 2, "uint64": 3, "bit": 4}
_CODE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CODE.items()}
_NUMPY_DTYPES = {
    "uint8": np.dtype("<u1"),
    "int64": np.dtype("<i8"),
    "uint64": np.dtype("<u8"),
}


def _enum_value(enum_type: type, value: int, name: str):
    try:
        return enum_type(value)
    except ValueError as exc:
        raise MessageValidationError(f"unknown {name} value: {value}") from exc


def validate_message(message: Message) -> None:
    if message.version != PROTOCOL_VERSION:
        raise MessageValidationError(
            f"unsupported message version {message.version}; expected {PROTOCOL_VERSION}"
        )
    if message.round_id < 0:
        raise MessageValidationError("round_id must be non-negative")
    if message.batch_id < 0:
        raise MessageValidationError("batch_id must be non-negative")
    if message.total_chunks <= 0:
        raise MessageValidationError("total_chunks must be positive")
    if not 0 <= message.chunk_id < message.total_chunks:
        raise MessageValidationError("chunk_id is outside total_chunks")
    if message.rows <= 0 or message.cols <= 0:
        raise MessageValidationError("message array dimensions must be positive")
    if message.dtype not in _DTYPE_TO_CODE:
        raise MessageValidationError(f"unsupported dtype: {message.dtype}")
    logical_items = message.rows * message.cols
    expected = (
        math.ceil(logical_items / 8)
        if message.dtype == "bit"
        else logical_items * _NUMPY_DTYPES[message.dtype].itemsize
    )
    if len(message.payload) != expected:
        raise MessageValidationError(
            f"payload length {len(message.payload)} does not match shape/dtype ({expected})"
        )


def serialize_message(message: Message) -> bytes:
    """Serialize a message without pickle, padding, or platform dependence."""

    validate_message(message)
    header = _HEADER.pack(
        _MAGIC,
        message.version,
        int(message.kind),
        int(message.phase),
        int(message.sender),
        int(message.receiver),
        message.round_id,
        message.batch_id,
        message.chunk_id,
        message.total_chunks,
        message.rows,
        message.cols,
        _DTYPE_TO_CODE[message.dtype],
        len(message.payload),
    )
    return header + message.payload


def deserialize_message(wire: bytes) -> Message:
    if not isinstance(wire, bytes):
        raise MessageValidationError("wire message must be bytes")
    if len(wire) < _HEADER.size:
        raise MessageValidationError("wire message is shorter than the fixed header")
    fields = _HEADER.unpack(wire[: _HEADER.size])
    (
        magic,
        version,
        kind,
        phase,
        sender,
        receiver,
        round_id,
        batch_id,
        chunk_id,
        total_chunks,
        rows,
        cols,
        dtype_code,
        payload_length,
    ) = fields
    if magic != _MAGIC:
        raise MessageValidationError("invalid message magic")
    if payload_length != len(wire) - _HEADER.size:
        raise MessageValidationError("declared payload length does not match wire bytes")
    if dtype_code not in _CODE_TO_DTYPE:
        raise MessageValidationError(f"unknown dtype code: {dtype_code}")
    message = Message(
        version=version,
        kind=_enum_value(MessageKind, kind, "message kind"),
        phase=_enum_value(ProtocolPhase, phase, "protocol phase"),
        sender=_enum_value(Endpoint, sender, "sender"),
        receiver=_enum_value(Endpoint, receiver, "receiver"),
        round_id=round_id,
        batch_id=batch_id,
        chunk_id=chunk_id,
        total_chunks=total_chunks,
        rows=rows,
        cols=cols,
        dtype=_CODE_TO_DTYPE[dtype_code],
        payload=wire[_HEADER.size :],
    )
    validate_message(message)
    return message


def array_to_messages(
    array: np.ndarray,
    *,
    kind: MessageKind,
    phase: ProtocolPhase,
    sender: Endpoint,
    receiver: Endpoint,
    round_id: int,
    chunk_size: int,
    batch_id: int = 0,
    pack_bits: bool = False,
) -> list[Message]:
    """Split a 2-D array into deterministic column-major logical chunks."""

    if not isinstance(array, np.ndarray) or array.ndim != 2:
        raise MessageValidationError("message payload must be a 2-D numpy array")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise MessageValidationError("message payload dimensions must be positive")
    if chunk_size <= 0:
        raise MessageValidationError("chunk_size must be positive")
    dtype_name = "bit" if pack_bits else array.dtype.name
    if pack_bits and (array.dtype != np.uint8 or np.any(array > 1)):
        raise MessageValidationError("packed Boolean array must contain uint8 bits 0 or 1")
    if dtype_name not in _DTYPE_TO_CODE:
        raise MessageValidationError(f"unsupported array dtype: {array.dtype}")
    canonical_dtype = np.uint8 if pack_bits else _NUMPY_DTYPES[dtype_name]
    canonical = np.asarray(array, dtype=canonical_dtype, order="C")
    total_chunks = math.ceil(canonical.shape[1] / chunk_size)
    messages: list[Message] = []
    for chunk_id in range(total_chunks):
        start = chunk_id * chunk_size
        stop = min(start + chunk_size, canonical.shape[1])
        chunk = np.ascontiguousarray(canonical[:, start:stop])
        messages.append(
            Message(
                version=PROTOCOL_VERSION,
                kind=kind,
                phase=phase,
                sender=sender,
                receiver=receiver,
                round_id=round_id,
                batch_id=batch_id,
                chunk_id=chunk_id,
                total_chunks=total_chunks,
                rows=chunk.shape[0],
                cols=chunk.shape[1],
                dtype=dtype_name,
                payload=(
                    np.packbits(chunk.reshape(-1), bitorder="little").tobytes()
                    if pack_bits
                    else chunk.tobytes(order="C")
                ),
            )
        )
    return messages


def messages_to_array(
    messages: Sequence[Message],
    *,
    expected_kind: MessageKind,
    expected_phase: ProtocolPhase,
    expected_sender: Endpoint,
    expected_receiver: Endpoint,
    expected_round_id: int,
    expected_batch_id: int = 0,
    expected_shape: tuple[int, int] | None = None,
    require_bits: bool = False,
) -> np.ndarray:
    """Validate and reassemble a complete sequence of array chunks."""

    if not messages:
        raise MessageValidationError("message sequence is empty")
    for message in messages:
        validate_message(message)
        expected_metadata = (
            ("kind", message.kind, expected_kind),
            ("phase", message.phase, expected_phase),
            ("sender", message.sender, expected_sender),
            ("receiver", message.receiver, expected_receiver),
            ("round_id", message.round_id, expected_round_id),
            ("batch_id", message.batch_id, expected_batch_id),
        )
        for name, actual, expected in expected_metadata:
            if actual != expected:
                raise MessageValidationError(
                    f"unexpected {name}: got {actual!r}, expected {expected!r}"
                )

    total_chunks = messages[0].total_chunks
    if len(messages) != total_chunks:
        raise MessageValidationError(
            f"incomplete chunk sequence: got {len(messages)}, expected {total_chunks}"
        )
    ids = [message.chunk_id for message in messages]
    if len(set(ids)) != len(ids):
        raise MessageValidationError("duplicate chunk_id in message sequence")
    if sorted(ids) != list(range(total_chunks)):
        raise MessageValidationError("missing or invalid chunk_id in message sequence")

    ordered = sorted(messages, key=lambda item: item.chunk_id)
    first = ordered[0]
    for message in ordered:
        if message.total_chunks != total_chunks:
            raise MessageValidationError("inconsistent total_chunks metadata")
        if message.rows != first.rows:
            raise MessageValidationError("inconsistent row count across chunks")
        if message.dtype != first.dtype:
            raise MessageValidationError("inconsistent dtype across chunks")

    if first.dtype == "bit":
        chunks = [
            np.unpackbits(
                np.frombuffer(message.payload, dtype=np.uint8), bitorder="little"
            )[: message.rows * message.cols]
            .reshape(message.rows, message.cols)
            .astype(np.uint8, copy=False)
            for message in ordered
        ]
    else:
        dtype = _NUMPY_DTYPES[first.dtype]
        chunks = [
            np.frombuffer(message.payload, dtype=dtype)
            .reshape(message.rows, message.cols)
            .copy()
            for message in ordered
        ]
    array = np.ascontiguousarray(np.concatenate(chunks, axis=1))
    if expected_shape is not None and array.shape != expected_shape:
        raise MessageValidationError(
            f"unexpected array shape: got {array.shape}, expected {expected_shape}"
        )
    if require_bits and (array.dtype != np.uint8 or np.any(array > 1)):
        raise MessageValidationError("Boolean payload must contain only uint8 values 0 or 1")
    return array
