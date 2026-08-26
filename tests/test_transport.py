from dataclasses import replace

import numpy as np
import pytest

from rain.errors import MessageValidationError
from rain.serialization import (
    array_to_messages,
    deserialize_message,
    messages_to_array,
    serialize_message,
)
from rain.transport import LogicalTransport
from rain.types import Endpoint, MessageKind, ProtocolPhase


def _messages(array, *, chunk_size=3):
    return array_to_messages(
        array,
        kind=MessageKind.ONLINE_Z1,
        phase=ProtocolPhase.ONLINE,
        sender=Endpoint.S1,
        receiver=Endpoint.S0,
        round_id=4,
        chunk_size=chunk_size,
    )


def test_serialization_round_trip_is_deterministic():
    array = np.arange(18, dtype=np.uint8).reshape(3, 6) % 2
    message = _messages(array)[0]
    wire = serialize_message(message)
    assert serialize_message(deserialize_message(wire)) == wire


def test_transport_counts_exact_stable_nonzero_wire_bytes():
    array = np.ones((4, 9), dtype=np.uint8)
    expected_wire_bytes = sum(
        len(serialize_message(message))
        for message in array_to_messages(
            array,
            kind=MessageKind.ONLINE_Z1,
            phase=ProtocolPhase.ONLINE,
            sender=Endpoint.S1,
            receiver=Endpoint.S0,
            round_id=8,
            chunk_size=4,
        )
    )
    snapshots = []
    for _ in range(2):
        transport = LogicalTransport()
        transport.send_array(
            array,
            kind=MessageKind.ONLINE_Z1,
            phase=ProtocolPhase.ONLINE,
            sender=Endpoint.S1,
            receiver=Endpoint.S0,
            round_id=8,
            chunk_size=4,
        )
        snapshots.append(transport.metrics)
    assert snapshots[0] == snapshots[1]
    assert snapshots[0].server_to_server_bytes == expected_wire_bytes
    assert snapshots[0].server_to_server_bytes > array.nbytes
    assert snapshots[0].online_bytes == snapshots[0].server_to_server_bytes
    assert snapshots[0].message_count == 3


@pytest.mark.parametrize("mutation", ["magic", "truncated", "length"])
def test_malformed_wire_message_is_rejected(mutation):
    wire = bytearray(serialize_message(_messages(np.ones((2, 2), dtype=np.uint8))[0]))
    if mutation == "magic":
        wire[0] ^= 0xFF
    elif mutation == "truncated":
        wire = wire[:-1]
    else:
        wire.extend(b"extra")
    with pytest.raises(MessageValidationError):
        deserialize_message(bytes(wire))


def test_missing_and_duplicate_chunks_are_rejected():
    messages = _messages(np.ones((2, 7), dtype=np.uint8), chunk_size=3)
    kwargs = dict(
        expected_kind=MessageKind.ONLINE_Z1,
        expected_phase=ProtocolPhase.ONLINE,
        expected_sender=Endpoint.S1,
        expected_receiver=Endpoint.S0,
        expected_round_id=4,
        expected_shape=(2, 7),
        require_bits=True,
    )
    with pytest.raises(MessageValidationError, match="incomplete"):
        messages_to_array(messages[:-1], **kwargs)
    with pytest.raises(MessageValidationError, match="duplicate"):
        messages_to_array([messages[0], messages[0], messages[2]], **kwargs)


def test_non_bit_payload_is_rejected_at_receiver():
    messages = _messages(np.array([[0, 2]], dtype=np.uint8))
    with pytest.raises(MessageValidationError, match="only uint8"):
        messages_to_array(
            messages,
            expected_kind=MessageKind.ONLINE_Z1,
            expected_phase=ProtocolPhase.ONLINE,
            expected_sender=Endpoint.S1,
            expected_receiver=Endpoint.S0,
            expected_round_id=4,
            require_bits=True,
        )


def test_wrong_round_is_rejected():
    messages = _messages(np.ones((2, 2), dtype=np.uint8))
    messages[0] = replace(messages[0], round_id=5)
    with pytest.raises(MessageValidationError, match="round_id"):
        messages_to_array(
            messages,
            expected_kind=MessageKind.ONLINE_Z1,
            expected_phase=ProtocolPhase.ONLINE,
            expected_sender=Endpoint.S1,
            expected_receiver=Endpoint.S0,
            expected_round_id=4,
        )


def test_wrong_batch_sender_and_shape_are_rejected():
    messages = array_to_messages(
        np.ones((2, 3), dtype=np.uint8), kind=MessageKind.ONLINE_Z1,
        phase=ProtocolPhase.ONLINE, sender=Endpoint.S1, receiver=Endpoint.S0,
        round_id=4, batch_id=7, chunk_size=3,
    )
    common = dict(expected_kind=MessageKind.ONLINE_Z1,
                  expected_phase=ProtocolPhase.ONLINE,
                  expected_receiver=Endpoint.S0, expected_round_id=4,
                  expected_batch_id=7)
    with pytest.raises(MessageValidationError, match="batch_id"):
        messages_to_array(messages, **{**common, "expected_batch_id": 8,
                          "expected_sender": Endpoint.S1})
    with pytest.raises(MessageValidationError, match="sender"):
        messages_to_array(messages, **{**common, "expected_sender": Endpoint.S0})
    with pytest.raises(MessageValidationError, match="shape"):
        messages_to_array(messages, **{**common, "expected_sender": Endpoint.S1,
                          "expected_shape": (3, 2)})


@pytest.mark.parametrize("dtype", [np.uint8, np.int64, np.uint64])
def test_supported_dtypes_have_deterministic_roundtrip(dtype):
    array = np.arange(12, dtype=dtype).reshape(3, 4)
    messages = array_to_messages(
        array, kind=MessageKind.ARITHMETIC_OPEN, phase=ProtocolPhase.ONLINE,
        sender=Endpoint.S0, receiver=Endpoint.S1, round_id=2, chunk_size=2,
    )
    assert np.array_equal(messages_to_array(
        messages, expected_kind=MessageKind.ARITHMETIC_OPEN,
        expected_phase=ProtocolPhase.ONLINE, expected_sender=Endpoint.S0,
        expected_receiver=Endpoint.S1, expected_round_id=2,
    ), array)


def test_unknown_protocol_version_is_rejected():
    message = replace(_messages(np.ones((2, 2), dtype=np.uint8))[0], version=99)
    with pytest.raises(MessageValidationError, match="version"):
        serialize_message(message)
