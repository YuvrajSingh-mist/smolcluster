"""Control protocol encode/decode roundtrip."""

from grove.control import MsgType, encode_ctrl, decode_ctrl, CTRL_MAGIC


def test_roundtrip():
    payload = {"rank": 3, "step": 42, "loss": 1.234}
    raw = encode_ctrl(MsgType.HEARTBEAT, payload)
    assert raw[:4] == CTRL_MAGIC
    msg_type, decoded = decode_ctrl(raw)
    assert msg_type == MsgType.HEARTBEAT
    assert decoded == payload


def test_empty_payload():
    raw = encode_ctrl(MsgType.SCRIPT_REQUEST, {})
    msg_type, decoded = decode_ctrl(raw)
    assert msg_type == MsgType.SCRIPT_REQUEST
    assert decoded == {}


def test_all_msg_types():
    for mt in MsgType:
        raw = encode_ctrl(mt, {"type": mt.name})
        msg_type, decoded = decode_ctrl(raw)
        assert msg_type == mt
        assert decoded["type"] == mt.name
