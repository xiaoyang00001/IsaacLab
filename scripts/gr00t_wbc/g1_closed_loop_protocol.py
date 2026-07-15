"""Wire protocol shared by the Isaac Lab runner and the GR00T DDS bridge."""

from __future__ import annotations

SCHEMA_VERSION = 1
ROBOT_NAME = "unitree_g1_29dof"
STATE_TOPIC = b"isaac_g1_state"
ACTION_TOPIC = b"g1_wbc_action"

# Hardware/MuJoCo order index -> GR00T policy (IsaacLab) order index.
# Derived by matching joint names against gear_sonic's G1_ISAACLab_ORDER.
MUJOCO_TO_ISAACLAB = (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28)


def pack_topic_message(msgpack_module, topic: bytes, payload: dict) -> bytes:
    return topic + msgpack_module.packb(payload, use_bin_type=True)


def unpack_topic_message(msgpack_module, topic: bytes, raw: bytes) -> dict:
    if not raw.startswith(topic):
        raise ValueError(f"Unexpected topic prefix; expected {topic!r}")
    value = msgpack_module.unpackb(raw[len(topic) :], raw=False)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a msgpack map, got {type(value).__name__}")
    return value
