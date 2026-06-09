# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""RobotYao Unity XR input subscriber.

This is a thin RobotYao-specific wrapper around :class:`ZeroMqGameSubDevice`.
The wire format remains the MGXR motion-controller packet parsed by the base
class, while the defaults match the Unity ``RobotYaoStereoFisheyeApp`` XR
publisher:

* PUB bind endpoint in Unity: ``tcp://*:5555``
* SUB connect endpoint in Isaac Lab: ``tcp://127.0.0.1:5555``
* topic: ``state``
* button bits: X/A=0, Y/B=1, thumbstick click=2, grip=3, trigger button=4
"""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr.zeromq_game_sub_device import ZeroMqGameSubDevice, ZeroMqGameSubDeviceCfg
from isaaclab.devices.retargeter_base import RetargeterBase


class RobotYaoXrSubDevice(ZeroMqGameSubDevice):
    """ZeroMQ subscriber for Unity XR controller input used by RobotYao demos."""

    def __init__(self, cfg: RobotYaoXrSubDeviceCfg, retargeters: list[RetargeterBase] | None = None):
        super().__init__(cfg, retargeters)


@dataclass
class RobotYaoXrSubDeviceCfg(ZeroMqGameSubDeviceCfg):
    """Configuration for RobotYao Unity XR input over ZeroMQ."""

    endpoint: str = "tcp://127.0.0.1:5555"
    topic: str = "state"
    local_player_id: int = 0
    target_remote_player_id: int | None = None

    button_0_mask: int = 1 << 0
    button_1_mask: int = 1 << 1
    thumbstick_button_mask: int = 1 << 2
    squeeze_button_mask: int = 1 << 3
    trigger_button_mask: int = 1 << 4

    class_type: type[DeviceBase] = RobotYaoXrSubDevice
