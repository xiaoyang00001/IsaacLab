# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Network configuration for distributed teleoperation components.

This module centralizes all network endpoints (IP addresses and ports) used by
ZeroMQ-based communication components, including:
- ZeroMqGameClient (motion controller data publisher)
- ZeroMqGameSubDevice (motion controller data subscriber)
- ZmqObjectSyncAction (object pose synchronization)

Usage:
    from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.network_cfg import NETWORK_CFG

    endpoint = NETWORK_CFG.zmq_game_server_endpoint
"""

import socket

from isaaclab.utils import configclass


def _get_local_ip() -> str:
    """获取本机局域网 IP 地址。

    通过建立 UDP 连接探测本机非回环 IP，如果失败则回退到 hostname 方式。

    Returns:
        本机 IP 地址字符串，如 "192.168.1.100"。
    """
    try:
        # 通过 UDP 连接外部地址来探测本机 IP（不会真正发送数据）
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except OSError:
        return "127.0.0.1"


@configclass
class NetworkCfg:
    """Centralized network configuration for ZeroMQ-based teleoperation components.

    All IP addresses and ports are defined here to avoid hardcoding in individual modules.
    Modify these values to match your network setup.
    """

    # ---- ZeroMQ Game Server (motion controller data publisher) ----
    zmq_game_server_ip: str = "192.168.50.105"
    """IP address of the ZeroMQ game server."""

    zmq_game_server_port: int = 14026
    """Port for the ZeroMQ game server (used by ZeroMqGameClient)."""

    # ---- ZeroMQ Game Sub Device (motion controller data subscriber) ----
    zmq_game_sub_port: int = 14025
    """Port for the ZeroMQ game sub device (used by ZeroMqGameSubDeviceCfg)."""

    # ---- ZeroMQ Object Sync ----
    zmq_object_sync_ip: str = _get_local_ip()
    """IP address for object synchronization (auto-detected from local network)."""

    zmq_object_sync_port: int = 15555
    """Port for object synchronization (used by ZmqObjectSyncActionCfg)."""

    # ---- Player IDs for multi-player teleoperation ----
    zmq_player_id: int = 1
    """Player ID for the ZeroMQ game client (used by OpenXRDeviceCfg)."""

    local_player_id: int = 1
    """Local player ID for the ZeroMQ game sub device."""

    target_remote_player_id: int = 2
    """Target remote player ID for the ZeroMQ game sub device."""

    # ---- Multi-robot role mapping ----
    primary_player_id: int = 1
    """Player ID that is mapped to scene robot ``robot`` / prim ``Robot``."""

    secondary_player_id: int = 2
    """Player ID that is mapped to scene robot ``remote_robot`` / prim ``RemoteRobot``."""

    primary_robot_scene_name: str = "robot"
    """Scene asset name for the primary robot."""

    secondary_robot_scene_name: str = "remote_robot"
    """Scene asset name for the secondary robot."""

    primary_robot_prim_name: str = "Robot"
    """USD prim name for the primary robot."""

    secondary_robot_prim_name: str = "RemoteRobot"
    """USD prim name for the secondary robot."""

    viewer_follow_local_robot: bool = True
    """Whether the GUI viewport should track the locally controlled robot."""

    viewer_follow_body_name: str | None = None
    """Optional body name for viewer tracking. If unset, track the robot root."""

    xr_anchor_follow_local_robot: bool = True
    """Whether the XR anchor should be attached to the locally controlled robot."""

    xr_anchor_body_name: str = "pelvis"
    """Body name used to anchor XR to the locally controlled robot."""

    object_sync_authority_player_id: int = 1
    """Player ID that owns object physics and publishes object synchronization."""

    # ---- Convenience properties ----
    @property
    def zmq_game_server_endpoint(self) -> str:
        """Full endpoint string for the ZeroMQ game server (e.g. tcp://192.168.1.149:14026)."""
        return f"tcp://{self.zmq_game_server_ip}:{self.zmq_game_server_port}"

    @property
    def zmq_game_sub_endpoint(self) -> str:
        """Full endpoint string for the ZeroMQ game sub device (e.g. tcp://192.168.1.149:14025)."""
        return f"tcp://{self.zmq_game_server_ip}:{self.zmq_game_sub_port}"

    @property
    def zmq_object_sync_endpoint(self) -> str:
        """Full endpoint string for object synchronization (e.g. tcp://192.168.1.149:15555)."""
        return f"tcp://{self.zmq_object_sync_ip}:{self.zmq_object_sync_port}"

    @property
    def local_is_primary_player(self) -> bool:
        """Whether this machine controls the primary robot."""
        if self.local_player_id == self.primary_player_id:
            return True
        if self.local_player_id == self.secondary_player_id:
            return False
        return True

    @property
    def local_robot_scene_name(self) -> str:
        """Scene asset name of the robot controlled by this machine."""
        if self.local_is_primary_player:
            return self.primary_robot_scene_name
        return self.secondary_robot_scene_name

    @property
    def remote_robot_scene_name(self) -> str:
        """Scene asset name of the robot driven by the subscribed remote player."""
        if self.local_is_primary_player:
            return self.secondary_robot_scene_name
        return self.primary_robot_scene_name

    @property
    def local_robot_prim_name(self) -> str:
        """USD prim name of the robot controlled by this machine."""
        if self.local_is_primary_player:
            return self.primary_robot_prim_name
        return self.secondary_robot_prim_name

    @property
    def remote_robot_prim_name(self) -> str:
        """USD prim name of the robot driven by the subscribed remote player."""
        if self.local_is_primary_player:
            return self.secondary_robot_prim_name
        return self.primary_robot_prim_name

    @property
    def zmq_object_sync_role(self) -> str:
        """Object-sync role for this machine."""
        if self.local_player_id == self.object_sync_authority_player_id:
            return "publisher"
        return "subscriber"

    def get_local_robot_prim_path(self, env_index: int = 0) -> str:
        """World prim path for the locally controlled robot."""
        return f"/World/envs/env_{env_index}/{self.local_robot_prim_name}"

    def get_remote_robot_prim_path(self, env_index: int = 0) -> str:
        """World prim path for the remotely controlled robot."""
        return f"/World/envs/env_{env_index}/{self.remote_robot_prim_name}"

    def get_local_robot_body_prim_path(self, env_index: int = 0, body_name: str | None = None) -> str:
        """World prim path for a body on the locally controlled robot."""
        resolved_body_name = body_name or self.xr_anchor_body_name
        return f"{self.get_local_robot_prim_path(env_index)}/{resolved_body_name}"

    def get_remote_robot_body_prim_path(self, env_index: int = 0, body_name: str | None = None) -> str:
        """World prim path for a body on the remotely controlled robot."""
        resolved_body_name = body_name or self.xr_anchor_body_name
        return f"{self.get_remote_robot_prim_path(env_index)}/{resolved_body_name}"


# Singleton instance for convenient import
NETWORK_CFG = NetworkCfg()
