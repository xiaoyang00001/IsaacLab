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

from isaaclab.utils import configclass


@configclass
class NetworkCfg:
    """Centralized network configuration for ZeroMQ-based teleoperation components.

    All IP addresses and ports are defined here to avoid hardcoding in individual modules.
    Modify these values to match your network setup.
    """

    # ---- ZeroMQ Game Server (motion controller data publisher) ----
    zmq_game_server_ip: str = "192.168.1.149"
    """IP address of the ZeroMQ game server."""

    zmq_game_server_port: int = 14026
    """Port for the ZeroMQ game server (used by ZeroMqGameClient)."""

    # ---- ZeroMQ Game Sub Device (motion controller data subscriber) ----
    zmq_game_sub_port: int = 14025
    """Port for the ZeroMQ game sub device (used by ZeroMqGameSubDeviceCfg)."""

    # ---- ZeroMQ Object Sync ----
    zmq_object_sync_ip: str = "192.168.1.149"
    """IP address for object synchronization."""

    zmq_object_sync_port: int = 15555
    """Port for object synchronization (used by ZmqObjectSyncActionCfg)."""

    # ---- Player IDs for multi-player teleoperation ----
    zmq_player_id: int = 1
    """Player ID for the ZeroMQ game client (used by OpenXRDeviceCfg)."""

    local_player_id: int = 1
    """Local player ID for the ZeroMQ game sub device."""

    target_remote_player_id: int = 2
    """Target remote player ID for the ZeroMQ game sub device."""

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


# Singleton instance for convenient import
NETWORK_CFG = NetworkCfg()
