from __future__ import annotations

import os
from dataclasses import dataclass


MACHINE_A_IP = os.environ.get("ISAACLAB_MACHINE_A_IP", "192.168.50.68")
MACHINE_B_IP = os.environ.get("ISAACLAB_MACHINE_B_IP", "192.168.50.105")
TRACKING_HUB_IP = os.environ.get("ISAACLAB_TRACKING_HUB_IP", MACHINE_A_IP)
DEPLOY_LOCAL_MACHINE_IP = os.environ.get("ISAACLAB_LOCAL_MACHINE_IP", MACHINE_A_IP)


@dataclass(frozen=True)
class DualMachineRuntimeCfg:
    """Small helper that derives all network roles from the local machine IP."""

    local_machine_ip: str
    peer_machine_ip: str
    tracking_hub_ip: str
    local_player_id: int
    remote_player_id: int
    object_sync_role: str
    tracking_state_port: int = 14025
    tracking_send_port: int = 14026
    object_sync_port: int = 15555
    robot_sync_port: int = 15565

    @property
    def tracking_send_endpoint(self) -> str:
        return f"tcp://{self.tracking_hub_ip}:{self.tracking_send_port}"

    @property
    def tracking_subscribe_endpoint(self) -> str:
        return f"tcp://{self.tracking_hub_ip}:{self.tracking_state_port}"

    @property
    def object_sync_endpoint(self) -> str:
        host = self.local_machine_ip if self.object_sync_role == "publisher" else self.peer_machine_ip
        return f"tcp://{host}:{self.object_sync_port}"

    @property
    def local_robot_sync_endpoint(self) -> str:
        return f"tcp://{self.local_machine_ip}:{self.robot_sync_port}"

    @property
    def remote_robot_sync_endpoint(self) -> str:
        return f"tcp://{self.peer_machine_ip}:{self.robot_sync_port}"


def build_dual_machine_runtime_cfg(local_machine_ip: str | None = None) -> DualMachineRuntimeCfg:
    """Build the dual-machine deployment config.

    Override these environment variables or edit this file when moving to a new
    network:
    - ``ISAACLAB_MACHINE_A_IP``
    - ``ISAACLAB_MACHINE_B_IP``
    - ``ISAACLAB_TRACKING_HUB_IP``
    - ``ISAACLAB_LOCAL_MACHINE_IP``
    """

    local_ip = local_machine_ip or DEPLOY_LOCAL_MACHINE_IP
    local_ip = str(local_ip).strip()

    if local_ip == MACHINE_A_IP:
        return DualMachineRuntimeCfg(
            local_machine_ip=MACHINE_A_IP,
            peer_machine_ip=MACHINE_B_IP,
            tracking_hub_ip=TRACKING_HUB_IP,
            local_player_id=1,
            remote_player_id=2,
            object_sync_role="subscriber",
        )
    if local_ip == MACHINE_B_IP:
        return DualMachineRuntimeCfg(
            local_machine_ip=MACHINE_B_IP,
            peer_machine_ip=MACHINE_A_IP,
            tracking_hub_ip=TRACKING_HUB_IP,
            local_player_id=2,
            remote_player_id=1,
            object_sync_role="publisher",
        )

    raise ValueError(
        f"Unsupported ISAACLAB_LOCAL_MACHINE_IP='{local_ip}'. "
        f"Expected '{MACHINE_A_IP}' or '{MACHINE_B_IP}'."
    )
