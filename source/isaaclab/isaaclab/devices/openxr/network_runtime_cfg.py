from __future__ import annotations

import os
import socket
from dataclasses import dataclass


MACHINE_A_IP = os.environ.get("ISAACLAB_MACHINE_A_IP", "192.168.1.142")
MACHINE_B_IP = os.environ.get("ISAACLAB_MACHINE_B_IP", "192.168.1.60")
TRACKING_HUB_IP = os.environ.get("ISAACLAB_TRACKING_HUB_IP", MACHINE_A_IP)


def _detect_local_machine_ip() -> str:
    """Best-effort detection of which configured machine IP belongs to this host."""

    configured_ips = {str(MACHINE_A_IP).strip(), str(MACHINE_B_IP).strip()}
    local_candidates = set()

    # Collect IPs from the hostname.
    try:
        _, _, host_ips = socket.gethostbyname_ex(socket.gethostname())
        local_candidates.update(ip.strip() for ip in host_ips if ip)
    except OSError:
        pass

    # Also collect the outbound interface IP that would be used on the local
    # dual-machine LAN. Using the peer addresses is more reliable than probing
    # a public internet address because these machines may run on an isolated
    # network without external routing.
    for probe_ip in (str(MACHINE_A_IP).strip(), str(MACHINE_B_IP).strip()):
        if not probe_ip:
            continue
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.connect((probe_ip, 1))
                local_candidates.add(probe.getsockname()[0].strip())
            finally:
                probe.close()
        except OSError:
            pass

    for candidate in local_candidates:
        if candidate in configured_ips:
            return candidate

    raise ValueError(
        "Failed to auto-detect ISAACLAB local machine IP. "
        f"Observed candidates={sorted(local_candidates)}; expected one of "
        f"{sorted(configured_ips)}. Set ISAACLAB_LOCAL_MACHINE_IP explicitly if needed."
    )


DEPLOY_LOCAL_MACHINE_IP = os.environ.get("ISAACLAB_LOCAL_MACHINE_IP", _detect_local_machine_ip())


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
            object_sync_role="publisher",
        )
    if local_ip == MACHINE_B_IP:
        return DualMachineRuntimeCfg(
            local_machine_ip=MACHINE_B_IP,
            peer_machine_ip=MACHINE_A_IP,
            tracking_hub_ip=TRACKING_HUB_IP,
            local_player_id=2,
            remote_player_id=1,
            object_sync_role="subscriber",
        )

    raise ValueError(
        f"Unsupported ISAACLAB_LOCAL_MACHINE_IP='{local_ip}'. "
        f"Expected '{MACHINE_A_IP}' or '{MACHINE_B_IP}'."
    )
