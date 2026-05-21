from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import torch

try:
    import zmq
except ModuleNotFoundError:
    zmq = None

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

logger = logging.getLogger(__name__)


class ZmqRobotPubSocketManager:
    """Share PUB sockets so multiple terms can publish without rebinding."""

    _context = None
    _sockets: dict[str, object] = {}

    @classmethod
    def get_pub_socket(cls, endpoint: str):
        if zmq is None:
            return None
        if endpoint not in cls._sockets:
            if cls._context is None:
                cls._context = zmq.Context()
            socket = cls._context.socket(zmq.PUB)
            socket.bind(endpoint)
            cls._sockets[endpoint] = socket
            logger.info(f"[ZMQ Robot Sync] Publisher bound to {endpoint}")
        return cls._sockets[endpoint]


@configclass
class ZmqRobotSyncActionCfg(ActionTermCfg):
    class_type: type = None

    role: str = "none"
    endpoint: str = "tcp://127.0.0.1:15565"
    topic: str | None = None
    sync_root_pose: bool = True

    def __post_init__(self):
        self.class_type = ZmqRobotSyncAction


class ZmqRobotSyncAction(ActionTerm):
    """Synchronize articulation root pose + joint state over ZeroMQ."""

    cfg: ZmqRobotSyncActionCfg

    def __init__(self, cfg: ZmqRobotSyncActionCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        if zmq is None:
            logger.error("pyzmq is not installed. ZmqRobotSyncAction cannot work.")

        self.role = str(self.cfg.role).lower()
        self.endpoint = self.cfg.endpoint
        topic = self.cfg.topic or self.cfg.asset_name
        self.topic = topic.encode("utf-8")
        self.sync_root_pose = bool(self.cfg.sync_root_pose)

        self._action_dim = 0
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._context = None
        self._socket = None
        self._joint_count = int(self._asset.data.joint_pos.shape[-1])
        self._zero_joint_vel = torch.zeros((self.num_envs, self._joint_count), device=self.device, dtype=torch.float32)
        self._zero_root_vel = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)

        if self.role == "publisher":
            self._socket = ZmqRobotPubSocketManager.get_pub_socket(self.endpoint)
        elif self.role == "subscriber" and zmq is not None:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.connect(self.endpoint)
            self._socket.setsockopt(zmq.SUBSCRIBE, self.topic)
            self._socket.setsockopt(zmq.RCVTIMEO, 1)
            logger.info(f"[ZMQ Robot Sync] Subscriber connected to {self.endpoint} with topic {topic}")

    def __del__(self):
        if self.role == "subscriber":
            if self._socket is not None:
                try:
                    self._socket.close(0)
                except Exception:
                    pass
            if self._context is not None:
                try:
                    self._context.term()
                except Exception:
                    pass
        super().__del__()

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        if self._socket is None:
            return

        if self.role == "publisher":
            joint_pos = self._asset.data.joint_pos[0].detach().cpu().tolist()
            data = {"joint_pos": joint_pos}
            if self.sync_root_pose:
                data["root_pos"] = self._asset.data.root_pos_w[0].detach().cpu().tolist()
                data["root_quat"] = self._asset.data.root_quat_w[0].detach().cpu().tolist()
            try:
                payload = json.dumps(data).encode("utf-8")
                self._socket.send_multipart([self.topic, payload], flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                pass
            return

        last_msg = None
        while True:
            try:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(parts) == 2 and parts[0] == self.topic:
                    last_msg = parts[1]
            except zmq.Again:
                break

        if last_msg is None:
            return

        try:
            data = json.loads(last_msg.decode("utf-8"))
            joint_pos = torch.tensor(data["joint_pos"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(
                self.num_envs, 1
            )
            self._asset.write_joint_state_to_sim(joint_pos, self._zero_joint_vel)
            self._asset.set_joint_position_target(joint_pos)

            if self.sync_root_pose and "root_pos" in data and "root_quat" in data:
                root_pos = torch.tensor(data["root_pos"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(
                    self.num_envs, 1
                )
                root_quat = torch.tensor(data["root_quat"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(
                    self.num_envs, 1
                )
                root_state = torch.cat([root_pos, root_quat, self._zero_root_vel], dim=-1)
                self._asset.write_root_state_to_sim(root_state)
        except Exception as err:
            logger.warning(f"[ZMQ Robot Sync] Error applying synchronized robot state: {err}")
