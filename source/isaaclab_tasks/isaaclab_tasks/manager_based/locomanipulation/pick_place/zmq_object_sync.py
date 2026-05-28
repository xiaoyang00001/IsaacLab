import json
import logging
import time
from typing import TYPE_CHECKING

import torch

try:
    import zmq
except ModuleNotFoundError:
    zmq = None

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs.manager_based_env import ManagerBasedEnv

logger = logging.getLogger(__name__)


class ZmqPubSocketManager:
    """Share PUB sockets so multiple sync terms can publish on one endpoint."""

    _context = None
    _sockets = {}

    @classmethod
    def get_pub_socket(cls, endpoint: str):
        if zmq is None:
            return None
        if endpoint not in cls._sockets:
            if cls._context is None:
                cls._context = zmq.Context()
            sock = cls._context.socket(zmq.PUB)
            sock.bind(endpoint)
            cls._sockets[endpoint] = sock
            logger.info(f"[ZMQ Object Sync] Publisher bound to shared endpoint {endpoint}")
        return cls._sockets[endpoint]


@configclass
class ZmqObjectSyncActionCfg(ActionTermCfg):
    """Configuration for the ZMQ object synchronization action term."""

    class_type: type = None

    role: str = "none"
    endpoint: str = "tcp://192.168.1.142:15555"
    local_endpoint: str | None = None
    remote_endpoint: str | None = None
    local_machine_ip: str = "127.0.0.1"
    remote_machine_ip: str = "127.0.0.1"
    default_owner_machine_ip: str | None = None
    robot_asset_name: str = "robot"
    grasp_distance_enter_m: float = 0.16
    grasp_distance_exit_m: float = 0.22
    hand_closed_enter: float = 0.16
    hand_closed_exit: float = 0.10
    remote_stale_timeout_s: float = 0.50
    apply_remote_updates: bool = True
    mirror_x_center: float | None = None

    def __post_init__(self):
        self.class_type = ZmqObjectSyncAction


class ZmqObjectSyncAction(ActionTerm):
    """Synchronize an object's pose, with runtime ownership handoff while grasping."""

    cfg: ZmqObjectSyncActionCfg

    def __init__(self, cfg: ZmqObjectSyncActionCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        if zmq is None:
            logger.error("pyzmq is not installed. ZmqObjectSyncAction cannot work.")

        self.role = str(self.cfg.role).lower()
        self.endpoint = self.cfg.endpoint
        self.topic = self.cfg.asset_name.encode("utf-8")
        self._mirror_x_center = self.cfg.mirror_x_center
        self._peer_mode = bool(self.cfg.local_endpoint and self.cfg.remote_endpoint)
        self._local_machine_ip = str(self.cfg.local_machine_ip).strip()
        self._remote_machine_ip = str(self.cfg.remote_machine_ip).strip()
        self._default_owner_machine_ip = str(
            self.cfg.default_owner_machine_ip or self.cfg.local_machine_ip
        ).strip()
        self._apply_remote_updates = bool(self.cfg.apply_remote_updates)
        self._last_owner_machine_ip = self._default_owner_machine_ip
        self._remote_state: dict[str, object] | None = None
        self._last_remote_rx_time = 0.0
        self._publish_sequence = 0
        self._side_grasp_latched = {"left": False, "right": False}

        self._action_dim = 0
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._context = None
        self._socket = None
        self._pub_socket = None
        self._sub_socket = None
        self._robot_asset = None
        self._left_palm_body_id = None
        self._right_palm_body_id = None
        self._left_hand_joint_ids: list[int] = []
        self._right_hand_joint_ids: list[int] = []

        self._setup_local_grasp_probes(env)

        if self._peer_mode:
            if zmq is not None:
                self._sub_socket = self._create_sub_socket(self.cfg.remote_endpoint)
                self._pub_socket = ZmqPubSocketManager.get_pub_socket(self.cfg.local_endpoint)
            return

        if self.role == "publisher":
            self._socket = ZmqPubSocketManager.get_pub_socket(self.endpoint)
        elif self.role == "subscriber":
            self._socket = self._create_sub_socket(self.endpoint)

    def __del__(self):
        if self.role == "subscriber" or self._peer_mode:
            if self._socket is not None:
                try:
                    self._socket.close(0)
                except Exception:
                    pass
            if self._sub_socket is not None:
                try:
                    self._sub_socket.close(0)
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
        if self._peer_mode:
            self._apply_peer_actions()
            return

        if self._socket is None:
            return

        if self.role == "publisher":
            root_pos_w = self._asset.data.root_pos_w[0:1]
            root_quat_w = self._asset.data.root_quat_w[0:1]
            root_pos_w, root_quat_w = self._mirror_pose_if_needed(root_pos_w, root_quat_w)
            data = {"pos": root_pos_w[0].tolist(), "quat": root_quat_w[0].tolist()}
            try:
                msg = json.dumps(data).encode("utf-8")
                self._socket.send_multipart([self.topic, msg], flags=zmq.NOBLOCK)
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
            pos = torch.tensor(data["pos"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
            quat = torch.tensor(data["quat"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
            pos, quat = self._mirror_pose_if_needed(pos, quat)
            root_pose = torch.cat([pos, quat], dim=-1)
            self._asset.write_root_pose_to_sim(root_pose)
        except Exception as err:
            logger.warning(f"[ZMQ Object Sync] Error applying received pose: {err}")

    def _apply_peer_actions(self):
        local_is_grasping, local_grasp_score = self._compute_local_grasp_state()
        self._publish_local_state(local_is_grasping, local_grasp_score)
        remote_state = self._receive_latest_remote_state()

        if not self._apply_remote_updates or remote_state is None:
            return

        owner_machine_ip = self._resolve_owner_machine_ip(local_is_grasping, local_grasp_score, remote_state)
        self._last_owner_machine_ip = owner_machine_ip
        if owner_machine_ip == self._local_machine_ip:
            return

        try:
            pos = torch.tensor(remote_state["pos"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
            quat = torch.tensor(remote_state["quat"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
            pos, quat = self._mirror_pose_if_needed(pos, quat)
            root_pose = torch.cat([pos, quat], dim=-1)
            self._asset.write_root_pose_to_sim(root_pose)
        except Exception as err:
            logger.warning(f"[ZMQ Object Sync] Error applying peer-owned pose: {err}")

    def _publish_local_state(self, local_is_grasping: bool, local_grasp_score: float):
        if self._pub_socket is None:
            return
        root_pos_w = self._asset.data.root_pos_w[0:1]
        root_quat_w = self._asset.data.root_quat_w[0:1]
        root_pos_w, root_quat_w = self._mirror_pose_if_needed(root_pos_w, root_quat_w)
        self._publish_sequence += 1
        payload = {
            "machine_ip": self._local_machine_ip,
            "pos": root_pos_w[0].tolist(),
            "quat": root_quat_w[0].tolist(),
            "is_grasping": bool(local_is_grasping),
            "grasp_score": float(local_grasp_score),
            "sequence": int(self._publish_sequence),
            "sent_time": float(time.monotonic()),
        }
        try:
            msg = json.dumps(payload).encode("utf-8")
            self._pub_socket.send_multipart([self.topic, msg], flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            pass

    def _receive_latest_remote_state(self) -> dict[str, object] | None:
        if self._sub_socket is None:
            return self._remote_state

        last_msg = None
        while True:
            try:
                parts = self._sub_socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(parts) == 2 and parts[0] == self.topic:
                    last_msg = parts[1]
            except zmq.Again:
                break

        if last_msg is None:
            return self._remote_state

        try:
            remote_state = json.loads(last_msg.decode("utf-8"))
        except Exception as err:
            logger.warning(f"[ZMQ Object Sync] Error decoding peer state: {err}")
            return self._remote_state

        if str(remote_state.get("machine_ip", "")).strip() != self._remote_machine_ip:
            return self._remote_state

        self._remote_state = remote_state
        self._last_remote_rx_time = time.monotonic()
        return self._remote_state

    def _resolve_owner_machine_ip(
        self, local_is_grasping: bool, local_grasp_score: float, remote_state: dict[str, object] | None
    ) -> str:
        remote_recent = (
            remote_state is not None
            and (time.monotonic() - self._last_remote_rx_time) <= float(self.cfg.remote_stale_timeout_s)
        )
        if not remote_recent:
            return self._local_machine_ip

        remote_is_grasping = bool(remote_state.get("is_grasping", False))
        remote_grasp_score = float(remote_state.get("grasp_score", 0.0))

        if local_is_grasping and not remote_is_grasping:
            return self._local_machine_ip
        if remote_is_grasping and not local_is_grasping:
            return self._remote_machine_ip
        if local_is_grasping and remote_is_grasping:
            if local_grasp_score > remote_grasp_score + 0.05:
                return self._local_machine_ip
            if remote_grasp_score > local_grasp_score + 0.05:
                return self._remote_machine_ip
            if self._last_owner_machine_ip in (self._local_machine_ip, self._remote_machine_ip):
                return self._last_owner_machine_ip
            return self._default_owner_machine_ip
        return self._default_owner_machine_ip

    def _setup_local_grasp_probes(self, env: "ManagerBasedEnv"):
        try:
            self._robot_asset = env.scene[self.cfg.robot_asset_name]
            left_body_ids, _ = self._robot_asset.find_bodies("left_hand_palm_link")
            right_body_ids, _ = self._robot_asset.find_bodies("right_hand_palm_link")
            left_joint_ids, _ = self._robot_asset.find_joints(["^left_hand_.*_joint$"])
            right_joint_ids, _ = self._robot_asset.find_joints(["^right_hand_.*_joint$"])
            self._left_palm_body_id = int(left_body_ids[0]) if len(left_body_ids) > 0 else None
            self._right_palm_body_id = int(right_body_ids[0]) if len(right_body_ids) > 0 else None
            self._left_hand_joint_ids = [int(joint_id) for joint_id in left_joint_ids]
            self._right_hand_joint_ids = [int(joint_id) for joint_id in right_joint_ids]
        except Exception as err:
            logger.warning(f"[ZMQ Object Sync] Unable to initialize grasp probes for {self.cfg.asset_name}: {err}")
            self._robot_asset = None

    def _compute_local_grasp_state(self) -> tuple[bool, float]:
        if self._robot_asset is None:
            return False, 0.0

        object_pos = self._asset.data.root_pos_w[0]
        left_grasping, left_score = self._compute_side_grasp_state(
            "left", self._left_palm_body_id, self._left_hand_joint_ids, object_pos
        )
        right_grasping, right_score = self._compute_side_grasp_state(
            "right", self._right_palm_body_id, self._right_hand_joint_ids, object_pos
        )
        return bool(left_grasping or right_grasping), float(max(left_score, right_score))

    def _compute_side_grasp_state(
        self, side_name: str, body_id: int | None, joint_ids: list[int], object_pos: torch.Tensor
    ) -> tuple[bool, float]:
        if body_id is None or not joint_ids:
            self._side_grasp_latched[side_name] = False
            return False, 0.0

        palm_pos = self._robot_asset.data.body_pos_w[0, body_id]
        palm_distance = float(torch.linalg.vector_norm(object_pos - palm_pos).item())
        hand_joint_pos = self._robot_asset.data.joint_pos[0, joint_ids]
        hand_closed_score = float(torch.mean(torch.abs(hand_joint_pos)).item())

        was_grasping = bool(self._side_grasp_latched.get(side_name, False))
        if was_grasping:
            is_grasping = (
                palm_distance <= float(self.cfg.grasp_distance_exit_m)
                and hand_closed_score >= float(self.cfg.hand_closed_exit)
            )
        else:
            is_grasping = (
                palm_distance <= float(self.cfg.grasp_distance_enter_m)
                and hand_closed_score >= float(self.cfg.hand_closed_enter)
            )

        self._side_grasp_latched[side_name] = bool(is_grasping)
        proximity_score = max(0.0, float(self.cfg.grasp_distance_exit_m) - palm_distance)
        grasp_score = hand_closed_score + 2.0 * proximity_score
        return bool(is_grasping), float(grasp_score)

    def _create_sub_socket(self, endpoint: str | None):
        if zmq is None or not endpoint:
            return None
        if self._context is None:
            self._context = zmq.Context()
        socket = self._context.socket(zmq.SUB)
        socket.connect(endpoint)
        socket.setsockopt(zmq.SUBSCRIBE, self.topic)
        socket.setsockopt(zmq.RCVTIMEO, 1)
        logger.info(f"[ZMQ Object Sync] Subscriber connected to {endpoint} with topic {self.topic.decode('utf-8')}")
        return socket

    def _mirror_pose_if_needed(self, pos: torch.Tensor, quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._mirror_x_center is None:
            return pos, quat

        mirrored_pos = pos.clone()
        mirrored_pos[:, 0] = 2.0 * float(self._mirror_x_center) - mirrored_pos[:, 0]

        rot_mat = math_utils.matrix_from_quat(quat)
        mirror_mat = torch.eye(3, device=quat.device, dtype=quat.dtype)
        mirror_mat[0, 0] = -1.0
        mirrored_rot_mat = mirror_mat.unsqueeze(0) @ rot_mat @ mirror_mat.unsqueeze(0)
        mirrored_quat = math_utils.quat_from_matrix(mirrored_rot_mat)
        return mirrored_pos, mirrored_quat
