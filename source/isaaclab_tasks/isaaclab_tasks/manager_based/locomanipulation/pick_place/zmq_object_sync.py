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

def _to_bind_endpoint(endpoint: str) -> str:
    """Rewrite a tcp endpoint host to ``*`` for binding.

    The configured endpoint carries the IP that remote subscribers connect to.
    Binding must not use that IP: if DHCP reassigns the host address, binding
    to the stale IP fails with EADDRNOTAVAIL. The publisher only needs the port.
    """
    if endpoint.startswith("tcp://"):
        host_port = endpoint[len("tcp://"):]
        host, sep, port = host_port.rpartition(":")
        if sep and host not in ("*", "0.0.0.0"):
            return f"tcp://*:{port}"
    return endpoint


class ZmqPubSocketManager:
    """A simple global manager to share the PUB socket so we don't bind multiple times to the same port."""
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
            bind_endpoint = _to_bind_endpoint(endpoint)
            sock.bind(bind_endpoint)
            cls._sockets[endpoint] = sock
            logger.info(f"[ZMQ Object Sync] Publisher bound to {bind_endpoint} (configured endpoint {endpoint})")
        return cls._sockets[endpoint]

@configclass
class ZmqObjectSyncActionCfg(ActionTermCfg):
    """Configuration for the ZMQ object synchronization action term."""

    class_type: type = None  # set in __post_init__

    role: str = "none"
    """The role of this synchronization: 'publisher', 'subscriber', or 'none'."""

    endpoint: str = "tcp://192.168.10.46:15555"
    """The ZMQ endpoint to bind/connect to."""

    def __post_init__(self):
        self.class_type = ZmqObjectSyncAction


class ZmqObjectSyncAction(ActionTerm):
    """An action term that syncs an asset's state over ZeroMQ.
    
    If role is 'publisher', it reads the asset's pose and publishes it.
    If role is 'subscriber', it subscribes to the pose and updates the asset.
    """

    cfg: ZmqObjectSyncActionCfg

    def __init__(self, cfg: ZmqObjectSyncActionCfg, env: "ManagerBasedEnv"):
        # initialize the base class
        super().__init__(cfg, env)
        
        if zmq is None:
            logger.error("pyzmq is not installed. ZmqObjectSyncAction cannot work. Please `pip install pyzmq`.")

        self.role = self.cfg.role.lower()
        self.endpoint = self.cfg.endpoint
        self.topic = self.cfg.asset_name.encode('utf-8')

        # we don't take any action inputs from the RL policy
        self._action_dim = 0
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        self._context = None
        self._socket = None

        if self.role == "publisher":
            self._socket = ZmqPubSocketManager.get_pub_socket(self.endpoint)
        elif self.role == "subscriber":
            if zmq is not None:
                self._context = zmq.Context()
                self._socket = self._context.socket(zmq.SUB)
                self._socket.connect(self.endpoint)
                self._socket.setsockopt(zmq.SUBSCRIBE, self.topic)
                self._socket.setsockopt(zmq.RCVTIMEO, 1) # Non-blocking roughly
                logger.info(f"[ZMQ Object Sync] Subscriber connected to {self.endpoint} with topic {self.topic.decode('utf-8')}")

    def __del__(self):
        """Cleanup ZMQ resources."""
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
            # Assuming env 0 is the one we want to track
            root_pos_w = self._asset.data.root_pos_w[0].tolist()
            root_quat_w = self._asset.data.root_quat_w[0].tolist()
            
            data = {
                "pos": root_pos_w,
                "quat": root_quat_w
            }
            try:
                msg = json.dumps(data).encode('utf-8')
                self._socket.send_multipart([self.topic, msg], flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                pass

        elif self.role == "subscriber":
            last_msg = self._drain_latest()
            if last_msg is not None:
                try:
                    data = json.loads(last_msg.decode('utf-8'))
                    pos = torch.tensor(data["pos"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
                    quat = torch.tensor(data["quat"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)

                    root_pose = torch.cat([pos, quat], dim=-1)
                    self._asset.write_root_pose_to_sim(root_pose)
                except Exception as e:
                    logger.warning(f"[ZMQ Object Sync] Error applying received pose: {e}")

    def _drain_latest(self):
        """Drain the SUB socket and return only the newest message for our topic."""
        last_msg = None
        while True:
            try:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(parts) == 2 and parts[0] == self.topic:
                    last_msg = parts[1]
            except zmq.Again:
                break
        return last_msg


@configclass
class ZmqRobotSyncActionCfg(ZmqObjectSyncActionCfg):
    """Configuration for the ZMQ articulation (robot) synchronization action term."""

    def __post_init__(self):
        self.class_type = ZmqRobotSyncAction


class ZmqRobotSyncAction(ZmqObjectSyncAction):
    """跨机同步一台 articulation：根位姿 + 关节角。

    publisher 端（SONIC 物理行走机）每 env 步（50Hz）发布 root pose 与
    joint_pos；subscriber 端写根位姿+关节状态并把 PD 目标也指向同步姿态，
    实现纯运动学跟随。订阅端本 term 须声明在 sonic_wholebody 等驱动 term
    之后：同一（子）步内后写覆盖前写（root pose 为即时写、joint target 为
    末次生效的缓冲写），收不到 deploy 包而锁根站立的对端机器人才能被同步
    流接管。
    订阅端建议保持默认固定根模式（SONIC_G1_PHYSICS_MODE 不设），避免本地
    物理与同步流互相拉扯。
    """

    cfg: ZmqRobotSyncActionCfg

    def __init__(self, cfg: ZmqRobotSyncActionCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        # 关节顺序按发布端 joint_names 映射到本地顺序（同款 USD 时为恒等映射），
        # 首包建立、名单变化才重建。名单缺本地关节或维度不符时退化为只同步根
        # 位姿（_joint_sync_ok=False，一次性警告，名单变化后重试）。
        self._joint_index_map: torch.Tensor | None = None
        self._joint_names_key: tuple | None = None
        self._joint_sync_ok = True
        self._first_packet_logged = False
        # 订阅端缓存最近一次同步状态并每步重放：对端 sonic_wholebody 每个物理
        # 子步都把根位姿写回出生锚点，只有同样逐子步重放才能全程压住它；且若
        # 只在收到新包的步才覆盖，收发 50Hz 相位差也会让无包帧闪回出生点。
        self._last_root_pose: torch.Tensor | None = None
        self._last_joint_pos: torch.Tensor | None = None
        # 发布节流：apply_actions 在 decimation 循环内按物理步频（200Hz）被调，
        # 发布只需 env 步频（50Hz）。process_actions 每 env 步恰好一次，借它置位。
        self._publish_pending = False

    def process_actions(self, actions: torch.Tensor):
        self._publish_pending = True

    def apply_actions(self):
        if self._socket is None:
            return

        if self.role == "publisher":
            if not self._publish_pending:
                return
            self._publish_pending = False
            data = {
                "pos": self._asset.data.root_pos_w[0].tolist(),
                "quat": self._asset.data.root_quat_w[0].tolist(),
                "joint_names": list(self._asset.joint_names),
                "joint_pos": self._asset.data.joint_pos[0].tolist(),
            }
            try:
                msg = json.dumps(data).encode('utf-8')
                self._socket.send_multipart([self.topic, msg], flags=zmq.NOBLOCK)
            except zmq.ZMQError:
                pass

        elif self.role == "subscriber":
            last_msg = self._drain_latest()
            if last_msg is not None:
                try:
                    data = json.loads(last_msg.decode('utf-8'))
                    pos = torch.tensor(data["pos"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
                    quat = torch.tensor(data["quat"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)

                    joint_pos = torch.tensor(data["joint_pos"], device=self.device, dtype=torch.float32)
                    names = tuple(data.get("joint_names") or ())
                    if names != self._joint_names_key:
                        self._joint_names_key = names
                        self._joint_index_map = None
                        self._joint_sync_ok = True
                        if names:
                            try:
                                self._joint_index_map = torch.tensor(
                                    [names.index(n) for n in self._asset.joint_names],
                                    device=self.device,
                                    dtype=torch.long,
                                )
                            except ValueError:
                                self._joint_sync_ok = False
                                missing = [n for n in self._asset.joint_names if n not in names]
                                logger.warning(
                                    f"[ZMQ Robot Sync] '{self.cfg.asset_name}' 发布端关节名缺少本地关节 "
                                    f"{missing}；退化为只同步根位姿（名单变化后重试）"
                                )
                    if self._joint_sync_ok:
                        mapped = joint_pos[self._joint_index_map] if self._joint_index_map is not None else joint_pos
                        if mapped.shape[-1] == len(self._asset.joint_names):
                            self._last_joint_pos = mapped.unsqueeze(0).repeat(self.num_envs, 1)
                        else:
                            self._joint_sync_ok = False
                            self._last_joint_pos = None
                            logger.warning(
                                f"[ZMQ Robot Sync] '{self.cfg.asset_name}' 关节维度不匹配 "
                                f"({mapped.shape[-1]} vs {len(self._asset.joint_names)})；退化为只同步根位姿"
                            )
                    self._last_root_pose = torch.cat([pos, quat], dim=-1)
                    if not self._first_packet_logged:
                        joints = self._last_joint_pos.shape[-1] if self._last_joint_pos is not None else 0
                        logger.info(
                            f"[ZMQ Robot Sync] first packet applied for '{self.cfg.asset_name}' ({joints} joints)"
                        )
                        self._first_packet_logged = True
                except Exception as e:
                    logger.warning(f"[ZMQ Robot Sync] Error parsing received robot state: {e}")

            if self._last_root_pose is None:
                return
            # 每个物理子步重放（非冗余）：sonic_wholebody 逐子步写回锚点，本 term
            # 的覆盖也必须逐子步进行
            try:
                self._asset.write_root_pose_to_sim(self._last_root_pose)
                self._asset.write_root_velocity_to_sim(
                    torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
                )
                if self._last_joint_pos is not None:
                    joint_vel = torch.zeros_like(self._last_joint_pos)
                    self._asset.write_joint_state_to_sim(self._last_joint_pos, joint_vel)
                    # PD 目标同样指向同步姿态：物理子步间不被先前 term（如对端
                    # sonic_wholebody 的站立目标）拉走
                    self._asset.set_joint_position_target(self._last_joint_pos)
            except Exception as e:
                logger.warning(f"[ZMQ Robot Sync] Error applying robot state: {e}")
