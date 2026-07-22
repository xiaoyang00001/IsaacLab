import hashlib
import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

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


def _spawn_is_kinematic(rigid_object) -> bool:
    """Whether this scene object was spawned as a kinematic body.

    跟随端（subscriber）的同步物体在 spawn cfg 里被翻成 kinematic，此时不能再写速度。
    发布端同名物体是普通动态刚体，仍然要写完整 root state。
    """

    rigid_props = getattr(getattr(rigid_object.cfg, "spawn", None), "rigid_props", None)
    return bool(getattr(rigid_props, "kinematic_enabled", False))


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
    """Share one PUB socket across the scene-state and reset topics."""

    _context = None
    _sockets = {}

    @classmethod
    def get_pub_socket(cls, endpoint: str, send_hwm: int):
        if zmq is None:
            return None
        if endpoint not in cls._sockets:
            if cls._context is None:
                cls._context = zmq.Context()
            sock = cls._context.socket(zmq.PUB)
            sock.setsockopt(zmq.SNDHWM, max(1, int(send_hwm)))
            sock.setsockopt(zmq.LINGER, 0)
            bind_endpoint = _to_bind_endpoint(endpoint)
            sock.bind(bind_endpoint)
            cls._sockets[endpoint] = sock
            logger.info(
                "[ZMQ Shared PUB] Publisher bound to %s (configured endpoint %s, SNDHWM=%d, LINGER=0)",
                bind_endpoint,
                endpoint,
                max(1, int(send_hwm)),
            )
        return cls._sockets[endpoint]


@configclass
class ZmqObjectSyncActionCfg(ActionTermCfg):
    """Configuration for a ZMQ rigid-object synchronization action term."""

    class_type: type = None  # set in __post_init__

    role: str = "none"
    """Synchronization role: ``publisher``, ``subscriber``, or ``none``."""

    endpoint: str = ""
    """Role-specific endpoint: publisher binds, subscriber connects."""

    send_hwm: int = 3
    """Publisher high-water mark. Old state packets may be dropped when full."""

    receive_hwm: int = 3
    """Subscriber high-water mark before the latest-state drain loop runs."""

    stale_timeout_s: float = 0.5
    """Seconds without a subscriber update before reporting a stale stream."""

    stale_log_interval_s: float = 2.0
    """Minimum interval between stale-stream warnings."""

    def __post_init__(self):
        self.class_type = ZmqObjectSyncAction


class ZmqObjectSyncAction(ActionTerm):
    """Publish or follow a rigid object's complete root state over ZeroMQ."""

    cfg: ZmqObjectSyncActionCfg

    def __init__(self, cfg: ZmqObjectSyncActionCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        self._action_dim = 0
        self._raw_actions = torch.zeros((self.num_envs, self._action_dim), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._export_IO_descriptor = False

        self.role = str(self.cfg.role).strip().lower()
        if self.role not in {"publisher", "subscriber", "none"}:
            logger.error("[ZMQ Object Sync] Unsupported role %r; disabling %s", self.cfg.role, cfg.asset_name)
            self.role = "none"

        self.endpoint = str(self.cfg.endpoint).strip()
        self.topic = self.cfg.asset_name.encode("utf-8")
        self._context = None
        self._socket = None

        self._publisher_session = uuid.uuid4().hex
        self._sequence = 0
        self._last_session: str | None = None
        self._last_sequence = -1
        self._subscriber_start_time = time.monotonic()
        self._last_receive_time: float | None = None
        self._last_stale_warning_time = 0.0
        self._stale_reported = False
        self._received_first_packet = False

        if self.role == "none":
            return
        if zmq is None:
            logger.error("[ZMQ Object Sync] pyzmq is not installed; disabling %s", cfg.asset_name)
            self.role = "none"
            return
        if not self.endpoint:
            logger.error("[ZMQ Object Sync] Empty endpoint; disabling %s", cfg.asset_name)
            self.role = "none"
            return

        try:
            if self.role == "publisher":
                self._socket = ZmqPubSocketManager.get_pub_socket(self.endpoint, self.cfg.send_hwm)
            else:
                self._context = zmq.Context()
                self._socket = self._context.socket(zmq.SUB)
                self._socket.setsockopt(zmq.RCVHWM, max(1, int(self.cfg.receive_hwm)))
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.setsockopt(zmq.SUBSCRIBE, self.topic)
                self._socket.connect(self.endpoint)
                logger.info(
                    "[ZMQ Object Sync] Subscriber connected to %s topic=%s (RCVHWM=%d, LINGER=0)",
                    self.endpoint,
                    self.topic.decode("utf-8"),
                    max(1, int(self.cfg.receive_hwm)),
                )
        except Exception as exc:
            logger.error(
                "[ZMQ Object Sync] Failed to initialize asset=%s role=%s endpoint=%s: %s",
                cfg.asset_name,
                self.role,
                self.endpoint,
                exc,
            )
            self._close_subscriber_resources()
            self._socket = None
            self.role = "none"

    def __del__(self):
        """Release subscriber-owned ZMQ resources."""

        self._close_subscriber_resources()
        try:
            super().__del__()
        except Exception:
            pass

    def _close_subscriber_resources(self) -> None:
        if getattr(self, "role", "none") != "subscriber":
            return
        socket = getattr(self, "_socket", None)
        if socket is not None:
            try:
                socket.close(0)
            except Exception:
                pass
        context = getattr(self, "_context", None)
        if context is not None:
            try:
                context.term()
            except Exception:
                pass

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
        self._raw_actions = actions
        self._processed_actions = actions

    def apply_actions(self):
        if self._socket is None:
            return
        if self.role == "publisher":
            self._publish_latest_state()
        elif self.role == "subscriber":
            received = self._receive_latest_state()
            now = time.monotonic()
            if received:
                if self._stale_reported:
                    logger.info(
                        "[ZMQ Object Sync] Stream recovered asset=%s endpoint=%s",
                        self.cfg.asset_name,
                        self.endpoint,
                    )
                self._last_receive_time = now
                self._stale_reported = False
            else:
                self._warn_if_stale(now)

    def _publish_latest_state(self) -> None:
        root_pos_w = self._asset.data.root_pos_w[0].tolist()
        root_quat_w = self._asset.data.root_quat_w[0].tolist()
        root_lin_vel_w = self._asset.data.root_lin_vel_w[0].tolist()
        root_ang_vel_w = self._asset.data.root_ang_vel_w[0].tolist()

        payload = {
            "version": 1,
            "session": self._publisher_session,
            "seq": self._sequence,
            "timestamp_s": time.time(),
            "pos": root_pos_w,
            "quat": root_quat_w,
            "lin_vel": root_lin_vel_w,
            "ang_vel": root_ang_vel_w,
        }
        self._sequence += 1

        try:
            message = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._socket.send_multipart([self.topic, message], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        except zmq.ZMQError as exc:
            logger.warning(
                "[ZMQ Object Sync] Publish failed asset=%s endpoint=%s: %s",
                self.cfg.asset_name,
                self.endpoint,
                exc,
            )

    def _receive_latest_state(self) -> bool:
        last_message = None
        while True:
            try:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except zmq.ZMQError as exc:
                logger.warning(
                    "[ZMQ Object Sync] Receive failed asset=%s endpoint=%s: %s",
                    self.cfg.asset_name,
                    self.endpoint,
                    exc,
                )
                break
            if len(parts) == 2 and parts[0] == self.topic:
                last_message = parts[1]

        if last_message is None:
            return False

        try:
            payload: dict[str, Any] = json.loads(last_message.decode("utf-8"))
            session = str(payload.get("session", "legacy"))
            sequence = int(payload.get("seq", self._last_sequence + 1))
            if session == self._last_session and sequence <= self._last_sequence:
                return False

            pos = self._payload_tensor(payload, "pos", 3)
            quat = self._payload_tensor(payload, "quat", 4)
            lin_vel = self._payload_tensor(payload, "lin_vel", 3, default=[0.0, 0.0, 0.0])
            ang_vel = self._payload_tensor(payload, "ang_vel", 3, default=[0.0, 0.0, 0.0])
            root_state = torch.cat([pos, quat, lin_vel, ang_vel], dim=-1)
            if not torch.isfinite(root_state).all():
                raise ValueError("received root state contains non-finite values")

            self._asset.write_root_state_to_sim(root_state.repeat(self.num_envs, 1))
            self._last_session = session
            self._last_sequence = sequence
            if not self._received_first_packet:
                logger.info(
                    "[ZMQ Object Sync] Received first packet asset=%s endpoint=%s seq=%d",
                    self.cfg.asset_name,
                    self.endpoint,
                    sequence,
                )
                self._received_first_packet = True
            return True
        except Exception as exc:
            logger.warning(
                "[ZMQ Object Sync] Error applying state asset=%s endpoint=%s: %s",
                self.cfg.asset_name,
                self.endpoint,
                exc,
            )
            return False

    def _payload_tensor(
        self,
        payload: dict[str, Any],
        key: str,
        size: int,
        default: list[float] | None = None,
    ) -> torch.Tensor:
        value = payload.get(key, default)
        if value is None:
            raise KeyError(f"missing payload field {key!r}")
        tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32).reshape(1, -1)
        if tensor.shape[1] != size:
            raise ValueError(f"payload field {key!r} expected {size} values, received {tensor.shape[1]}")
        return tensor

    def _warn_if_stale(self, now: float) -> None:
        timeout = max(0.0, float(self.cfg.stale_timeout_s))
        if timeout <= 0.0:
            return
        baseline = self._last_receive_time if self._last_receive_time is not None else self._subscriber_start_time
        age = now - baseline
        if age < timeout:
            return
        interval = max(0.1, float(self.cfg.stale_log_interval_s))
        if now - self._last_stale_warning_time < interval:
            return
        self._last_stale_warning_time = now
        self._stale_reported = True
        logger.warning(
            "[ZMQ Object Sync] Stream stale asset=%s endpoint=%s last_packet_age=%.3fs",
            self.cfg.asset_name,
            self.endpoint,
            age,
        )


@configclass
class ZmqSceneStateSyncActionCfg(ActionTermCfg):
    """Configuration for one fixed dual-G1/three-box scene-state frame."""

    class_type: type = None  # set in __post_init__

    role: str = "none"
    """Synchronization role: ``publisher``, ``subscriber``, or ``none``."""

    endpoint: str = ""
    """Shared endpoint: PC1 publisher binds and PC2 subscriber connects."""

    topic: str = "scene_state"
    """PUB/SUB topic for the unified scene frame."""

    robot_names: tuple[str, str] = ("robot_1", "robot_2")
    """Fixed articulation names included in every frame."""

    object_names: tuple[str, ...] = ("small_box_1", "small_box_2", "long_box")
    """Fixed rigid-object names included in every frame."""

    send_hwm: int = 3
    """Publisher high-water mark."""

    receive_hwm: int = 3
    """Subscriber high-water mark before the latest-frame drain loop."""

    stale_timeout_s: float = 0.5
    """Seconds without a frame before PC2 reports the stream as stale."""

    stale_log_interval_s: float = 2.0
    """Minimum interval between stale warnings."""

    def __post_init__(self):
        self.class_type = ZmqSceneStateSyncAction


class ZmqSceneStateSyncAction(ActionTerm):
    """Publish or apply one atomic frame containing two robots and three boxes."""

    cfg: ZmqSceneStateSyncActionCfg

    def __init__(self, cfg: ZmqSceneStateSyncActionCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._action_dim = 0
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._export_IO_descriptor = False

        self.role = str(cfg.role).strip().lower()
        if self.role not in {"publisher", "subscriber", "none"}:
            logger.error("[ZMQ Scene Sync] Unsupported role %r; disabling scene synchronization", cfg.role)
            self.role = "none"

        self.endpoint = str(cfg.endpoint).strip()
        self.topic = str(cfg.topic).encode("utf-8")
        self._context = None
        self._socket = None
        self._robots = {name: self._env.scene[name] for name in cfg.robot_names}
        self._objects = {name: self._env.scene[name] for name in cfg.object_names}
        self._object_is_kinematic = {
            name: _spawn_is_kinematic(rigid_object) for name, rigid_object in self._objects.items()
        }

        self._publisher_session = uuid.uuid4().hex
        self._publisher_frame_id = 0
        self._publisher_reset_id = f"{self._publisher_session}:initial"
        self._last_session: str | None = None
        self._last_frame_id = -1
        self._active_reset_id: str | None = None
        self._expected_reset_id: str | None = None
        self._received_first_frame = False
        self._subscriber_start_time = time.monotonic()
        self._last_receive_time: float | None = None
        self._last_stale_warning_time = 0.0
        self._stale_reported = False

        joint_orders = [tuple(robot.joint_names) for robot in self._robots.values()]
        if not joint_orders or any(order != joint_orders[0] for order in joint_orders[1:]):
            raise RuntimeError("robot_1 and robot_2 must use the same fixed joint order for scene synchronization")
        self._joint_count = len(joint_orders[0])
        self._joint_order_hash = hashlib.sha256("\0".join(joint_orders[0]).encode("utf-8")).hexdigest()[:16]

        if self.num_envs != 1:
            logger.error("[ZMQ Scene Sync] Only num_envs=1 is supported; disabling scene synchronization")
            self.role = "none"
            return
        if self.role == "none":
            return
        if zmq is None:
            logger.error("[ZMQ Scene Sync] pyzmq is not installed; disabling scene synchronization")
            self.role = "none"
            return
        if not self.endpoint:
            logger.error("[ZMQ Scene Sync] Empty endpoint; disabling scene synchronization")
            self.role = "none"
            return

        try:
            if self.role == "publisher":
                self._socket = ZmqPubSocketManager.get_pub_socket(self.endpoint, cfg.send_hwm)
            else:
                self._context = zmq.Context()
                self._socket = self._context.socket(zmq.SUB)
                self._socket.setsockopt(zmq.RCVHWM, max(1, int(cfg.receive_hwm)))
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.setsockopt(zmq.SUBSCRIBE, self.topic)
                self._socket.connect(self.endpoint)
                logger.info(
                    "[ZMQ Scene Sync] Subscriber connected to %s topic=%s joint_count=%d joint_hash=%s",
                    self.endpoint,
                    self.topic.decode("utf-8"),
                    self._joint_count,
                    self._joint_order_hash,
                )
        except Exception as exc:
            logger.error(
                "[ZMQ Scene Sync] Failed to initialize role=%s endpoint=%s: %s",
                self.role,
                self.endpoint,
                exc,
            )
            self._close_subscriber_resources()
            self._socket = None
            self.role = "none"

    def __del__(self):
        self._close_subscriber_resources()
        try:
            super().__del__()
        except Exception:
            pass

    def _close_subscriber_resources(self) -> None:
        if getattr(self, "role", "none") != "subscriber":
            return
        socket = getattr(self, "_socket", None)
        if socket is not None:
            try:
                socket.close(0)
            except Exception:
                pass
        context = getattr(self, "_context", None)
        if context is not None:
            try:
                context.term()
            except Exception:
                pass

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def current_reset_id(self) -> str | None:
        """Current publisher reset ID or the last accepted subscriber reset ID."""

        return self._publisher_reset_id if self.role == "publisher" else self._active_reset_id

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions = actions
        self._processed_actions = actions

    def reset(self, env_ids=None) -> None:
        """Preserve network sequence and reset gating across ``env.reset()``."""

        self._raw_actions.zero_()
        self._processed_actions.zero_()

    def set_publisher_reset_id(self, reset_id: str | None) -> None:
        """Start publishing post-reset frames under ``reset_id`` on PC1."""

        if self.role == "publisher" and reset_id:
            self._publisher_reset_id = str(reset_id)
            logger.info("[ZMQ Scene Sync] Publisher entered reset_id=%s", self._publisher_reset_id)

    def expect_reset_id(self, reset_id: str | None) -> None:
        """Reject pre-reset frames on PC2 until a frame with ``reset_id`` arrives."""

        if self.role == "subscriber" and reset_id:
            self._expected_reset_id = str(reset_id)
            logger.info("[ZMQ Scene Sync] Subscriber waiting for reset_id=%s", self._expected_reset_id)

    def apply_actions(self):
        if self._socket is None:
            return
        if self.role == "publisher":
            self._publish_scene_frame()
        elif self.role == "subscriber":
            received = self._receive_latest_scene_frame()
            now = time.monotonic()
            if received:
                if self._stale_reported:
                    logger.info("[ZMQ Scene Sync] Stream recovered endpoint=%s", self.endpoint)
                self._last_receive_time = now
                self._stale_reported = False
            else:
                self._warn_if_stale(now)

    def _publish_scene_frame(self) -> None:
        payload = {
            "schema": "dual_g1_scene_state.v1",
            "session": self._publisher_session,
            "frame_id": self._publisher_frame_id,
            "timestamp_s": time.time(),
            "reset_id": self._publisher_reset_id,
            "joint_count": self._joint_count,
            "joint_order_hash": self._joint_order_hash,
            "robots": {
                name: {
                    "root_state": robot.data.root_state_w[0].tolist(),
                    "joint_pos": robot.data.joint_pos[0].tolist(),
                    "joint_vel": robot.data.joint_vel[0].tolist(),
                }
                for name, robot in self._robots.items()
            },
            "objects": {
                name: {"root_state": rigid_object.data.root_state_w[0].tolist()}
                for name, rigid_object in self._objects.items()
            },
        }
        self._publisher_frame_id += 1

        try:
            message = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._socket.send_multipart([self.topic, message], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass
        except zmq.ZMQError as exc:
            logger.warning("[ZMQ Scene Sync] Publish failed endpoint=%s: %s", self.endpoint, exc)

    def _receive_latest_scene_frame(self) -> bool:
        latest_message = None
        while True:
            try:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except zmq.ZMQError as exc:
                logger.warning("[ZMQ Scene Sync] Receive failed endpoint=%s: %s", self.endpoint, exc)
                return False
            if len(parts) == 2 and parts[0] == self.topic:
                latest_message = parts[1]

        if latest_message is None:
            return False

        try:
            payload: dict[str, Any] = json.loads(latest_message.decode("utf-8"))
            if payload.get("schema") != "dual_g1_scene_state.v1":
                raise ValueError(f"unsupported schema {payload.get('schema')!r}")

            session = str(payload["session"])
            frame_id = int(payload["frame_id"])
            reset_id = str(payload["reset_id"])
            if self._last_session is not None and session != self._last_session:
                logger.info(
                    "[ZMQ Scene Sync] Publisher session changed %s -> %s; accepting the new scene stream",
                    self._last_session,
                    session,
                )
                self._last_frame_id = -1
                self._expected_reset_id = None
            if session == self._last_session and frame_id <= self._last_frame_id:
                return False
            if int(payload["joint_count"]) != self._joint_count:
                raise ValueError(
                    f"joint_count mismatch: remote={payload['joint_count']} local={self._joint_count}"
                )
            if str(payload["joint_order_hash"]) != self._joint_order_hash:
                raise ValueError(
                    "joint order mismatch: "
                    f"remote={payload['joint_order_hash']} local={self._joint_order_hash}"
                )
            if self._expected_reset_id is not None and reset_id != self._expected_reset_id:
                return False

            robot_states = self._parse_robot_states(payload["robots"])
            object_states = self._parse_object_states(payload["objects"])
            self._apply_scene_states(robot_states, object_states)

            self._last_session = session
            self._last_frame_id = frame_id
            self._active_reset_id = reset_id
            if self._expected_reset_id == reset_id:
                self._expected_reset_id = None
                logger.info("[ZMQ Scene Sync] Subscriber accepted post-reset frame reset_id=%s", reset_id)
            if not self._received_first_frame:
                logger.info(
                    "[ZMQ Scene Sync] Received first frame endpoint=%s frame_id=%d reset_id=%s",
                    self.endpoint,
                    frame_id,
                    reset_id,
                )
                self._received_first_frame = True
            return True
        except Exception as exc:
            logger.warning("[ZMQ Scene Sync] Ignored invalid scene frame: %s", exc)
            return False

    def _parse_robot_states(
        self, payload: dict[str, Any]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        parsed = {}
        for name in self.cfg.robot_names:
            state = payload[name]
            root_state = self._payload_tensor(state, "root_state", 13)
            joint_pos = self._payload_tensor(state, "joint_pos", self._joint_count)
            joint_vel = self._payload_tensor(state, "joint_vel", self._joint_count)
            parsed[name] = (root_state, joint_pos, joint_vel)
        return parsed

    def _parse_object_states(self, payload: dict[str, Any]) -> dict[str, torch.Tensor]:
        return {
            name: self._payload_tensor(payload[name], "root_state", 13)
            for name in self.cfg.object_names
        }

    def _payload_tensor(self, payload: dict[str, Any], key: str, size: int) -> torch.Tensor:
        tensor = torch.as_tensor(payload[key], device=self.device, dtype=torch.float32).reshape(1, -1)
        if tensor.shape[1] != size:
            raise ValueError(f"field {key!r} expected {size} values, received {tensor.shape[1]}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"field {key!r} contains non-finite values")
        return tensor

    def _apply_scene_states(
        self,
        robot_states: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        object_states: dict[str, torch.Tensor],
    ) -> None:
        for name, (root_state, joint_pos, joint_vel) in robot_states.items():
            robot = self._robots[name]
            robot.write_root_state_to_sim(root_state)
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.set_joint_position_target(joint_pos)
            robot.set_joint_velocity_target(joint_vel)

        for name, root_state in object_states.items():
            if self._object_is_kinematic[name]:
                # 跟随端的同步物体 spawn 时就被翻成 kinematic，而 write_root_state_to_sim
                # 会把速度一并写下去。**CPU pipeline**（--device cpu）下这个写入会落到逐
                # body 的 PhysX C++ 调用上，而 PhysX 拒绝给 kinematic 刚体设速度：
                #     PxRigidDynamic::setLinearVelocity: Body must be non-kinematic!
                # 每帧每个物体刷一对 linear/angular，几十秒就撞满 PhysX 的 1000 条错误上限，
                # 于是 "PhysX has reported too many errors, simulation has been stopped"
                # —— 跟随端整个仿真被掐停，箱子和机器人一起僵住，看着就像同步断了。
                # GPU pipeline（--device cuda:0）走的是 tensor API 批量写入，不经过这条
                # 逐 body 路径，所以同样的代码在 GPU 上没有症状，只有 CPU 上会炸。
                # kinematic 体的运动完全由位姿决定，速度写入本来就没有意义，直接跳过。
                self._objects[name].write_root_pose_to_sim(root_state[:, :7])
            else:
                self._objects[name].write_root_state_to_sim(root_state)

    def _warn_if_stale(self, now: float) -> None:
        timeout = max(0.0, float(self.cfg.stale_timeout_s))
        if timeout <= 0.0:
            return
        baseline = self._last_receive_time if self._last_receive_time is not None else self._subscriber_start_time
        age = now - baseline
        if age < timeout:
            return
        interval = max(0.1, float(self.cfg.stale_log_interval_s))
        if now - self._last_stale_warning_time < interval:
            return
        self._last_stale_warning_time = now
        self._stale_reported = True
        logger.warning(
            "[ZMQ Scene Sync] Stream stale endpoint=%s last_frame_age=%.3fs; holding the last mirrored scene",
            self.endpoint,
            age,
        )


@configclass
class ZmqEnvResetSyncActionCfg(ActionTermCfg):
    """Configuration for broadcasting a full-environment reset event."""

    class_type: type = None  # set in __post_init__

    role: str = "none"
    """Synchronization role: ``publisher``, ``subscriber``, or ``none``."""

    endpoint: str = ""
    """Shared scene-sync endpoint: publisher binds, subscriber connects."""

    topic: str = "env_reset"
    """PUB/SUB topic used for reset events."""

    repeat_frames: int = 10
    """Number of action-application frames over which a reset event is repeated."""

    send_hwm: int = 3
    """Publisher high-water mark used by the shared PUB socket."""

    receive_hwm: int = 3
    """Subscriber high-water mark."""

    def __post_init__(self):
        self.class_type = ZmqEnvResetSyncAction


class ZmqEnvResetSyncAction(ActionTerm):
    """Send PC1 reset events and expose de-duplicated reset requests on PC2."""

    cfg: ZmqEnvResetSyncActionCfg

    def __init__(self, cfg: ZmqEnvResetSyncActionCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._action_dim = 0
        self._raw_actions = torch.zeros((self.num_envs, 0), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._export_IO_descriptor = False

        self.role = str(cfg.role).strip().lower()
        if self.role not in {"publisher", "subscriber", "none"}:
            logger.error("[ZMQ Env Reset] Unsupported role %r; disabling reset sync", cfg.role)
            self.role = "none"
        self.endpoint = str(cfg.endpoint).strip()
        self.topic = str(cfg.topic).encode("utf-8")
        self._context = None
        self._socket = None

        self._publisher_session = uuid.uuid4().hex
        self._next_reset_sequence = 0
        self._pending_reset_id: str | None = None
        self._pending_repeat_frames = 0
        self._last_received_reset_id: str | None = None
        self._remote_reset_id: str | None = None

        if self.role == "none":
            return
        if zmq is None:
            logger.error("[ZMQ Env Reset] pyzmq is not installed; disabling reset synchronization")
            self.role = "none"
            return
        if not self.endpoint:
            logger.error("[ZMQ Env Reset] Empty endpoint; disabling reset synchronization")
            self.role = "none"
            return

        try:
            if self.role == "publisher":
                self._socket = ZmqPubSocketManager.get_pub_socket(self.endpoint, cfg.send_hwm)
            else:
                self._context = zmq.Context()
                self._socket = self._context.socket(zmq.SUB)
                self._socket.setsockopt(zmq.RCVHWM, max(1, int(cfg.receive_hwm)))
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.setsockopt(zmq.SUBSCRIBE, self.topic)
                self._socket.connect(self.endpoint)
                logger.info(
                    "[ZMQ Env Reset] Subscriber connected to %s topic=%s",
                    self.endpoint,
                    self.topic.decode("utf-8"),
                )
        except Exception as exc:
            logger.error(
                "[ZMQ Env Reset] Failed to initialize role=%s endpoint=%s: %s",
                self.role,
                self.endpoint,
                exc,
            )
            self._close_subscriber_resources()
            self._socket = None
            self.role = "none"

    def __del__(self):
        self._close_subscriber_resources()
        try:
            super().__del__()
        except Exception:
            pass

    def _close_subscriber_resources(self) -> None:
        if getattr(self, "role", "none") != "subscriber":
            return
        socket = getattr(self, "_socket", None)
        if socket is not None:
            try:
                socket.close(0)
            except Exception:
                pass
        context = getattr(self, "_context", None)
        if context is not None:
            try:
                context.term()
            except Exception:
                pass

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
        self._raw_actions = actions
        self._processed_actions = actions

    def reset(self, env_ids=None) -> None:
        """Preserve event IDs and repeat state across ``env.reset()``.

        The subscriber must retain the last processed ID so repeated packets do
        not cause a reset loop. The publisher must retain its queued repeats so
        PC2 can still receive the event after PC1 has reset locally.
        """

        self._raw_actions.zero_()
        self._processed_actions.zero_()

    def request_local_reset(self) -> str | None:
        """Queue a reset event on the authoritative PC1 publisher."""

        if self.role != "publisher":
            return None
        reset_id = f"{self._publisher_session}:{self._next_reset_sequence}"
        self._next_reset_sequence += 1
        self._pending_reset_id = reset_id
        self._pending_repeat_frames = max(1, int(self.cfg.repeat_frames))
        logger.info(
            "[ZMQ Env Reset] Queued reset_id=%s repeats=%d",
            reset_id,
            self._pending_repeat_frames,
        )
        return reset_id

    def consume_remote_reset_request(self) -> str | None:
        """Return and clear the subscriber's pending full-env reset ID."""

        reset_id = self._remote_reset_id
        self._remote_reset_id = None
        return reset_id

    def apply_actions(self):
        if self._socket is None:
            return
        if self.role == "publisher":
            self._publish_pending_reset()
        elif self.role == "subscriber":
            self._receive_reset_events()

    def _publish_pending_reset(self) -> None:
        if self._pending_reset_id is None or self._pending_repeat_frames <= 0:
            return
        payload = {
            "version": 1,
            "reset_id": self._pending_reset_id,
            "timestamp_s": time.time(),
        }
        try:
            message = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self._socket.send_multipart([self.topic, message], flags=zmq.NOBLOCK)
            self._pending_repeat_frames -= 1
            if self._pending_repeat_frames <= 0:
                logger.info("[ZMQ Env Reset] Finished publishing reset_id=%s", self._pending_reset_id)
                self._pending_reset_id = None
        except zmq.Again:
            pass
        except zmq.ZMQError as exc:
            logger.warning("[ZMQ Env Reset] Publish failed endpoint=%s: %s", self.endpoint, exc)

    def _receive_reset_events(self) -> None:
        latest_message = None
        while True:
            try:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except zmq.ZMQError as exc:
                logger.warning("[ZMQ Env Reset] Receive failed endpoint=%s: %s", self.endpoint, exc)
                return
            if len(parts) == 2 and parts[0] == self.topic:
                latest_message = parts[1]

        if latest_message is None:
            return
        try:
            payload: dict[str, Any] = json.loads(latest_message.decode("utf-8"))
            reset_id = str(payload["reset_id"])
        except Exception as exc:
            logger.warning("[ZMQ Env Reset] Invalid reset event: %s", exc)
            return
        if reset_id == self._last_received_reset_id:
            return
        self._last_received_reset_id = reset_id
        self._remote_reset_id = reset_id
        logger.info("[ZMQ Env Reset] Received reset_id=%s", reset_id)


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

    def _drain_latest(self):
        """Drain the SUB socket and return only the newest message for our topic.

        本类继承 conveyor 侧富版 ``ZmqObjectSyncAction``（其订阅逻辑走 ``_receive_latest_state``，
        不提供 ``_drain_latest``），因此机器人同步分支自带一份最新包排空实现，保持自足。
        """
        last_msg = None
        while True:
            try:
                parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
                if len(parts) == 2 and parts[0] == self.topic:
                    last_msg = parts[1]
            except zmq.Again:
                break
        return last_msg
