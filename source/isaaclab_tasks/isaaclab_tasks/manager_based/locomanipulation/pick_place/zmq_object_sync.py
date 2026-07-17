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


class ZmqPubSocketManager:
    """Share one PUB socket across all synchronized objects on an endpoint."""

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
            sock.bind(endpoint)
            cls._sockets[endpoint] = sock
            logger.info(
                "[ZMQ Object Sync] Publisher bound to shared endpoint %s (SNDHWM=%d, LINGER=0)",
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
