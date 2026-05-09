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
            sock.bind(endpoint)
            cls._sockets[endpoint] = sock
            logger.info(f"[ZMQ Object Sync] Publisher bound to shared endpoint {endpoint}")
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
            last_msg = None
            while True:
                try:
                    parts = self._socket.recv_multipart(flags=zmq.NOBLOCK)
                    if len(parts) == 2 and parts[0] == self.topic:
                        last_msg = parts[1]
                except zmq.Again:
                    break
            
            if last_msg is not None:
                try:
                    data = json.loads(last_msg.decode('utf-8'))
                    pos = torch.tensor(data["pos"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
                    quat = torch.tensor(data["quat"], device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1)
                    
                    root_pose = torch.cat([pos, quat], dim=-1)
                    self._asset.write_root_pose_to_sim(root_pose)
                except Exception as e:
                    logger.warning(f"[ZMQ Object Sync] Error applying received pose: {e}")
