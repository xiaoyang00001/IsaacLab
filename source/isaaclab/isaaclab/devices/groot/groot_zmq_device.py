# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GROOT ZMQ device for feeding whole-body G1 joint targets into Isaac Lab."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from isaaclab.devices.device_base import DeviceBase, DeviceCfg

logger = logging.getLogger(__name__)


MUJOCO_TO_ISAACLAB: tuple[int, ...] = (
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
)
"""Map a 29-DOF G1 vector from MuJoCo order to IsaacLab order."""


GROOT_DEFAULT_ANGLES_MUJOCO: tuple[float, ...] = (
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
)
"""GROOT default G1 joint angles in MuJoCo order."""


class GrootZmqDevice(DeviceBase):
    """Subscribe to GROOT ZMQ debug output and emit 29-DOF G1 joint targets.

    The command returned by :meth:`advance` is a 29-element tensor in IsaacLab joint order.
    The environment action term is responsible for converting this absolute target into a
    bounded per-step joint-position target.
    """

    def __init__(self, cfg: GrootZmqDeviceCfg):
        """Initialize the GROOT ZMQ device.

        Args:
            cfg: Configuration for the ZMQ endpoint and source fields.
        """
        super().__init__(retargeters=None)
        self.cfg = cfg
        self._sim_device = torch.device(cfg.sim_device)
        self._additional_callbacks: dict[Any, Callable] = {}

        self._mujoco_to_isaaclab = torch.tensor(MUJOCO_TO_ISAACLAB, dtype=torch.long, device=self._sim_device)
        default_mujoco = torch.tensor(GROOT_DEFAULT_ANGLES_MUJOCO, dtype=torch.float32, device=self._sim_device)
        self._default_target_isaaclab = self._to_isaaclab_order(default_mujoco)
        self._target_isaaclab = self._default_target_isaaclab.clone()
        self._last_receive_time: float | None = None
        self._last_source_field: str | None = None
        self._warned_missing_fields = False
        self._warned_bad_shape = False
        self._anchor_sync = None
        self._xr_pre_sync_update_subscription = None
        self._xr_anchor_headset_path: str | None = None

        try:
            import msgpack
            import zmq

            try:
                import msgpack_numpy as msgpack_numpy

                msgpack_numpy.patch()
            except ModuleNotFoundError:
                pass
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "GrootZmqDevice requires the Python packages 'pyzmq' and 'msgpack'. "
                "Install them in the IsaacLab Python environment before launching GROOT teleoperation."
            ) from exc

        self._msgpack = msgpack
        self._zmq = zmq
        self._topic_bytes = cfg.topic.encode("utf-8")
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, cfg.topic)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.connect(f"tcp://{cfg.host}:{cfg.port}")

        logger.info(
            "GROOT ZMQ device connected to tcp://%s:%s topic '%s' using field '%s'",
            cfg.host,
            cfg.port,
            cfg.topic,
            cfg.source_field,
        )

        if cfg.xr_cfg is not None:
            self._setup_xr_anchor(cfg.xr_cfg)

    def __del__(self):
        """Close the ZMQ socket when the device is destroyed."""
        try:
            self.close()
        except Exception:
            pass

    def __str__(self) -> str:
        """Return a short device description."""
        stale_text = "never received"
        if self._last_receive_time is not None:
            age = time.monotonic() - self._last_receive_time
            stale_text = f"last frame {age:.3f}s ago from '{self._last_source_field}'"
        return (
            "GROOT ZMQ Device\n"
            f"\tEndpoint: tcp://{self.cfg.host}:{self.cfg.port}\n"
            f"\tTopic: {self.cfg.topic}\n"
            f"\tSource field: {self.cfg.source_field}\n"
            f"\tFallback fields: {', '.join(self.cfg.fallback_fields)}\n"
            f"\tXR anchor: {self.cfg.xr_cfg.anchor_prim_path if self.cfg.xr_cfg is not None else 'disabled'}\n"
            f"\tState: {stale_text}"
        )

    def reset(self):
        """Reset command state to the GROOT default pose until a new ZMQ frame arrives."""
        self._target_isaaclab = self._default_target_isaaclab.clone()
        self._last_receive_time = None
        self._last_source_field = None

    def add_callback(self, key: Any, func: Callable):
        """Store callbacks for compatibility with teleoperation scripts."""
        self._additional_callbacks[key] = func

    def close(self):
        """Close ZMQ resources."""
        if hasattr(self, "_socket") and self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if hasattr(self, "_context") and self._context is not None:
            self._context.term()
            self._context = None
        self._xr_pre_sync_update_subscription = None
        self._anchor_sync = None

    def advance(self) -> torch.Tensor:
        """Return the latest 29-DOF G1 target in IsaacLab joint order."""
        msg = self._poll_latest_msg()
        if msg is not None:
            target_mujoco, source_field = self._extract_target(msg)
            if target_mujoco is not None:
                target_isaaclab = self._to_isaaclab_order(target_mujoco)
                if self.cfg.smoothing_alpha < 1.0:
                    alpha = max(0.0, self.cfg.smoothing_alpha)
                    target_isaaclab = self._target_isaaclab + alpha * (target_isaaclab - self._target_isaaclab)
                self._target_isaaclab = target_isaaclab
                self._last_receive_time = time.monotonic()
                self._last_source_field = source_field

        return self._target_isaaclab.clone()

    def _poll_latest_msg(self) -> dict[str, Any] | None:
        """Poll the newest ZMQ message without blocking."""
        if self._socket is None:
            return None
        try:
            raw = self._socket.recv(self._zmq.NOBLOCK)
        except self._zmq.Again:
            return None

        payload = raw[len(self._topic_bytes) :] if raw.startswith(self._topic_bytes) else raw
        try:
            msg = self._msgpack.unpackb(payload, raw=False)
        except Exception as exc:
            logger.warning("Failed to decode GROOT ZMQ payload: %s", exc)
            return None

        if not isinstance(msg, dict):
            logger.warning("Ignoring GROOT ZMQ payload of type %s; expected dict.", type(msg).__name__)
            return None
        return msg

    def _extract_target(self, msg: dict[str, Any]) -> tuple[torch.Tensor | None, str | None]:
        """Extract a 29-element MuJoCo-order target from a decoded GROOT frame."""
        candidate_fields = (self.cfg.source_field, *self.cfg.fallback_fields)
        for field_name in candidate_fields:
            value = msg.get(field_name)
            if value is None:
                continue
            target = torch.as_tensor(value, dtype=torch.float32, device=self._sim_device).flatten()
            if target.numel() != 29:
                if not self._warned_bad_shape:
                    logger.warning(
                        "Ignoring GROOT field '%s' with %d elements; expected 29.",
                        field_name,
                        target.numel(),
                    )
                    self._warned_bad_shape = True
                continue
            return target, field_name

        if not self._warned_missing_fields:
            logger.warning(
                "GROOT ZMQ frame does not contain any usable joint target field. Tried: %s",
                ", ".join(candidate_fields),
            )
            self._warned_missing_fields = True
        return None, None

    def _to_isaaclab_order(self, q_mujoco: torch.Tensor) -> torch.Tensor:
        """Convert a 29-element G1 joint vector from MuJoCo order to IsaacLab order."""
        q_isaaclab = torch.empty_like(q_mujoco)
        q_isaaclab[self._mujoco_to_isaaclab] = q_mujoco
        return q_isaaclab

    def _setup_xr_anchor(self, xr_cfg: Any):
        """Create and synchronize an XR anchor without using OpenXR retargeters."""
        try:
            import carb
            from isaacsim.core.prims import SingleXFormPrim

            from isaaclab.devices.openxr.xr_anchor_utils import XrAnchorSynchronizer

            XRCore = None
            XRCoreEventType = None
            try:
                from omni.kit.xr.core import XRCore, XRCoreEventType
            except ModuleNotFoundError:
                pass

            if xr_cfg.anchor_prim_path is not None:
                anchor_path = xr_cfg.anchor_prim_path.rstrip("/")
                self._xr_anchor_headset_path = f"{anchor_path}/XRAnchor"
            else:
                self._xr_anchor_headset_path = "/World/XRAnchor"

            _ = SingleXFormPrim(
                self._xr_anchor_headset_path,
                position=xr_cfg.anchor_pos,
                orientation=xr_cfg.anchor_rot,
            )

            if hasattr(carb, "settings"):
                carb.settings.get_settings().set_float("/persistent/xr/profile/ar/render/nearPlane", xr_cfg.near_plane)
                carb.settings.get_settings().set_string("/persistent/xr/profile/ar/anchorMode", "custom anchor")
                carb.settings.get_settings().set_string(
                    "/xrstage/profile/ar/customAnchor", self._xr_anchor_headset_path
                )

            xr_core = XRCore.get_singleton() if XRCore is not None else None
            if xr_core is not None and xr_cfg.anchor_prim_path is not None:
                self._anchor_sync = XrAnchorSynchronizer(
                    xr_core=xr_core,
                    xr_cfg=xr_cfg,
                    xr_anchor_headset_path=self._xr_anchor_headset_path,
                )
                if XRCoreEventType is not None:
                    self._xr_pre_sync_update_subscription = (
                        xr_core.get_message_bus().create_subscription_to_pop_by_type(
                            XRCoreEventType.pre_sync_update,
                            lambda _: self._anchor_sync.sync_headset_to_anchor(),
                            name="isaaclab_groot_xr_pre_sync_update",
                        )
                    )

            logger.info("GROOT XR anchor configured at %s", self._xr_anchor_headset_path)
        except Exception as exc:
            logger.warning("GROOT XR anchor setup failed: %s", exc)


@dataclass
class GrootZmqDeviceCfg(DeviceCfg):
    """Configuration for :class:`GrootZmqDevice`."""

    host: str = "localhost"
    """ZMQ publisher host."""

    port: int = 5557
    """ZMQ publisher port."""

    topic: str = "g1_debug"
    """ZMQ topic prefix used by GROOT deploy."""

    source_field: str = "last_action"
    """Primary 29-DOF MuJoCo-order field to read from each GROOT frame."""

    fallback_fields: tuple[str, ...] = ("body_q_target", "body_q", "body_q_measured")
    """Fallback 29-DOF fields used when :attr:`source_field` is missing."""

    smoothing_alpha: float = 1.0
    """First-order smoothing alpha for target updates. 1.0 disables smoothing."""

    xr_cfg: Any | None = None
    """Optional XR configuration used only for anchoring the XR view."""

    teleoperation_active_default: bool = True
    """GROOT control should start stepping immediately, even when XR is enabled."""

    retargeters: list = field(default_factory=list)
    """GROOT outputs native joint targets and does not use retargeters."""

    class_type: type[DeviceBase] = GrootZmqDevice
