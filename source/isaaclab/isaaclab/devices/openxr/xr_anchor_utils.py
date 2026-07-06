# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Utilities for synchronizing XR anchor pose with a reference prim and XR config."""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any

import numpy as np

# import logger
logger = logging.getLogger(__name__)

from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage_id

from .xr_cfg import XrAnchorRotationMode

with contextlib.suppress(ModuleNotFoundError):
    import usdrt
    from pxr import Gf as pxrGf
    from usdrt import Rt


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_from_wxyz(w: float, x: float, y: float, z: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _yaw_from_pxr_quat(quat: Any) -> float:
    ix, iy, iz = quat.GetImaginary()
    return _yaw_from_wxyz(quat.GetReal(), ix, iy, iz)


def _axis_xy_angle(axis: tuple[float, float, float]) -> float | None:
    horizontal_norm = math.hypot(axis[0], axis[1])
    if horizontal_norm < 1.0e-9:
        return None
    return math.atan2(axis[1], axis[0])


def _perpendicular_horizontal_axis(axis: tuple[float, float, float]) -> tuple[float, float, float]:
    if math.hypot(axis[0], axis[1]) < 1.0e-9:
        return (1.0, 0.0, 0.0)
    return (-axis[1], axis[0], 0.0)


def _rotate_vector_by_quat(quat: Any, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    w = quat.GetReal()
    x, y, z = quat.GetImaginary()
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-9 or not math.isfinite(norm):
        return vector
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    vx, vy, vz = vector
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _yaw_from_matrix_axis(
    matrix: Any,
    local_axis: tuple[float, float, float],
    fallback_axis: tuple[float, float, float] | None = None,
    min_horizontal_norm: float = 1.0e-9,
) -> float:
    direction = _rotate_vector_by_quat(matrix.ExtractRotationQuat(), local_axis)
    horizontal_norm = math.hypot(direction[0], direction[1])
    if horizontal_norm >= min_horizontal_norm:
        return math.atan2(direction[1], direction[0])

    if fallback_axis is not None:
        fallback_direction = _rotate_vector_by_quat(matrix.ExtractRotationQuat(), fallback_axis)
        fallback_horizontal_norm = math.hypot(fallback_direction[0], fallback_direction[1])
        local_axis_yaw = _axis_xy_angle(local_axis)
        fallback_axis_yaw = _axis_xy_angle(fallback_axis)
        if (
            fallback_horizontal_norm >= 1.0e-9
            and local_axis_yaw is not None
            and fallback_axis_yaw is not None
        ):
            fallback_world_yaw = math.atan2(fallback_direction[1], fallback_direction[0])
            return _wrap_angle(fallback_world_yaw + local_axis_yaw - fallback_axis_yaw)

    return _yaw_from_pxr_quat(matrix.ExtractRotationQuat())


def _make_yaw_quat(yaw: float) -> Any:
    return pxrGf.Quatd(math.cos(yaw * 0.5), pxrGf.Vec3d(0.0, 0.0, math.sin(yaw * 0.5)))


def _normalize_pxr_quat(quat: Any) -> Any:
    w = quat.GetReal()
    x, y, z = quat.GetImaginary()
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1.0e-9 or not math.isfinite(norm):
        return pxrGf.Quatd(1.0, pxrGf.Vec3d(0.0, 0.0, 0.0))
    return pxrGf.Quatd(w / norm, pxrGf.Vec3d(x / norm, y / norm, z / norm))


class XrAnchorSynchronizer:
    """Keeps the XR anchor prim aligned with a reference prim according to XR config."""

    def __init__(self, xr_core: Any, xr_cfg: Any, xr_anchor_headset_path: str):
        self._xr_core = xr_core
        self._xr_cfg = xr_cfg
        self._xr_anchor_headset_path = xr_anchor_headset_path

        self.__anchor_prim_initial_quat = None
        self.__anchor_prim_initial_yaw = None
        self.__anchor_prim_initial_height = None
        self.__smoothed_anchor_quat = None
        self.__last_anchor_quat = None
        self.__anchor_rotation_enabled = True
        self.__initial_yaw_recenter_done = False
        self.__initial_yaw_recenter_in_progress = False

        # Resolve USD layer identifier of the anchor for updates
        try:
            from isaacsim.core.utils.stage import get_current_stage

            stage = get_current_stage()
            xr_anchor_headset_prim = stage.GetPrimAtPath(self._xr_anchor_headset_path)
            prim_stack = xr_anchor_headset_prim.GetPrimStack() if xr_anchor_headset_prim is not None else None
            self.__anchor_headset_layer_identifier = prim_stack[0].layer.identifier if prim_stack else None
        except Exception:
            self.__anchor_headset_layer_identifier = None

    def reset(self):
        self.__anchor_prim_initial_quat = None
        self.__anchor_prim_initial_yaw = None
        self.__anchor_prim_initial_height = None
        self.__smoothed_anchor_quat = None
        self.__last_anchor_quat = None
        self.__anchor_rotation_enabled = True
        self.__initial_yaw_recenter_done = False
        self.__initial_yaw_recenter_in_progress = False
        self.sync_headset_to_anchor()

    def toggle_anchor_rotation(self):
        self.__anchor_rotation_enabled = not self.__anchor_rotation_enabled
        logger.info(f"XR: Toggling anchor rotation: {self.__anchor_rotation_enabled}")

    def recenter_yaw_to_anchor_prim(self) -> bool:
        """Align the current headset yaw with the configured anchor prim yaw."""
        self.__initial_yaw_recenter_in_progress = True
        try:
            recentered = self._recenter_yaw_to_anchor_prim(log_failures=True, reason="manual")
        finally:
            self.__initial_yaw_recenter_in_progress = False
        if recentered:
            self.__initial_yaw_recenter_done = True
        return recentered

    def _recenter_yaw_to_anchor_prim(self, *, log_failures: bool, reason: str) -> bool:
        try:
            rotation_matrix = self._get_anchor_rotation_world_matrix()
            if rotation_matrix is None:
                if log_failures:
                    logger.warning("XR: Cannot recenter yaw; anchor rotation prim world matrix is unavailable")
                return False

            head_device = self._xr_core.get_input_device("/user/head") if self._xr_core is not None else None
            if head_device is None:
                if log_failures:
                    logger.warning("XR: Cannot recenter yaw; head input device is unavailable")
                return False

            try:
                head_matrix = head_device.get_virtual_world_pose("")
            except TypeError:
                head_matrix = head_device.get_virtual_world_pose()
            if head_matrix is None:
                if log_failures:
                    logger.warning("XR: Cannot recenter yaw; head pose is unavailable")
                return False

            headset_forward_axis = getattr(self._xr_cfg, "recenter_headset_forward_axis", (1.0, 0.0, 0.0))
            headset_fallback_axis = getattr(self._xr_cfg, "recenter_headset_fallback_axis", None)
            if headset_fallback_axis is None:
                headset_fallback_axis = _perpendicular_horizontal_axis(headset_forward_axis)
            if self.__anchor_prim_initial_quat is None:
                self._set_anchor_rotation_origin(rotation_matrix)

            anchor_prim_yaw = self._get_anchor_yaw(rotation_matrix)
            headset_yaw = _yaw_from_matrix_axis(
                head_matrix,
                headset_forward_axis,
                fallback_axis=headset_fallback_axis,
                min_horizontal_norm=0.15,
            )
            yaw_correction = _wrap_angle(anchor_prim_yaw - headset_yaw)

            w, x, y, z = self._xr_cfg.anchor_rot
            cfg_quat = pxrGf.Quatd(w, pxrGf.Vec3d(x, y, z))
            prim_delta_yaw_quat = pxrGf.Quatd(1.0, pxrGf.Vec3d(0.0, 0.0, 0.0))
            if self._xr_cfg.anchor_rotation_mode in (
                XrAnchorRotationMode.FOLLOW_PRIM,
                XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
            ):
                initial_yaw = (
                    self.__anchor_prim_initial_yaw
                    if self.__anchor_prim_initial_yaw is not None
                    else anchor_prim_yaw
                )
                prim_delta_yaw_quat = _make_yaw_quat(_wrap_angle(anchor_prim_yaw - initial_yaw))

            current_anchor_quat = self.__last_anchor_quat
            if current_anchor_quat is None:
                current_anchor_quat = prim_delta_yaw_quat * cfg_quat

            desired_anchor_quat = _make_yaw_quat(yaw_correction) * current_anchor_quat
            recentered_cfg_quat = _normalize_pxr_quat(prim_delta_yaw_quat.GetInverse() * desired_anchor_quat)
            cfg_imag = recentered_cfg_quat.GetImaginary()
            self._xr_cfg.anchor_rot = (
                recentered_cfg_quat.GetReal(),
                cfg_imag[0],
                cfg_imag[1],
                cfg_imag[2],
            )

            self.__smoothed_anchor_quat = None
            self.__last_anchor_quat = None
            self.sync_headset_to_anchor()

            logger.info(
                "XR: Recentered yaw to anchor prim "
                f"(reason={reason}, anchor_yaw={anchor_prim_yaw:.3f}, headset_yaw={headset_yaw:.3f}, "
                f"correction={yaw_correction:.3f})"
            )
            return True
        except Exception as e:
            if log_failures:
                logger.warning(f"XR: Recenter yaw failed: {e}")
            return False

    def sync_headset_to_anchor(self):
        """Sync XR anchor pose in USD from reference prim (in Fabric/usdrt)."""
        try:
            if self._xr_cfg.anchor_prim_path is None:
                return

            position_matrix = self._get_anchor_prim_world_matrix()
            if position_matrix is None:
                return
            rotation_matrix = self._get_anchor_rotation_world_matrix()
            if rotation_matrix is None:
                rotation_matrix = position_matrix
            rt_pos = position_matrix.ExtractTranslation()

            if self.__anchor_prim_initial_quat is None:
                self._set_anchor_rotation_origin(rotation_matrix)

            if (
                getattr(self._xr_cfg, "recenter_yaw_on_start", False)
                and not self.__initial_yaw_recenter_done
                and not self.__initial_yaw_recenter_in_progress
            ):
                self.__initial_yaw_recenter_in_progress = True
                try:
                    recentered = self._recenter_yaw_to_anchor_prim(log_failures=False, reason="startup")
                finally:
                    self.__initial_yaw_recenter_in_progress = False
                if recentered:
                    self.__initial_yaw_recenter_done = True
                    return

            if getattr(self._xr_cfg, "fixed_anchor_height", False):
                if self.__anchor_prim_initial_height is None:
                    self.__anchor_prim_initial_height = rt_pos[2]
                rt_pos[2] = self.__anchor_prim_initial_height

            pxr_anchor_pos = pxrGf.Vec3d(*rt_pos) + pxrGf.Vec3d(*self._xr_cfg.anchor_pos)

            w, x, y, z = self._xr_cfg.anchor_rot
            pxr_cfg_quat = pxrGf.Quatd(w, pxrGf.Vec3d(x, y, z))

            pxr_anchor_quat = pxr_cfg_quat

            if self._xr_cfg.anchor_rotation_mode in (
                XrAnchorRotationMode.FOLLOW_PRIM,
                XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
            ):
                # yaw-only about Z (right-handed, Z-up)
                current_yaw = self._get_anchor_yaw(rotation_matrix)
                initial_yaw = self.__anchor_prim_initial_yaw
                if initial_yaw is None:
                    initial_yaw = current_yaw
                    self.__anchor_prim_initial_yaw = current_yaw
                yaw = _wrap_angle(current_yaw - initial_yaw)
                pxr_delta_yaw_only_quat = _make_yaw_quat(yaw)
                pxr_anchor_quat = pxr_delta_yaw_only_quat * pxr_cfg_quat

                if self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED:
                    if self.__smoothed_anchor_quat is None:
                        self.__smoothed_anchor_quat = pxr_anchor_quat
                    else:
                        dt = SimulationContext.instance().get_rendering_dt()
                        alpha = 1.0 - math.exp(-dt / max(self._xr_cfg.anchor_rotation_smoothing_time, 1e-6))
                        alpha = min(1.0, max(0.05, alpha))
                        self.__smoothed_anchor_quat = pxrGf.Slerp(alpha, self.__smoothed_anchor_quat, pxr_anchor_quat)
                        pxr_anchor_quat = self.__smoothed_anchor_quat

            elif self._xr_cfg.anchor_rotation_mode == XrAnchorRotationMode.CUSTOM:
                if self._xr_cfg.anchor_rotation_custom_func is not None:
                    rt_prim_quat = rotation_matrix.ExtractRotationQuat()
                    anchor_prim_pose = np.array(
                        [
                            rt_pos[0],
                            rt_pos[1],
                            rt_pos[2],
                            rt_prim_quat.GetReal(),
                            rt_prim_quat.GetImaginary()[0],
                            rt_prim_quat.GetImaginary()[1],
                            rt_prim_quat.GetImaginary()[2],
                        ],
                        dtype=np.float64,
                    )
                    # Previous headpose must be provided by caller; fall back to zeros.
                    prev_head = getattr(self, "_previous_headpose", np.zeros(7, dtype=np.float64))
                    np_array_quat = self._xr_cfg.anchor_rotation_custom_func(prev_head, anchor_prim_pose)
                    w, x, y, z = np_array_quat
                    pxr_anchor_quat = pxrGf.Quatd(w, pxrGf.Vec3d(x, y, z))

            pxr_mat = pxrGf.Matrix4d()
            pxr_mat.SetTranslateOnly(pxr_anchor_pos)

            if self.__anchor_rotation_enabled:
                pxr_mat.SetRotateOnly(pxr_anchor_quat)
                self.__last_anchor_quat = pxr_anchor_quat
            else:
                if self.__last_anchor_quat is None:
                    self.__last_anchor_quat = pxr_anchor_quat

                pxr_mat.SetRotateOnly(self.__last_anchor_quat)
                self.__smoothed_anchor_quat = self.__last_anchor_quat

            self._xr_core.set_world_transform_matrix(
                self._xr_anchor_headset_path, pxr_mat, self.__anchor_headset_layer_identifier
            )
        except Exception as e:
            logger.warning(f"XR: Anchor sync failed: {e}")

    def _get_anchor_prim_world_matrix(self) -> Any | None:
        return self._get_prim_world_matrix(self._xr_cfg.anchor_prim_path)

    def _get_anchor_rotation_world_matrix(self) -> Any | None:
        anchor_rotation_prim_path = getattr(self._xr_cfg, "anchor_rotation_prim_path", None)
        if anchor_rotation_prim_path is None:
            anchor_rotation_prim_path = self._xr_cfg.anchor_prim_path
        return self._get_prim_world_matrix(anchor_rotation_prim_path)

    def _set_anchor_rotation_origin(self, rotation_matrix: Any) -> None:
        self.__anchor_prim_initial_quat = rotation_matrix.ExtractRotationQuat()
        self.__anchor_prim_initial_yaw = self._get_anchor_yaw(rotation_matrix)

    def _get_anchor_yaw(self, rotation_matrix: Any) -> float:
        anchor_forward_axis = getattr(self._xr_cfg, "recenter_anchor_forward_axis", (1.0, 0.0, 0.0))
        return _yaw_from_matrix_axis(rotation_matrix, anchor_forward_axis)

    def _get_prim_world_matrix(self, prim_path: str | None) -> Any | None:
        if prim_path is None:
            return None

        stage_id = get_current_stage_id()
        rt_stage = usdrt.Usd.Stage.Attach(stage_id)
        if rt_stage is None:
            return None

        rt_prim = rt_stage.GetPrimAtPath(prim_path)
        if rt_prim is None:
            return None

        rt_xformable = Rt.Xformable(rt_prim)
        if rt_xformable is None:
            return None

        world_matrix_attr = rt_xformable.GetFabricHierarchyWorldMatrixAttr()
        if world_matrix_attr is None:
            return None

        return world_matrix_attr.Get()
