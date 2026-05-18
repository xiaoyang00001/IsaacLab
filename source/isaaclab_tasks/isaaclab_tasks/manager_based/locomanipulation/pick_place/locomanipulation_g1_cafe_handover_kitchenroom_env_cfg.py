# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from .locomanipulation_g1_cafe_handover_env_cfg import (
    CafeHandoverG1EnvCfg as BaseCafeHandoverG1EnvCfg,
    CafeHandoverG1SceneCfg as BaseCafeHandoverG1SceneCfg,
    FIXED_G1_29DOF_CFG,
    REMOTE_FIXED_G1_29DOF_CFG,
    ZMQ_SYNC_ROLE,
)


def _resolve_lightwheel_kitchen_room_usd_path() -> str:
    """Resolve the Lightwheel KitchenRoom USD path from env var or default download location."""
    root_dir = os.environ.get(
        "LIGHTWHEEL_OPEN_SOURCE_ROOT_DIR",
        r"D:\Downloads\Lightwheel_OpenSource\Lightwheel_OpenSource",
    )
    return os.path.join(root_dir, "Locomotion", "KitchenRoom", "KitchenRoom.usd")


LIGHTWHEEL_KITCHEN_ROOM_USD_PATH = _resolve_lightwheel_kitchen_room_usd_path()

KITCHEN_ROOM_ROBOT_A_POS = (0.18, -0.42, 0.75)
KITCHEN_ROOM_ROBOT_A_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)
KITCHEN_ROOM_ROBOT_B_POS = (0.26, 0.80, 0.75)
KITCHEN_ROOM_ROBOT_B_QUAT = (0.7071068, 0.0, 0.0, -0.7071068)
KITCHEN_ROOM_CUP_SPAWN_POS = (0.02, 0.04, 0.92)
KITCHEN_ROOM_HANDOVER_ZONE_POS = (0.22, 0.22, 1.00)
KITCHEN_ROOM_SERVE_ZONE_POS = (0.45, 0.46, 0.94)
KITCHEN_ROOM_VIEWER_ANCHOR_POS = (0.22, 0.22, 1.00)


@configclass
class CafeHandoverG1KitchenRoomSceneCfg(BaseCafeHandoverG1SceneCfg):
    """Scene cfg that swaps the placeholder cafe scene for the Lightwheel KitchenRoom USD."""

    # Remove placeholder cafe geometry so only the downloaded KitchenRoom background remains.
    counter = None
    serve_counter = None
    ground = None
    robot_spawn_a_marker = None
    robot_spawn_b_marker = None
    cup_spawn_marker = None
    handover_zone_marker = None
    serve_zone_marker = None
    viewer_anchor_marker = None

    robot = FIXED_G1_29DOF_CFG.copy()
    robot.init_state.pos = KITCHEN_ROOM_ROBOT_A_POS
    robot.init_state.rot = KITCHEN_ROOM_ROBOT_A_QUAT

    remote_robot = REMOTE_FIXED_G1_29DOF_CFG.copy()
    remote_robot.prim_path = "{ENV_REGEX_NS}/RemoteRobot"
    remote_robot.init_state.pos = KITCHEN_ROOM_ROBOT_B_POS
    remote_robot.init_state.rot = KITCHEN_ROOM_ROBOT_B_QUAT

    cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=KITCHEN_ROOM_CUP_SPAWN_POS, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.035,
            height=0.12,
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.53, 0.33, 0.12)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=(ZMQ_SYNC_ROLE == "subscriber"),
                disable_gravity=(ZMQ_SYNC_ROLE == "subscriber"),
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
        ),
    )

    robot_spawn_a_anchor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotSpawnA",
        init_state=AssetBaseCfg.InitialStateCfg(pos=KITCHEN_ROOM_ROBOT_A_POS, rot=KITCHEN_ROOM_ROBOT_A_QUAT),
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.55, 1.0), opacity=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    robot_spawn_b_anchor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotSpawnB",
        init_state=AssetBaseCfg.InitialStateCfg(pos=KITCHEN_ROOM_ROBOT_B_POS, rot=KITCHEN_ROOM_ROBOT_B_QUAT),
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.15), opacity=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    cup_spawn_anchor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CupSpawn",
        init_state=AssetBaseCfg.InitialStateCfg(pos=KITCHEN_ROOM_CUP_SPAWN_POS),
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.9, 0.15), opacity=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    handover_zone_anchor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/HandoverZone",
        init_state=AssetBaseCfg.InitialStateCfg(pos=KITCHEN_ROOM_HANDOVER_ZONE_POS),
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.14, 0.12),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.8, 0.3), opacity=0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    serve_zone_anchor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ServeZone",
        init_state=AssetBaseCfg.InitialStateCfg(pos=KITCHEN_ROOM_SERVE_ZONE_POS),
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.14, 0.12),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.25, 0.2), opacity=0.12),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )
    viewer_anchor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ViewerAnchor",
        init_state=AssetBaseCfg.InitialStateCfg(pos=KITCHEN_ROOM_VIEWER_ANCHOR_POS),
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), opacity=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
    )

    background = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Background",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=UsdFileCfg(
            usd_path=LIGHTWHEEL_KITCHEN_ROOM_USD_PATH,
        ),
    )


@configclass
class CafeHandoverG1KitchenRoomEnvCfg(BaseCafeHandoverG1EnvCfg):
    """Cafe handover env cfg using the downloaded Lightwheel KitchenRoom background."""

    scene: CafeHandoverG1KitchenRoomSceneCfg = CafeHandoverG1KitchenRoomSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=False,
    )

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy.handover_zone_pos.params["fallback_pos"] = KITCHEN_ROOM_HANDOVER_ZONE_POS
        self.observations.policy.serve_zone_pos.params["fallback_pos"] = KITCHEN_ROOM_SERVE_ZONE_POS

        for term_name in (
            "task_phase_index",
            "task_phase_one_hot",
            "pickup_success",
            "handover_zone_reached",
            "handover_success",
            "serve_success",
        ):
            term_params = getattr(self.observations.policy, term_name).params
            term_params["fallback_cup_spawn_pos"] = KITCHEN_ROOM_CUP_SPAWN_POS
            term_params["fallback_handover_zone_pos"] = KITCHEN_ROOM_HANDOVER_ZONE_POS
            term_params["fallback_serve_zone_pos"] = KITCHEN_ROOM_SERVE_ZONE_POS

        for event_name in ("place_robots_startup", "place_robots_reset"):
            term_params = getattr(self.events, event_name).params
            term_params["fallback_robot_a_pos"] = KITCHEN_ROOM_ROBOT_A_POS
            term_params["fallback_robot_a_quat"] = KITCHEN_ROOM_ROBOT_A_QUAT
            term_params["fallback_robot_b_pos"] = KITCHEN_ROOM_ROBOT_B_POS
            term_params["fallback_robot_b_quat"] = KITCHEN_ROOM_ROBOT_B_QUAT

        for event_name in ("place_cup_startup", "place_cup_reset"):
            getattr(self.events, event_name).params["fallback_pos"] = KITCHEN_ROOM_CUP_SPAWN_POS

        self.events.align_viewer_startup.params["fallback_target"] = KITCHEN_ROOM_VIEWER_ANCHOR_POS
        self.events.log_phase_transitions.params["fallback_cup_spawn_pos"] = KITCHEN_ROOM_CUP_SPAWN_POS
        self.events.log_phase_transitions.params["fallback_handover_zone_pos"] = KITCHEN_ROOM_HANDOVER_ZONE_POS
        self.events.log_phase_transitions.params["fallback_serve_zone_pos"] = KITCHEN_ROOM_SERVE_ZONE_POS

        self.terminations.success.params["fallback_target_pos"] = KITCHEN_ROOM_SERVE_ZONE_POS
