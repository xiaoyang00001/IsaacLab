# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments.

Supports multiple input devices (e.g., keyboard, spacemouse, gamepad) and devices
configured within the environment (including OpenXR-based hand tracking or motion
controllers)."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Teleoperation for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help=(
        "Teleop device. Set here (legacy) or via the environment config. If using the environment config, pass the"
        " device key/name defined under 'teleop_devices' (it can be a custom name, not necessarily 'handtracking')."
        " Built-ins: keyboard, spacemouse, gamepad. Not all tasks support all built-ins."
    ),
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

G1_LOCOMANIP_TASK_ID = "Isaac-PickPlace-Locomanipulation-G1-Abs-v0"
G1_LOCOMANIP_UDP_TASK_ID = "Isaac-PickPlace-Locomanipulation-G1-UDP-Abs-v0"
G1_LOCOMANIP_HYBRID_TASK_ID = "Isaac-PickPlace-Locomanipulation-G1-Hybrid-Abs-v0"
G1_LOCOMANIP_TASK_ALIASES = {
    "Isaac-PickPlace-Locomanipulation-G1-Abs": G1_LOCOMANIP_TASK_ID,
    G1_LOCOMANIP_TASK_ID: G1_LOCOMANIP_TASK_ID,
    "Isaac-PickPlace-Locomanipulation-G1-UDP-Abs": G1_LOCOMANIP_UDP_TASK_ID,
    G1_LOCOMANIP_UDP_TASK_ID: G1_LOCOMANIP_UDP_TASK_ID,
    "Isaac-PickPlace-Locomanipulation-G1-Hybrid-Abs": G1_LOCOMANIP_HYBRID_TASK_ID,
    G1_LOCOMANIP_HYBRID_TASK_ID: G1_LOCOMANIP_HYBRID_TASK_ID,
}
G1_LOCOMANIP_TASK_IDS = {G1_LOCOMANIP_TASK_ID, G1_LOCOMANIP_UDP_TASK_ID, G1_LOCOMANIP_HYBRID_TASK_ID}
G1_PINK_LOCOMANIP_TASK_IDS = {G1_LOCOMANIP_TASK_ID, G1_LOCOMANIP_HYBRID_TASK_ID}


def _cli_flag_present(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv)


args_cli.task = G1_LOCOMANIP_TASK_ALIASES.get(args_cli.task, args_cli.task)

if (
    args_cli.task in G1_LOCOMANIP_TASK_IDS
    and not _cli_flag_present("--teleop_device")
    and args_cli.teleop_device == "keyboard"
):
    args_cli.teleop_device = "motion_controllers"

if args_cli.task in G1_PINK_LOCOMANIP_TASK_IDS:
    args_cli.enable_pinocchio = True

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and
    # not the one installed by Isaac Sim pinocchio is required by the Pink IK controllers and the
    # GR1T2 retargeter
    import pinocchio  # noqa: F401
if args_cli.teleop_device.lower() in {"handtracking", "motion_controllers"}:
    app_launcher_args["xr"] = True

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import logging

import gymnasium as gym
import torch

from isaaclab.devices import Se3Gamepad, Se3GamepadCfg, Se3Keyboard, Se3KeyboardCfg, Se3SpaceMouse, Se3SpaceMouseCfg
from isaaclab.devices.openxr import remove_camera_configs
from isaaclab.devices.teleop_device_factory import create_teleop_device
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.utils import parse_env_cfg

if args_cli.task in G1_LOCOMANIP_TASK_IDS:
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

if args_cli.enable_pinocchio:
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

# import logger
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # check environment name (for reach , we don't allow the gripper)
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # Flags for controlling teleoperation flow
    should_reset_recording_instance = False
    teleoperation_active = True
    local_robot_id = int(getattr(env_cfg, "local_robot_id", 1))
    env_reset_sync_term = None
    scene_state_sync_term = None
    pending_local_reset_id = None
    if args_cli.task == G1_LOCOMANIP_UDP_TASK_ID and "env_reset_sync" in env.action_manager.active_terms:
        env_reset_sync_term = env.action_manager.get_term("env_reset_sync")
    if args_cli.task == G1_LOCOMANIP_UDP_TASK_ID and "scene_state_sync" in env.action_manager.active_terms:
        scene_state_sync_term = env.action_manager.get_term("scene_state_sync")

    # Callback handlers
    def reset_recording_instance() -> None:
        """
        Reset the environment to its initial state.

        Sets a flag to reset the environment on the next simulation step.

        Returns:
            None
        """
        nonlocal should_reset_recording_instance
        should_reset_recording_instance = True
        print("Reset triggered - Environment will reset on next step")

    def request_synchronized_g1_reset() -> None:
        """Queue PC1's reset event, then reset the local environment at the safe step boundary."""

        nonlocal pending_local_reset_id
        if env_reset_sync_term is not None:
            pending_local_reset_id = env_reset_sync_term.request_local_reset()
        reset_recording_instance()

    def start_teleoperation() -> None:
        """
        Activate teleoperation control of the robot.

        Enables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = True
        print("Teleoperation activated")

    def stop_teleoperation() -> None:
        """
        Deactivate teleoperation control of the robot.

        Disables the application of teleoperation commands to the environment.

        Returns:
            None
        """
        nonlocal teleoperation_active
        teleoperation_active = False
        print("Teleoperation deactivated")

    # Create device config if not already in env_cfg
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
    }
    if args_cli.task != G1_LOCOMANIP_UDP_TASK_ID:
        teleoperation_callbacks["RESET"] = reset_recording_instance
    elif local_robot_id == 1:
        # OpenXR maps RESET to the left controller X button. Robot 2 does not
        # register this callback, so its local X button intentionally has no effect.
        teleoperation_callbacks["RESET"] = request_synchronized_g1_reset

    # For hand tracking devices, add additional callbacks
    if args_cli.xr and args_cli.task not in G1_LOCOMANIP_TASK_IDS:
        # Default to inactive for hand tracking
        teleoperation_active = False
    else:
        # Always active for other devices
        teleoperation_active = True

    # Create teleop device from config if present, otherwise create manually
    teleop_interface = None
    try:
        if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
            teleop_interface = create_teleop_device(
                args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
            )
        else:
            logger.warning(
                f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
            )
            # Create fallback teleop device
            sensitivity = args_cli.sensitivity
            if args_cli.teleop_device.lower() == "keyboard":
                teleop_interface = Se3Keyboard(
                    Se3KeyboardCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "spacemouse":
                teleop_interface = Se3SpaceMouse(
                    Se3SpaceMouseCfg(pos_sensitivity=0.05 * sensitivity, rot_sensitivity=0.05 * sensitivity)
                )
            elif args_cli.teleop_device.lower() == "gamepad":
                teleop_interface = Se3Gamepad(
                    Se3GamepadCfg(pos_sensitivity=0.1 * sensitivity, rot_sensitivity=0.1 * sensitivity)
                )
            else:
                logger.error(f"Unsupported teleop device: {args_cli.teleop_device}")
                logger.error("Configure the teleop device in the environment config.")
                env.close()
                simulation_app.close()
                return

            # Add callbacks to fallback device
            for key, callback in teleoperation_callbacks.items():
                try:
                    teleop_interface.add_callback(key, callback)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Failed to add callback for key {key}: {e}")
    except Exception as e:
        logger.error(f"Failed to create teleop device: {e}")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is None:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    print(f"Using teleop device: {teleop_interface}")
    print(
        f"[INFO] Teleoperation control loop: active={teleoperation_active}, "
        f"action_shape={env.action_space.shape}, active_terms={env.action_manager.active_terms}"
    )

    # reset environment
    env.reset()
    teleop_interface.reset()

    if args_cli.task == G1_LOCOMANIP_UDP_TASK_ID and local_robot_id == 1:
        print("Teleoperation started. Press robot-1 left-controller X to reset both Isaac Lab environments.")
    else:
        print("Teleoperation started. Press 'R' to reset the environment.")

    # simulate environment
    while simulation_app.is_running():
        try:
            # run everything in inference mode
            with torch.inference_mode():
                # get device command
                action = teleop_interface.advance()

                # Only apply teleop commands when active
                if teleoperation_active:
                    # The single-G1 GR00T/MuJoCo mirror is a zero-dimensional
                    # background action term. OpenXR remains active for the XR
                    # anchor/recenter path, but its device payload (often a dict
                    # when no retargeters are configured) is not an environment
                    # action. Let the mirror term consume UDP data internally.
                    if args_cli.task == G1_LOCOMANIP_UDP_TASK_ID and env.action_space.shape[-1] == 0:
                        actions = torch.zeros(
                            env.action_space.shape,
                            dtype=torch.float32,
                            device=env.unwrapped.device,
                        )
                    else:
                        # process actions
                        actions = action.repeat(env.num_envs, 1)
                    # apply actions
                    env.step(actions)
                else:
                    env.sim.render()

                remote_reset_id = (
                    env_reset_sync_term.consume_remote_reset_request()
                    if env_reset_sync_term is not None
                    else None
                )
                if remote_reset_id is not None:
                    if scene_state_sync_term is not None:
                        scene_state_sync_term.expect_reset_id(remote_reset_id)
                    should_reset_recording_instance = True
                    print(f"Remote reset received ({remote_reset_id}) - Environment will reset now")

                if should_reset_recording_instance:
                    env.reset()
                    if scene_state_sync_term is not None and pending_local_reset_id is not None:
                        scene_state_sync_term.set_publisher_reset_id(pending_local_reset_id)
                    pending_local_reset_id = None
                    teleop_interface.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
        except Exception as e:
            logger.error(f"Error during simulation step: {e}")
            break

    # close the simulator
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
