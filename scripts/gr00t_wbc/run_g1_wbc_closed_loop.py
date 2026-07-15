"""Run a physically simulated G1 controlled by the GR00T WBC DDS bridge."""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-G1-GR00T-WBC-ClosedLoop-v0")
parser.add_argument("--state-port", type=int, default=5560)
parser.add_argument("--action-port", type=int, default=5561)
parser.add_argument("--action-timeout", type=float, default=0.20)
parser.add_argument("--startup-timeout", type=float, default=30.0)
parser.add_argument("--no-wait-for-action", action="store_true")
parser.add_argument("--no-realtime", action="store_true", help="Allow simulation to run faster than wall time.")
parser.add_argument("--min-root-height", type=float, default=0.45)
parser.add_argument(
    "--exit-on-fall",
    action="store_true",
    help="Exit on a detected fall instead of resetting G1 and recovering in place.",
)
parser.add_argument(
    "--fall-reset-hold-time",
    type=float,
    default=2.0,
    help="Seconds to hold the reset standing pose before blending back to WBC control.",
)
parser.add_argument("--settling-time", type=float, default=3.5)
parser.add_argument(
    "--control-blend-time",
    type=float,
    default=1.0,
    help="Seconds to blend from the supported default pose to live WBC targets before releasing the root.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import gymnasium as gym
import msgpack
import torch
import zmq

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import quat_apply_inverse
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.groot_joint_cfg import (
    G1_29DOF_JOINT_NAMES_ISAACLAB_ORDER,
)
from isaaclab_tasks.utils import parse_env_cfg

from g1_closed_loop_protocol import (
    ACTION_TOPIC,
    MUJOCO_TO_ISAACLAB,
    ROBOT_NAME,
    SCHEMA_VERSION,
    STATE_TOPIC,
    pack_topic_message,
    unpack_topic_message,
)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    robot = env.scene["robot"]
    joint_ids, resolved_names = robot.find_joints(G1_29DOF_JOINT_NAMES_ISAACLAB_ORDER, preserve_order=True)
    if list(resolved_names) != list(G1_29DOF_JOINT_NAMES_ISAACLAB_ORDER):
        raise RuntimeError(f"G1 joint order mismatch: {resolved_names}")
    torso_ids, torso_names = robot.find_bodies("torso_link", preserve_order=True)
    if len(torso_ids) != 1:
        raise RuntimeError(f"Expected one torso_link, resolved {torso_names}")
    torso_id = torso_ids[0]

    ctx = zmq.Context()
    state_pub = ctx.socket(zmq.PUB)
    state_pub.setsockopt(zmq.SNDHWM, 1)
    state_pub.bind(f"tcp://*:{args_cli.state_port}")
    action_sub = ctx.socket(zmq.SUB)
    action_sub.setsockopt(zmq.SUBSCRIBE, ACTION_TOPIC)
    action_sub.setsockopt(zmq.CONFLATE, 1)
    action_sub.connect(f"tcp://localhost:{args_cli.action_port}")

    env.reset()
    initial_root_pose = robot.data.root_state_w[:, :7].clone()
    default_action = robot.data.default_joint_pos[:, joint_ids].clone()
    action = default_action.clone()
    print(
        f"Initial G1 state: root_z={float(robot.data.root_pos_w[0, 2]):.3f}, "
        f"left_hip_pitch={float(robot.data.joint_pos[0, joint_ids[0]]):+.3f}, "
        f"left_knee={float(robot.data.joint_pos[0, joint_ids[1]]):+.3f}",
        flush=True,
    )
    sequence = 0
    reset_counter = 0
    last_action_wall_time = None
    first_action_deadline = time.monotonic() + args_cli.startup_timeout
    announced_active = False
    first_action_time = None
    controller_state = "INIT"
    controller_control_time = None
    last_telemetry_time = time.monotonic()
    next_control_deadline = time.monotonic()
    recovery_start_time = None
    hardware_to_policy = torch.tensor(MUJOCO_TO_ISAACLAB, dtype=torch.long, device=robot.device)

    print(f"Publishing Isaac state on tcp://*:{args_cli.state_port}")
    print(f"Receiving WBC actions from tcp://localhost:{args_cli.action_port}")
    print(
        "PhysX owns root motion during WBC control; initialization and fall recovery "
        "temporarily support the root pose."
    )
    if args_cli.exit_on_fall:
        print(f"Fall handling: exit when root_z < {args_cli.min_root_height:.3f} m")
    else:
        print(
            f"Fall handling: auto-reset when root_z < {args_cli.min_root_height:.3f} m, "
            f"hold {args_cli.fall_reset_hold_time:.2f}s, then blend back over "
            f"{args_cli.control_blend_time:.2f}s"
        )

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                q_policy = robot.data.joint_pos[0, joint_ids]
                dq_policy = robot.data.joint_vel[0, joint_ids]
                q_hw = torch.empty_like(q_policy)
                dq_hw = torch.empty_like(dq_policy)
                q_hw[torch.arange(29, device=robot.device)] = q_policy[hardware_to_policy]
                dq_hw[torch.arange(29, device=robot.device)] = dq_policy[hardware_to_policy]

                root_quat = robot.data.root_quat_w[0]
                root_ang_vel_b = quat_apply_inverse(root_quat, robot.data.root_ang_vel_w[0])
                torso_quat = robot.data.body_quat_w[0, torso_id]
                torso_ang_vel_b = quat_apply_inverse(torso_quat, robot.data.body_ang_vel_w[0, torso_id])
                state = {
                    "schema_version": SCHEMA_VERSION,
                    "robot": ROBOT_NAME,
                    "sequence": sequence,
                    "reset_counter": reset_counter,
                    "sim_time_s": sequence * env.step_dt,
                    "physics_dt": env_cfg.sim.dt,
                    "control_dt": env.step_dt,
                    "joint_order": "unitree_hardware_mujoco",
                    "joint_pos": q_hw.cpu().tolist(),
                    "joint_vel": dq_hw.cpu().tolist(),
                    "root_pos_world": robot.data.root_pos_w[0].cpu().tolist(),
                    "base_quat_wxyz": root_quat.cpu().tolist(),
                    "base_ang_vel_body": root_ang_vel_b.cpu().tolist(),
                    "torso_quat_wxyz": torso_quat.cpu().tolist(),
                    "torso_ang_vel_body": torso_ang_vel_b.cpu().tolist(),
                }
                state_pub.send(pack_topic_message(msgpack, STATE_TOPIC, state), flags=zmq.NOBLOCK)

                try:
                    raw = action_sub.recv(flags=zmq.NOBLOCK)
                    cmd = unpack_topic_message(msgpack, ACTION_TOPIC, raw)
                    if cmd.get("schema_version") != SCHEMA_VERSION or cmd.get("robot") != ROBOT_NAME:
                        raise ValueError(f"Incompatible WBC action header: {cmd}")
                    q_target_hw = torch.as_tensor(cmd["joint_pos_target"], device=robot.device, dtype=torch.float32)
                    if q_target_hw.numel() != 29 or not torch.isfinite(q_target_hw).all():
                        raise ValueError("WBC action must contain 29 finite joint targets")
                    q_target_policy = torch.empty_like(q_target_hw)
                    q_target_policy[hardware_to_policy] = q_target_hw
                    action[0] = q_target_policy
                    last_action_wall_time = time.monotonic()
                    controller_state = str(cmd.get("controller_state", "INIT"))
                    if controller_state == "CONTROL":
                        if controller_control_time is None:
                            controller_control_time = last_action_wall_time
                    else:
                        controller_control_time = None
                    if first_action_time is None:
                        first_action_time = last_action_wall_time
                except zmq.Again:
                    pass

                now = time.monotonic()
                if last_action_wall_time is None:
                    if not args_cli.no_wait_for_action and now > first_action_deadline:
                        raise TimeoutError("No WBC action received before startup timeout")
                    env.sim.render()
                    continue
                if now - last_action_wall_time > args_cli.action_timeout:
                    raise TimeoutError(f"WBC action stale for {now - last_action_wall_time:.3f}s")
                if not announced_active:
                    print("WBC action stream active; starting PhysX settling and closed-loop stepping.")
                    announced_active = True

                minimum_settling = first_action_time is not None and now - first_action_time < args_cli.settling_time
                control_elapsed = (
                    now - controller_control_time
                    if controller_state == "CONTROL" and controller_control_time is not None
                    else 0.0
                )
                if args_cli.control_blend_time > 0.0:
                    control_blend = min(max(control_elapsed / args_cli.control_blend_time, 0.0), 1.0)
                else:
                    control_blend = 1.0
                control_release_ready = controller_state == "CONTROL" and control_blend >= 1.0
                settling = minimum_settling or not control_release_ready
                active_blend = control_blend
                if controller_state == "CONTROL":
                    # Keep the floating base supported while smoothly transferring joint
                    # targets to the live controller.  This avoids applying a discontinuous
                    # pose change on the same frame that PhysX receives root ownership.
                    applied_action = torch.lerp(default_action, action, control_blend)
                else:
                    applied_action = default_action

                if recovery_start_time is not None:
                    recovery_elapsed = now - recovery_start_time
                    recovery_blend_elapsed = recovery_elapsed - args_cli.fall_reset_hold_time
                    if controller_state != "CONTROL" or recovery_blend_elapsed <= 0.0:
                        recovery_blend = 0.0
                    elif args_cli.control_blend_time > 0.0:
                        recovery_blend = min(recovery_blend_elapsed / args_cli.control_blend_time, 1.0)
                    else:
                        recovery_blend = 1.0
                    active_blend = recovery_blend
                    settling = recovery_blend < 1.0
                    applied_action = torch.lerp(default_action, action, recovery_blend)
                    if recovery_blend >= 1.0:
                        recovery_start_time = None
                        print(
                            f"Fall recovery complete after reset #{reset_counter}; "
                            "root ownership returned to PhysX/WBC.",
                            flush=True,
                        )

                _, _, terminated, truncated, _ = env.step(applied_action)
                if settling:
                    # Initialization-only support: keep the floating base at its reset pose while
                    # WBC fills histories and transitions INIT -> CONTROL. Root ownership is
                    # released permanently when settling ends.
                    robot.write_root_pose_to_sim(initial_root_pose)
                    robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))
                sequence += 1
                root_height = float(robot.data.root_pos_w[0, 2])
                fell = root_height < args_cli.min_root_height
                episode_done = bool(torch.any(terminated | truncated))
                if fell and args_cli.exit_on_fall:
                    raise RuntimeError(f"G1 fell below minimum root height: {root_height:.3f} m")
                if fell or episode_done:
                    reason = (
                        f"fall detected at root_z={root_height:.3f} m"
                        if fell
                        else "environment termination/truncation"
                    )
                    env.reset()
                    reset_counter += 1
                    initial_root_pose = robot.data.root_state_w[:, :7].clone()
                    recovery_start_time = time.monotonic()
                    next_control_deadline = recovery_start_time
                    print(
                        f"G1 auto-reset #{reset_counter}: {reason}. Holding the standing pose for "
                        f"{args_cli.fall_reset_hold_time:.2f}s before WBC blending.",
                        flush=True,
                    )
                    continue
                if now - last_telemetry_time >= 1.0:
                    source_sequence = cmd.get("source_state_sequence", -1) if "cmd" in locals() else -1
                    print(
                        f"seq={sequence} action_source={source_sequence} root_z={root_height:.3f} "
                        f"q0={float(q_policy[0]):+.3f} target0={float(applied_action[0, 0]):+.3f} "
                        f"mode={'SETTLING' if settling else 'WBC'} controller={controller_state} "
                        f"blend={active_blend:.2f} resets={reset_counter}",
                        flush=True,
                    )
                    last_telemetry_time = now
                if not args_cli.no_realtime:
                    next_control_deadline += env.step_dt
                    sleep_time = next_control_deadline - time.monotonic()
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
                    elif sleep_time < -5.0 * env.step_dt:
                        next_control_deadline = time.monotonic()
    finally:
        state_pub.close(linger=0)
        action_sub.close(linger=0)
        ctx.term()
        env.close()


if __name__ == "__main__":
    main()
