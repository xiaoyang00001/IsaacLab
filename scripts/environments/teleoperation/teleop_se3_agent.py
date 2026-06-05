# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run teleoperation with Isaac Lab manipulation environments.

Supports multiple input devices (e.g., keyboard, spacemouse, gamepad) and devices
configured within the environment (including OpenXR-based hand tracking or motion
controllers).

本脚本是 IsaacLab 遥操作入口，当前同时承担两类运行模式：

1. 普通 teleop：键盘 / SpaceMouse / Gamepad / OpenXR 设备直接产生环境 action。
2. GR00T/SONIC deploy target：外部 deploy 进程通过 ZMQ/DDS 写入目标，本脚本只给
   IsaacLab 环境送一个 shape 正确的零 action，用来推进仿真时钟。

第二种模式很重要：Locomanipulation G1 的 action space 不是 keyboard 的 7 维 SE(3)
命令，而是环境 action manager 汇总后的维度。直接把 7 维 keyboard action 送进去会触发
Invalid action shape。因此 deploy 模式下必须使用 env.action_space.shape 创建占位 action。
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher

# 这些 CLI 参数会在 Isaac Sim 启动前解析。AppLauncher 也会向同一个 parser 追加
# Isaac Sim/Kit 自己的参数（例如 --headless、--device、--renderer 等），所以自定义参数
# 必须先注册，再调用 AppLauncher.add_app_launcher_args(parser)。
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
# 追加 IsaacLab/Isaac Sim 通用启动参数。
AppLauncher.add_app_launcher_args(parser)
# 解析后的 args_cli 会在模块级使用；这是 IsaacLab 官方脚本常见模式，因为 simulation_app
# 必须在导入大部分 Isaac/Omni 模块之前启动。
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

if args_cli.enable_pinocchio:
    # Pinocchio 必须在 AppLauncher 之前导入，确保拿到 IsaacLab 环境里的版本，而不是
    # Isaac Sim 自带路径里的版本。Pink IK controller 和部分 humanoid retargeter 都依赖它。
    import pinocchio  # noqa: F401
if "handtracking" in args_cli.teleop_device.lower():
    # OpenXR hand tracking 需要 Isaac Sim 以 XR 模式启动；启动后再开会太晚。
    app_launcher_args["xr"] = True

# 启动 Omniverse/Isaac Sim 应用。下面很多 isaaclab / isaacsim 模块依赖 Kit runtime，
# 所以必须先创建 simulation_app。
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import logging
import os

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

if args_cli.task and "Locomanipulation" in args_cli.task:
    # Locomanipulation 任务不是默认必然被导入的包。这里显式 import 会执行该包的
    # __init__.py，把 Isaac-PickPlace-Locomanipulation-G1-Abs-v0 注册进 gymnasium registry。
    import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401

if args_cli.enable_pinocchio:
    # PickPlace 的 Pink IK 配置也会引用 Pinocchio 相关对象；显式导入保证相关 task/config 可用。
    import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401

logger = logging.getLogger(__name__)


def main() -> None:
    """
    Run teleoperation with an Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # 解析 task 对应的环境配置。parse_env_cfg 会根据 gym 注册信息创建 cfg 对象，并把
    # --device / --num_envs 等 CLI 覆盖进去。
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.env_name = args_cli.task
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise ValueError(
            "Teleoperation is only supported for ManagerBasedRLEnv environments. "
            f"Received environment config type: {type(env_cfg).__name__}"
        )
    # Teleop/可视化通常不希望因为 episode timeout 自动 reset；手动按 R 更可控。
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # Lift 任务默认会周期性重采样目标；遥操作时把 resampling 时间设得很大，避免目标跳变。
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # Lift 任务如果没有目标到达终止条件，成功后不会自然 reset。
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)

    if args_cli.xr:
        # XR 模式下移除普通相机配置，减少渲染负担；DLSS 通常比默认抗锯齿更适合 XR 画面。
        env_cfg = remove_camera_configs(env_cfg)
        env_cfg.sim.render.antialiasing_mode = "DLSS"

    try:
        # gym.make 会实例化 ManagerBasedRLEnv，并在内部创建 scene、manager、action/observation term 等。
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        # Reach 任务没有 gripper action，普通 SE(3) 设备里的 gripper bit 会被忽略。
        if "Reach" in args_cli.task:
            logger.warning(
                f"The environment '{args_cli.task}' does not support gripper control. The device command will be"
                " ignored."
            )
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        simulation_app.close()
        return

    # 运行时状态：
    # - should_reset_recording_instance：设备回调只置位，不直接 reset，避免在输入回调里改仿真状态。
    # - teleoperation_active：XR/设备可以临时暂停 action，但仍保持渲染。
    should_reset_recording_instance = False
    teleoperation_active = True
    # SONIC deploy target 模式：
    # Locomanipulation G1 环境内部有 SonicDeployTargetAction / UnitreeDdsLowCmdAction 这类 action term，
    # 它们自己从 ZMQ/DDS 收目标。外层 env.step(action) 仍然需要一个 action tensor，但该 tensor
    # 只用于满足 ActionManager 的 shape 检查，不代表真实键盘命令。
    #
    # SONIC_DEPLOY_TRANSPORT 默认按 "zmq" 处理，和 locomanipulation_g1_env_cfg.py 的默认值一致。
    # 如果这里默认空字符串，就会误走 keyboard teleop，导致 7 维 action 打进 28/29 维 action space。
    deploy_target_mode = bool(
        args_cli.task
        and "Locomanipulation" in args_cli.task
        and os.environ.get("SONIC_DEPLOY_TRANSPORT", "zmq").lower() in ("zmq", "dds")
    )

    # 输入设备回调。不同设备的按键名不完全一致，所以同一个语义可能绑定多个 key。
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

    # 传给 teleop device 的回调表。create_teleop_device 会根据设备类型选择可用 key。
    teleoperation_callbacks: dict[str, Callable[[], None]] = {
        "R": reset_recording_instance,
        "START": start_teleoperation,
        "STOP": stop_teleoperation,
        "RESET": reset_recording_instance,
    }

    # 手部追踪通常启动时先让用户摆好姿态/校准，所以默认不立即发送 teleop action；
    # 键盘、SpaceMouse、Gamepad 则默认 active，按键/摇杆一动就能控制。
    if args_cli.xr:
        teleoperation_active = False
    else:
        teleoperation_active = True

    # 普通 teleop 模式需要创建输入设备；deploy target 模式不需要，因为机器人目标来自外部进程。
    teleop_interface = None
    if deploy_target_mode:
        print(
            "[teleop_se3_agent] SONIC deploy target mode enabled; "
            "using zero env actions while SonicDeployTargetAction consumes external packets."
        )
    else:
        try:
            if hasattr(env_cfg, "teleop_devices") and args_cli.teleop_device in env_cfg.teleop_devices.devices:
                # 优先使用环境配置里的设备定义。Locomanipulation 这类任务可能提供自定义 retargeter，
                # 例如 XR 手柄/手追踪到 G1 上肢的映射。
                teleop_interface = create_teleop_device(
                    args_cli.teleop_device, env_cfg.teleop_devices.devices, teleoperation_callbacks
                )
            else:
                logger.warning(
                    f"No teleop device '{args_cli.teleop_device}' found in environment config. Creating default."
                )
                # fallback 设备只输出通用 SE(3) 命令，适合 Lift/Reach 这类标准 manipulation task。
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

                # fallback 设备手动挂回调；环境配置创建的设备通常会在 factory 内完成绑定。
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

    if teleop_interface is None and not deploy_target_mode:
        logger.error("Failed to create teleop interface")
        env.close()
        simulation_app.close()
        return

    if teleop_interface is not None:
        print(f"Using teleop device: {teleop_interface}")

    # 第一次 reset 会触发 reset events、同步默认 root/joint state，并清空 manager 内部历史。
    env.reset()
    if teleop_interface is not None:
        teleop_interface.reset()
    deploy_zero_actions = None
    if deploy_target_mode:
        # 关键点：这里必须按 env.action_space.shape 创建，而不是硬编码维度。
        # Locomanipulation 配置可能启用/关闭 walker、DDS/ZMQ、IK 等 action term，shape 会随配置变化。
        deploy_zero_actions = torch.zeros(env.action_space.shape, device=env.device)

    print("Teleoperation started. Press 'R' to reset the environment.")

    # 主循环：AppLauncher 持有的 simulation_app 关闭前持续推进环境。
    while simulation_app.is_running():
        try:
            # Teleop/deploy 都不需要 autograd；inference_mode 可以减少 tensor bookkeeping 开销。
            with torch.inference_mode():
                if deploy_target_mode:
                    # deploy 模式：零 action 只负责推进 ActionManager/Simulation。
                    # SonicDeployTargetAction 会在 process/apply 阶段自行消费最新 ZMQ/DDS 目标。
                    env.step(deploy_zero_actions)
                else:
                    # 普通 teleop：设备 advance() 返回一个单环境 action，例如 keyboard 的 7 维 SE(3)。
                    action = teleop_interface.advance()

                    # num_envs > 1 时，把同一条设备命令复制到所有并行环境。
                    if teleoperation_active:
                        actions = action.repeat(env.num_envs, 1)
                        env.step(actions)
                    else:
                        # 暂停 teleop 时不推进物理，只保持画面刷新。
                        env.sim.render()

                if should_reset_recording_instance:
                    # 延迟到主循环中 reset，保证 reset 与 env.step 不会在同一输入回调栈里交错。
                    env.reset()
                    if teleop_interface is not None:
                        teleop_interface.reset()
                    should_reset_recording_instance = False
                    print("Environment reset complete")
        except Exception as e:
            # 这里捕获错误是为了能优雅关闭 Isaac Sim；具体异常会打印在日志里。
            logger.error(f"Error during simulation step: {e}")
            break

    # 释放环境资源；simulation_app.close() 在 __main__ 里执行。
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
