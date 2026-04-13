# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Diagnostic: compare exact forces sent to PhysX between WrenchComposer and v2.3.1 paths.

Monkey-patches apply_forces_and_torques_at_position to capture every call,
then runs the same scenario through both paths and diffs them step by step.
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import torch
import warp as wp
import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim import build_simulation_context

from isaaclab_assets.robots.anymal import ANYMAL_D_CFG


N_STEPS = 10
SEED = 42


def run_diagnostic():
    with build_simulation_context(device="cuda:0", gravity_enabled=True, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        # Create two identical robots
        cfg_composer = ANYMAL_D_CFG.replace(prim_path="/World/Composer/Robot")
        cfg_raw = ANYMAL_D_CFG.replace(prim_path="/World/Raw/Robot")

        robot_composer = Articulation(cfg_composer)
        robot_raw = Articulation(cfg_raw)

        sim.reset()

        robot_composer.update(sim.cfg.dt)
        robot_raw.update(sim.cfg.dt)

        # Apply a global downward payload force on base body
        body_ids_c, _ = robot_composer.find_bodies("base")
        body_ids_r, _ = robot_raw.find_bodies("base")
        num_bodies = robot_composer.num_bodies

        torch.manual_seed(SEED)
        payload_force = 2.0 * 9.81

        # Forces shaped for single body (for set_external_force_and_torque)
        forces_single = torch.zeros(1, 1, 3, device="cuda:0")
        forces_single[..., 2] = -payload_force
        torques_single = torch.zeros(1, 1, 3, device="cuda:0")

        # Forces shaped for all bodies (for raw PhysX call)
        forces_all = torch.zeros(1, num_bodies, 3, device="cuda:0")
        forces_all[:, body_ids_r[0], 2] = -payload_force
        torques_all = torch.zeros(1, num_bodies, 3, device="cuda:0")

        # Set payload on composer robot via wrench composer
        robot_composer.set_external_force_and_torque(
            forces_single.clone(), torques_single.clone(), body_ids=body_ids_c, is_global=True
        )

        # Capture PhysX calls
        calls_composer = []
        calls_raw = []

        def make_interceptor(original_fn, log_list):
            def interceptor(force_data=None, torque_data=None, position_data=None, indices=None, is_global=False):
                log_list.append({
                    "force": force_data.clone() if force_data is not None else None,
                    "torque": torque_data.clone() if torque_data is not None else None,
                    "position": position_data.clone() if position_data is not None else None,
                    "is_global": is_global,
                })
                return original_fn(
                    force_data=force_data, torque_data=torque_data,
                    position_data=position_data, indices=indices, is_global=is_global
                )
            return interceptor

        orig_composer = robot_composer.root_physx_view.apply_forces_and_torques_at_position
        orig_raw = robot_raw.root_physx_view.apply_forces_and_torques_at_position

        robot_composer.root_physx_view.apply_forces_and_torques_at_position = make_interceptor(orig_composer, calls_composer)
        robot_raw.root_physx_view.apply_forces_and_torques_at_position = make_interceptor(orig_raw, calls_raw)

        for step in range(N_STEPS):
            calls_composer.clear()
            calls_raw.clear()

            # Composer path: write_data_to_sim handles everything
            robot_composer.write_data_to_sim()

            # v2.3.1 path: apply forces directly to PhysX (simulating old behavior)
            robot_raw.root_physx_view.apply_forces_and_torques_at_position(
                force_data=forces_all.view(-1, 3),
                torque_data=torques_all.view(-1, 3),
                position_data=None,
                indices=robot_raw._ALL_INDICES,
                is_global=True,
            )

            sim.step()
            robot_composer.update(sim.cfg.dt)
            robot_raw.update(sim.cfg.dt)

            # Compare
            print(f"\n=== Step {step} ===")
            print(f"  Composer calls: {len(calls_composer)}")
            for i, call in enumerate(calls_composer):
                f = call['force']
                t = call['torque']
                ig = call['is_global']
                # Show only non-zero components for base body
                f_base = f[0] if f is not None else None
                t_base = t[0] if t is not None else None
                print(f"    Call {i}: is_global={ig}")
                print(f"      force[base]: {f_base}")
                print(f"      torque[base]: {t_base}")

            print(f"  Raw calls: {len(calls_raw)}")
            for i, call in enumerate(calls_raw):
                f = call['force']
                t = call['torque']
                ig = call['is_global']
                print(f"    Call {i}: is_global={ig}")
                print(f"      force: {f}")
                print(f"      torque: {t}")

            # Compare velocities
            vel_c = robot_composer.data.root_lin_vel_w[0]
            vel_r = robot_raw.data.root_lin_vel_w[0]
            diff = (vel_c - vel_r).abs().max().item()
            print(f"  Vel composer: {vel_c}")
            print(f"  Vel raw:      {vel_r}")
            print(f"  Max vel diff: {diff:.8f}")

            ang_c = robot_composer.data.root_ang_vel_w[0]
            ang_r = robot_raw.data.root_ang_vel_w[0]
            ang_diff = (ang_c - ang_r).abs().max().item()
            print(f"  Ang vel diff: {ang_diff:.8f}")


if __name__ == "__main__":
    run_diagnostic()
    simulation_app.close()
