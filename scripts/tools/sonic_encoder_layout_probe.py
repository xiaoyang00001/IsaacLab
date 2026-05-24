# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone SONIC encoder layout probe.

Quickly A/B test different layout / mode / fill strategies for the SONIC encoder
without booting IsaacSim. Compares the resulting token_state + decoder action.

Run::

    D:/miniconda3/envs/env_isaaclab/python.exe scripts/tools/sonic_encoder_layout_probe.py
"""

import numpy as np
import onnxruntime as ort

ENC_PATH = "D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx"
DEC_PATH = "D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx"


def fmt_row(label, tokens, action):
    return (
        f"{label:55s} | "
        f"tok abs={np.abs(tokens).max():6.3f} mean={tokens.mean():+6.3f} std={tokens.std():6.3f} | "
        f"act abs={np.abs(action).max():6.3f} mean={action.mean():+6.3f} std={action.std():6.3f}"
    )


def main():
    enc = ort.InferenceSession(ENC_PATH, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(DEC_PATH, providers=["CPUExecutionProvider"])
    enc_in_name = enc.get_inputs()[0].name
    enc_out_name = enc.get_outputs()[0].name
    dec_in_name = dec.get_inputs()[0].name
    dec_out_name = dec.get_outputs()[0].name

    # G1 default-pose 14-body positions in pelvis frame (rough estimate matching abs ~0.75)
    # 顺序: pelvis, lhip_roll, lknee, lankle_roll, rhip_roll, rknee, rankle_roll,
    #       torso, lshoulder_roll, lelbow, lwrist_yaw, rshoulder_roll, relbow, rwrist_yaw
    body_pos_b = np.array(
        [
            [0.0, 0.0, 0.0],  # pelvis (root itself)
            [0.0, 0.08, -0.1],  # left hip roll
            [0.0, 0.08, -0.4],  # left knee
            [0.0, 0.08, -0.75],  # left ankle roll
            [0.0, -0.08, -0.1],  # right hip roll
            [0.0, -0.08, -0.4],  # right knee
            [0.0, -0.08, -0.75],  # right ankle roll
            [0.0, 0.0, 0.4],  # torso
            [0.0, 0.15, 0.5],  # left shoulder
            [0.0, 0.20, 0.2],  # left elbow
            [0.0, 0.25, -0.1],  # left wrist
            [0.0, -0.15, 0.5],  # right shoulder
            [0.0, -0.20, 0.2],  # right elbow
            [0.0, -0.25, -0.1],  # right wrist
        ],
        dtype=np.float32,
    )
    body_flat = body_pos_b.flatten()  # (42,)
    id_col = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)  # rotation matrix first-2-cols, col-major
    id_row = np.array([1, 0, 0, 1, 0, 0], dtype=np.float32)  # row-major flatten

    # ── G1 default joint pos (按 SONIC_G1_29DOF_JOINT_ORDER 顺序) ────
    default_jp = np.zeros(29, dtype=np.float32)
    default_jp[0] = -0.10  # lhp
    default_jp[3] = 0.30   # lknee
    default_jp[4] = -0.20  # lap
    default_jp[6] = -0.10  # rhp
    default_jp[9] = 0.30   # rknee
    default_jp[10] = -0.20  # rap

    def build_decoder_history(joint_pos_rel=None, gravity_b=None):
        """构造 decoder 994D 输入，按 dim-major 拼。"""
        dec = np.zeros((1, 994), dtype=np.float32)
        # offset 64:94 = his_base_ang_vel × 10 (zero static)
        # offset 94:384 = his_joint_pos 29 × 10 (dim-major)
        if joint_pos_rel is not None:
            dec[0, 94:384] = np.repeat(joint_pos_rel, 10)
        # offset 384:674 = his_joint_vel × 10 (zero static)
        # offset 674:964 = his_last_actions × 10 (zero)
        # offset 964:994 = his_gravity_dir × 10 (dim-major)
        if gravity_b is not None:
            dec[0, 964:994] = np.repeat(gravity_b, 10)
        return dec

    def run(label, enc_input, dec_history=None):
        tokens = enc.run([enc_out_name], {enc_in_name: enc_input})[0]
        if dec_history is None:
            dec_in = np.zeros((1, 994), dtype=np.float32)
        else:
            dec_in = dec_history.copy()
        dec_in[:, :64] = tokens
        action = dec.run([dec_out_name], {dec_in_name: dec_in})[0][0]
        print(fmt_row(label, tokens, action))

    print("=" * 130)
    print("SONIC encoder layout probe — 1762D → 64D tokens → 994D decoder (zero history) → 29D action")
    print("=" * 130)

    # ── baselines ──────────────────────────────────────────────
    e0 = np.zeros((1, 1762), dtype=np.float32)
    run("baseline: all zero (mode=0)", e0)

    e1 = e0.copy()
    e1[0, 0] = 0.0  # explicit mode_id=0
    run("explicit mode_id=0 only", e1)

    # ── mode sanity (cast long > 0 picks other encoders) ───────
    for mid in [1.0, 2.0]:
        e = e0.copy()
        e[0, 0] = mid
        run(f"mode_id={mid:.0f} only ({int(mid)}=teleop/smpl)", e)

    # ── body_pos at offset 1:421 (420D) ───────────────────────
    for layout, fill in [("frame-major tile", np.tile(body_flat, 10)), ("dim-major repeat", np.repeat(body_flat, 10))]:
        e = e0.copy()
        e[0, 1:421] = fill
        run(f"body_pos only [{layout}]", e)

    # ── identity 6D at offset 431:491 (60D) ───────────────────
    for id_name, id_vec in [("col-major [1,0,0,0,1,0]", id_col), ("row-major [1,0,0,1,0,0]", id_row)]:
        for layout, fn in [("tile", np.tile), ("repeat", np.repeat)]:
            e = e0.copy()
            e[0, 431:491] = fn(id_vec, 10)
            run(f"identity6 only [{id_name}, {layout}]", e)

    # ── combined ──────────────────────────────────────────────
    for body_layout, body_fill in [
        ("body frame-major", np.tile(body_flat, 10)),
        ("body dim-major", np.repeat(body_flat, 10)),
    ]:
        for id_name, id_vec in [("id col-major", id_col), ("id row-major", id_row)]:
            for id_layout, fn in [("id tile", np.tile), ("id repeat", np.repeat)]:
                e = e0.copy()
                e[0, 1:421] = body_fill
                e[0, 431:491] = fn(id_vec, 10)
                run(f"COMBINED {body_layout} + {id_name} {id_layout}", e)

    # ── shift body_pos to different offsets (in case 1:421 is wrong) ──
    for off, label in [(1, "default"), (5, "+4 shift"), (4, "+3 (after 4D mode)")]:
        e = e0.copy()
        e[0, off : off + 420] = np.repeat(body_flat, 10)
        run(f"body_pos dim-major at offset {off} ({label})", e)

    print()
    print("=" * 130)
    print("E4 诊断: 真实 decoder history (静止 default pose) 是否复现 absmax=12 garbage")
    print("=" * 130)

    # 静止机器人在 default pose 时的真实 history:
    # - joint_vel = 0, base_ang_vel = 0 (静止)
    # - gravity_b = [0, 0, -1] (机器人直立)
    # - joint_pos: 当前代码传 ABSOLUTE → default_jp; 训练用 joint_pos_rel → 应该是 0
    gravity_b_static = np.array([0, 0, -1], dtype=np.float32)
    hist_abs = build_decoder_history(joint_pos_rel=default_jp, gravity_b=gravity_b_static)
    hist_rel = build_decoder_history(joint_pos_rel=np.zeros(29, dtype=np.float32), gravity_b=gravity_b_static)

    # 测试：(encoder, decoder history) 4 种组合
    e_zero = np.zeros((1, 1762), dtype=np.float32)
    e_ref = e_zero.copy()
    e_ref[0, 1:421] = np.repeat(body_flat, 10)
    e_ref[0, 431:491] = np.repeat(id_row, 10)

    print("\n[A] encoder zero + static history (模拟 sonic_verify 阶段 3.1 dim-major baseline):")
    run("  joint_pos=absolute (current code BUG?)", e_zero, hist_abs)
    run("  joint_pos=relative=0 (correct)         ", e_zero, hist_rel)

    print("\n[B] encoder self-ref + static history (模拟当前 D1 v2 实际跑的):")
    run("  joint_pos=absolute (current code BUG?)", e_ref, hist_abs)
    run("  joint_pos=relative=0 (correct)         ", e_ref, hist_rel)

    print("\n[C] sanity: zero everything")
    run("  zero encoder + zero history (baseline) ", e_zero, None)

    print()
    print("如果 [A] absolute 的 absmax >> relative → 复现了 sonic_verify 里的 garbage，joint_pos_rel 是关键修复点")


if __name__ == "__main__":
    main()
