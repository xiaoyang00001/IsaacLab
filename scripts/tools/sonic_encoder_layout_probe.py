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

    def run(label, enc_input):
        tokens = enc.run([enc_out_name], {enc_in_name: enc_input})[0]
        dec_in = np.zeros((1, 994), dtype=np.float32)
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


if __name__ == "__main__":
    main()
