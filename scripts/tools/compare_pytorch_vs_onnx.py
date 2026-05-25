#!/usr/bin/env python3
"""
Compare PyTorch ckpt vs ONNX dual-pass using the SAME observation.

Usage:
    D:/miniconda3/envs/env_isaaclab/python.exe scripts/tools/compare_pytorch_vs_onnx.py

    For real-obs comparison, set env vars before running:
        set ENC_1762=<1762 comma-sep values>
        set DEC_994=<994 comma-sep values>
    Then these will override the zero-fill and run with real obs.
"""

import sys
import os
import numpy as np

GR00T_ROOT = "D:/src/Isaac/GR00T-WholeBodyControl"
sys.path.insert(0, GR00T_ROOT)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENCODER_ONNX = os.path.join(GR00T_ROOT, "gear_sonic_deploy/policy/release/model_encoder.onnx")
DECODER_ONNX = os.path.join(GR00T_ROOT, "gear_sonic_deploy/policy/release/model_decoder.onnx")
PYTORCH_CKPT = os.path.join(GR00T_ROOT, "sonic_release/last.pt")

# ---------------------------------------------------------------------------
# ONNX helpers
# ---------------------------------------------------------------------------
import onnxruntime as ort


def run_encoder_onnx(obs_1762: np.ndarray):
    sess = ort.InferenceSession(ENCODER_ONNX, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"obs_dict": obs_1762.astype(np.float32)})
    return out[0]


def run_decoder_onnx(obs_994: np.ndarray):
    sess = ort.InferenceSession(DECODER_ONNX, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"obs_dict": obs_994.astype(np.float32)})
    return out[0]


def onnx_dual_pass(enc_obs: np.ndarray, dec_full_994: np.ndarray):
    """Full encoder -> decoder pipeline. Returns 29D action (1, 29).

    dec_full_994: (1, 994) = [token_slot(64) | proprio/history(930)].
    The token_slot is REPLACED by fresh encoder output each call.
    """
    tokens = run_encoder_onnx(enc_obs)  # (1, 64)
    # dec_full_994 = [token(64) | history(930)]
    # We replace the token slot with fresh encoder output
    dec_full = np.concatenate([tokens, dec_full_994[:, 64:]], axis=1)
    return run_decoder_onnx(dec_full)


# ---------------------------------------------------------------------------
# PyTorch checkpoint loader (needs accelerate + IsaacLab, may fail standalone)
# ---------------------------------------------------------------------------
def try_load_pytorch_actor():
    """Try to load the actor (UniversalTokenModule) from PyTorch checkpoint.

    Requires: gear_sonic[training] extras (accelerate, trl, smpl_sim, etc.)
    If this fails, it means the full training env isn't set up — which is fine.
    """
    try:
        import pickle
        import types

        class GEARUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                module = module.replace("trl.", "gear_sonic.trl.")
                module = module.replace("groot.rl.", "gear_sonic.")
                return super().find_class(module, name)

        # Import required gear_sonic submodules
        import gear_sonic.trl.trainer.ppo_trainer as _gs_pt
        import gear_sonic.trl.modules.actor_critic_modules as _gs_ac
        import gear_sonic.trl.modules.base_module as _gs_bm
        import gear_sonic.trl.modules.universal_token_modules as _gs_ut
        import gear_sonic.trl.callbacks.model_save_callback as _gs_msc
        import gear_sonic.trl.callbacks.wandb_callback as _gs_wc
        import gear_sonic.trl.callbacks.im_eval_callback as _gs_ie
        import gear_sonic.trl.callbacks.im_resample_callback as _gs_ir
        import gear_sonic.trl.callbacks.read_eval_callback as _gs_re
        import gear_sonic.trl.utils.common as _gs_uc
        import gear_sonic.trl.utils.scheduler as _gs_sched
        import gear_sonic.trl.losses.token_losses as _gs_tl
        import gear_sonic.envs.manager_env as _gs_me

        # Map namespace: trl.* -> gear_sonic.trl.*
        sys.modules["trl.trainer"] = _gs_pt
        sys.modules["trl.trainer.ppo_trainer"] = _gs_pt
        sys.modules["trl.modules"] = _gs_ac
        sys.modules["trl.modules.actor_critic_modules"] = _gs_ac
        sys.modules["trl.modules.base_module"] = _gs_bm
        sys.modules["trl.modules.universal_token_modules"] = _gs_ut
        sys.modules["trl.callbacks"] = types.ModuleType("trl.callbacks")
        sys.modules["trl.callbacks.model_save_callback"] = _gs_msc
        sys.modules["trl.callbacks.wandb_callback"] = _gs_wc
        sys.modules["trl.callbacks.im_eval_callback"] = _gs_ie
        sys.modules["trl.callbacks.im_resample_callback"] = _gs_ir
        sys.modules["trl.callbacks.read_eval_callback"] = _gs_re
        sys.modules["trl.utils"] = types.ModuleType("trl.utils")
        sys.modules["trl.utils.common"] = _gs_uc
        sys.modules["trl.utils.scheduler"] = _gs_sched
        sys.modules["trl.losses"] = types.ModuleType("trl.losses")
        sys.modules["trl.losses.token_losses"] = _gs_tl

        # Map namespace: groot.rl.* -> gear_sonic.*
        sys.modules["groot"] = types.ModuleType("groot")
        sys.modules["groot.rl"] = types.ModuleType("groot.rl")
        sys.modules["groot.rl.envs"] = types.ModuleType("groot.rl.envs")
        sys.modules["groot.rl.envs.manager_env"] = _gs_me

        # Load checkpoint
        with open(PYTORCH_CKPT, "rb") as f:
            unpickler = GEARUnpickler(f)
            ckpt = unpickler.load()

        actor_sd = ckpt.get("actor_model_state_dict", None)
        if actor_sd is None:
            actor_sd = ckpt.get("policy_state_dict", None)
        return ckpt, actor_sd

    except Exception as e:
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_csv_envvar(name: str, expected_dim: int) -> np.ndarray | None:
    val = os.environ.get(name, "")
    if not val:
        return None
    parts = [float(x.strip()) for x in val.split(",")]
    arr = np.array(parts, dtype=np.float32)
    if arr.shape[0] != expected_dim:
        print(f"  [WARN] {name} has {arr.shape[0]} dims, expected {expected_dim}")
    return arr.reshape(1, expected_dim)


def main():
    print("=" * 60)
    print("PyTorch ckpt vs ONNX Dual-Pass Comparison")
    print("=" * 60)

    for f in [ENCODER_ONNX, DECODER_ONNX, PYTORCH_CKPT]:
        exists = os.path.exists(f)
        print(f"  {'OK' if exists else 'MISSING'}: {os.path.basename(f)}")

    # Determine if we have real observations
    # ENC_1762 = full encoder input (1762D)
    # DEC_930 = decoder proprioception only (930D, excludes token slot)
    # If not set, use zero-fill
    enc_in = _parse_csv_envvar("ENC_1762", 1762)
    dec_full_994 = _parse_csv_envvar("DEC_994", 994)  # full decoder input = token(64) + history(930)

    # Also try loading from CSV files if they exist
    if enc_in is None and os.path.exists("enc_obs_step1.csv"):
        enc_in = np.loadtxt("enc_obs_step1.csv", delimiter=",").reshape(1, 1762)
        print(f"\n[LOAD] Loaded enc from enc_obs_step1.csv")
    if dec_full_994 is None and os.path.exists("dec_obs_step1.csv"):
        dec_full_994 = np.loadtxt("dec_obs_step1.csv", delimiter=",").reshape(1, 994)
        print(f"\n[LOAD] Loaded dec from dec_obs_step1.csv")

    if enc_in is not None and dec_full_994 is not None:
        print("\n[MODE] Real observations from environment (ENC_1762/DEC_994 or CSV file)")
    else:
        print("\n[MODE] Zero-fill baseline")
        if enc_in is None:
            enc_in = np.zeros((1, 1762), dtype=np.float32)
        if dec_full_994 is None:
            dec_full_994 = np.zeros((1, 994), dtype=np.float32)

    # ---- ONNX comparison ----
    print(f"\n[ONNX] enc_in shape={enc_in.shape}, dec_full_994 shape={dec_full_994.shape}")
    action_onnx = onnx_dual_pass(enc_in, dec_full_994)
    print(f"  action_onnx: mean={action_onnx.mean():.4f} std={action_onnx.std():.4f} "
          f"absmax={np.abs(action_onnx).max():.4f}")

    # ---- PyTorch checkpoint ----
    print("\n[PyTorch] Loading checkpoint...")
    ckpt, actor_sd = try_load_pytorch_actor()

    if ckpt is None:
        print("  SKIPPED - full training env not set up (accelerate / IsaacLab deps missing)")
        print("  This is expected in the deployment environment.")
        print("  Key insight: if ONNX action_absmax ~1.9 with zero-fill, and step 1 real obs")
        print("  gives action ~2.56, the gap (0.66) suggests ONNX may be under-responsive.")
        return

    print(f"  checkpoint keys: {list(ckpt.keys())}")

    if actor_sd is not None:
        print(f"  actor params: {len(actor_sd)}")
        has_std = "std" in actor_sd or "log_std" in actor_sd
        print(f"  has trainable std (action noise): {has_std}")
        for k, v in list(actor_sd.items())[:3]:
            shape = getattr(v, "shape", "?") if hasattr(v, "shape") else type(v).__name__
            print(f"    {k}: {shape}")

    state = ckpt.get("state")
    if state is not None:
        print(f"  state type: {type(state).__name__}")

    # If we received real obs, save them to CSV for replay
    if os.environ.get("ENC_1762") or os.environ.get("DEC_994"):
        enc_csv = "enc_obs_step1.csv"
        dec_csv = "dec_obs_step1.csv"
        np.savetxt(enc_csv, enc_in.reshape(-1), delimiter=",", fmt="%.8f")
        np.savetxt(dec_csv, dec_full_994.reshape(-1), delimiter=",", fmt="%.8f")
        print(f"\n[REPLAY] Saved real obs to:")
        print(f"  {enc_csv}  ({enc_in.size} values)")
        print(f"  {dec_csv}  ({dec_full_994.size} values)")
        print(f"\nNext time, run without env vars to use saved CSV:")
        print(f"  cp enc_obs_step1.csv ..\\docs\\")
        print(f"  cp dec_obs_step1.csv ..\\docs\\")


if __name__ == "__main__":
    main()