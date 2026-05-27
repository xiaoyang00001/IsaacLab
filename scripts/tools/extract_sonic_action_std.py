#!/usr/bin/env python3
"""Extract per-joint action noise std (29,) from sonic_release/last.pt.

Output: source/isaaclab_tasks/.../pick_place/data/sonic_action_std_29d.npy

Usage:
    D:/miniconda3/envs/env_isaaclab/python.exe scripts/tools/extract_sonic_action_std.py
"""
import sys
import os
import types
import pickle
import importlib
import numpy as np

GR00T_ROOT = "D:/src/Isaac/GR00T-WholeBodyControl"
sys.path.insert(0, GR00T_ROOT)

if "open3d" not in sys.modules:
    _o3d = types.ModuleType("open3d")
    _o3d_io = types.ModuleType("open3d.io")
    _o3d_io.read_triangle_mesh = lambda *a, **kw: None
    _o3d_io.write_triangle_mesh = lambda *a, **kw: None
    _o3d.io = _o3d_io
    sys.modules["open3d"] = _o3d
    sys.modules["open3d.io"] = _o3d_io

PYTORCH_CKPT = os.path.join(GR00T_ROOT, "sonic_release/last.pt")
OUT_DIR = "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/data"
OUT_PATH = os.path.join(OUT_DIR, "sonic_action_std_29d.npy")


def load_ckpt():
    import torch

    for m in [
        "gear_sonic.trl.trainer.ppo_trainer",
        "gear_sonic.trl.modules.actor_critic_modules",
        "gear_sonic.trl.modules.base_module",
        "gear_sonic.trl.modules.universal_token_modules",
        "gear_sonic.envs.manager_env",
    ]:
        importlib.import_module(m)

    class _GEARUnpickler(pickle.Unpickler):
        _PREFIX_MAP = (
            ("trl.trainer.", "gear_sonic.trl.trainer."),
            ("trl.modules.", "gear_sonic.trl.modules."),
            ("trl.callbacks.", "gear_sonic.trl.callbacks."),
            ("trl.utils.", "gear_sonic.trl.utils."),
            ("trl.losses.", "gear_sonic.trl.losses."),
            ("groot.rl.envs.manager_env", "gear_sonic.envs.manager_env"),
            ("groot.rl.envs.", "gear_sonic.envs."),
            ("groot.rl.", "gear_sonic."),
        )

        def find_class(self, module, name):
            for src, dst in self._PREFIX_MAP:
                if module.startswith(src):
                    mapped = dst + module[len(src):]
                    try:
                        return super().find_class(mapped, name)
                    except (ImportError, AttributeError, ModuleNotFoundError):
                        break
            return super().find_class(module, name)

    class _PickleModule:
        Unpickler = _GEARUnpickler

    return torch.load(PYTORCH_CKPT, map_location="cpu", weights_only=False, pickle_module=_PickleModule)


def main():
    print(f"[LOAD] {PYTORCH_CKPT}")
    ckpt = load_ckpt()
    actor_sd = None
    for key in ("actor_model_state_dict", "policy_state_dict", "model_state_dict", "actor", "policy"):
        if isinstance(ckpt, dict) and key in ckpt:
            actor_sd = ckpt[key]
            break
    if actor_sd is None:
        print("[ERR] actor state_dict not found")
        sys.exit(1)

    print(f"[INFO] actor params: {len(actor_sd)}")
    std_candidates = [k for k in actor_sd if k.endswith("std") or "log_std" in k]
    print(f"[INFO] std-like keys: {std_candidates}")

    if "std" not in actor_sd:
        print("[ERR] 'std' key not found in actor_sd")
        sys.exit(1)

    std_tensor = actor_sd["std"]
    std_np = std_tensor.detach().cpu().numpy().astype(np.float32)
    print(f"[INFO] std shape={std_np.shape} dtype={std_np.dtype}")
    print(f"[INFO] std min={std_np.min():.4f} max={std_np.max():.4f} mean={std_np.mean():.4f}")
    print(f"[INFO] std values:\n{std_np}")

    if std_np.shape != (29,):
        print(f"[WARN] expected (29,), got {std_np.shape}")

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(OUT_PATH, std_np)
    print(f"[SAVED] {OUT_PATH}")


if __name__ == "__main__":
    main()
