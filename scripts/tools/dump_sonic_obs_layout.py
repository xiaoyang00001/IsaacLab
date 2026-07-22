#!/usr/bin/env python3
"""Dump SONIC encoder/decoder obs layout from ckpt's UniversalTokenModule.

Goal: validate our hand-written _build_encoder_input matches ONNX export
      wrapper layout (tokenizer_obs_names order + flatten convention).

Output: prints
  - encoders list + encoder_input_features per encoder
  - tokenizer_obs_names (the canonical concatenation order)
  - tokenizer_obs_dims (shape per obs, used to reshape flat ↔ structured)
  - per-feature offset table for g1 encoder (so we can map to _build_encoder_input)

Usage:
    D:/miniconda3/envs/env_isaaclab/python.exe scripts/tools/dump_sonic_obs_layout.py
"""
import sys
import os
import types
import pickle
import importlib

GR00T_ROOT = "D:/src/Isaac/GR00T-WholeBodyControl"
sys.path.insert(0, GR00T_ROOT)

if "open3d" not in sys.modules:
    _o = types.ModuleType("open3d")
    _io = types.ModuleType("open3d.io")
    _io.read_triangle_mesh = lambda *a, **kw: None
    _io.write_triangle_mesh = lambda *a, **kw: None
    _o.io = _io
    sys.modules["open3d"] = _o
    sys.modules["open3d.io"] = _io

PYTORCH_CKPT = os.path.join(GR00T_ROOT, "sonic_release/last.pt")


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
        _MAP = (
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
            for src, dst in self._MAP:
                if module.startswith(src):
                    mapped = dst + module[len(src):]
                    try:
                        return super().find_class(mapped, name)
                    except (ImportError, AttributeError, ModuleNotFoundError):
                        break
            return super().find_class(module, name)

    class _PM:
        Unpickler = _GEARUnpickler

    return torch.load(PYTORCH_CKPT, map_location="cpu", weights_only=False, pickle_module=_PM)


def find_universal_token_module(obj, depth=0, path=""):
    """Recurse into ckpt to find a UniversalTokenModule instance."""
    if depth > 8:
        return None
    cls_name = type(obj).__name__
    if cls_name == "UniversalTokenModule":
        print(f"[FOUND] UniversalTokenModule at {path}")
        return obj

    if isinstance(obj, dict):
        for k, v in obj.items():
            r = find_universal_token_module(v, depth + 1, f"{path}[{k!r}]")
            if r is not None:
                return r

    if hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if k.startswith("_"):
                continue
            r = find_universal_token_module(v, depth + 1, f"{path}.{k}")
            if r is not None:
                return r

    return None


def main():
    print(f"[LOAD] {PYTORCH_CKPT}")
    ckpt = load_ckpt()
    print(f"[INFO] ckpt top keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__}")

    module = find_universal_token_module(ckpt, path="ckpt")
    if module is None:
        print("[ERR] UniversalTokenModule not found in ckpt")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("ENCODERS")
    print("=" * 70)
    enc_names = list(getattr(module, "encoders_to_iterate", []))
    print(f"  encoders_to_iterate: {enc_names}")
    enc_features = getattr(module, "encoder_input_features", {})
    for n in enc_names:
        print(f"  encoder_input_features[{n!r}]: {enc_features.get(n, '?')}")

    print("\n" + "=" * 70)
    print("DECODERS")
    print("=" * 70)
    dec_features = getattr(module, "decoder_input_features", {})
    for k, v in dec_features.items():
        print(f"  decoder_input_features[{k!r}]: {v}")

    print("\n" + "=" * 70)
    print("TOKENIZER OBS")
    print("=" * 70)
    obs_names = list(getattr(module, "tokenizer_obs_names", []))
    obs_dims = dict(getattr(module, "tokenizer_obs_dims", {}))
    print(f"  tokenizer_obs_names: {obs_names}")
    print(f"  tokenizer_obs_dims:")
    for n in obs_names:
        print(f"    {n}: {obs_dims.get(n, '?')}")

    print("\n" + "=" * 70)
    print("ONNX ENCODER INPUT LAYOUT (replicates inference_helpers.py logic)")
    print("=" * 70)
    print("  [0]            encoder_index (scalar)")
    offset = 1
    g1_features = set(enc_features.get("g1", []))
    special = {"token", "token_flattened", "proprioception", "action", "meta_action"}
    print(f"  g1 encoder_input_features: {g1_features}")
    print()
    print(f"  {'offset':>9}  {'name':30s}  {'dims':20s}  {'size':>6}  {'used_by_g1':10s}")
    print(f"  {'-'*9}  {'-'*30}  {'-'*20}  {'-'*6}  {'-'*10}")
    for n in obs_names:
        if n in special:
            continue
        if n not in obs_dims:
            continue
        dims = obs_dims[n]
        # torch.prod over tuple
        size = 1
        for d in dims:
            size *= int(d)
        used = "YES" if n in g1_features else "no"
        print(f"  {offset:>9d}  {n:30s}  {str(tuple(dims)):20s}  {size:>6d}  {used:10s}")
        offset += size
    print()
    print(f"  Total (incl. encoder_index): {offset}")

    print("\n" + "=" * 70)
    print("DECODER INPUT LAYOUT")
    print("=" * 70)
    g1_dyn = "g1_dyn"
    if g1_dyn in dec_features:
        print(f"  decoder '{g1_dyn}' input_features: {dec_features[g1_dyn]}")

    # Try to also dump decoder feature dim per name (might be different attr)
    proprio_dim = getattr(module, "proprioception_dim", None)
    token_dim = getattr(module, "token_dim", None)
    print(f"  proprioception_dim: {proprio_dim}")
    print(f"  token_dim: {token_dim}")


if __name__ == "__main__":
    main()
