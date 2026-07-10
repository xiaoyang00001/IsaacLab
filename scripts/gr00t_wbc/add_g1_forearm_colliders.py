# -*- coding: utf-8 -*-
"""Add capsule colliders to G1 forearm/wrist links that ship without collision.

The stock ``g1_43dof.usd`` in GR00T-WholeBodyControl has no collision shape on
``left/right_elbow_link`` (the forearm body), ``left/right_wrist_roll_link`` and
``left/right_wrist_yaw_link``. Arm-hug manipulation therefore passes straight
through objects regardless of how the joints are driven. This script authors a
``collisions`` Capsule prim on each of those links, following the same
convention (purpose/visibility) as the existing shoulder_yaw/wrist_pitch
colliders. Capsule sizes are derived from the links' visual-mesh bounds.

The original file is backed up next to it before saving. Run this once per
machine that hosts a GR00T-WholeBodyControl checkout:

    python scripts/gr00t_wbc/add_g1_forearm_colliders.py

Requires the ``usd-core`` package (``pip install usd-core``) or any Python
environment that provides ``pxr``.
"""

import os
import shutil
import sys
from datetime import date
from pathlib import Path

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    sys.exit("pxr module not found. Install with: pip install usd-core")

ROBOT_ROOT_PRIM = "/g1_29dof_with_hand_rev_1_0"

# link name -> (radius, height, axis, translate) — sized from visual-mesh bounds
CAPSULES = {
    "left_elbow_link": (0.030, 0.075, "X", (0.035, 0.0, -0.008)),
    "right_elbow_link": (0.030, 0.075, "X", (0.035, 0.0, -0.008)),
    "left_wrist_roll_link": (0.028, 0.010, "X", (0.028, 0.0, 0.0)),
    "right_wrist_roll_link": (0.028, 0.010, "X", (0.028, 0.0, 0.0)),
    "left_wrist_yaw_link": (0.028, 0.005, "X", (0.012, 0.0, 0.0)),
    "right_wrist_yaw_link": (0.028, 0.005, "X", (0.012, 0.0, 0.0)),
}


def _resolve_usd_path() -> Path:
    candidates = []
    if "GR00T_WBC_ROOT" in os.environ:
        candidates.append(Path(os.environ["GR00T_WBC_ROOT"]).expanduser())
    candidates.extend(
        [
            Path("D:/src/Isaac/GR00T-WholeBodyControl"),
            Path("F:/ISAACWholeBody/GR00T-WholeBodyControl"),
            Path.cwd() / "GR00T-WholeBodyControl",
        ]
    )
    for root in candidates:
        usd_path = root / "gear_sonic/data/robots/g1/g1_43dof.usd"
        if usd_path.is_file():
            return usd_path
    sys.exit("g1_43dof.usd not found. Set GR00T_WBC_ROOT to the repo root.")


def main() -> None:
    usd_path = _resolve_usd_path()
    backup_path = usd_path.with_suffix(f".usd.bak_no_forearm_collision_{date.today():%Y%m%d}")

    stage = Usd.Stage.Open(str(usd_path))

    missing = [
        link for link in CAPSULES if not stage.GetPrimAtPath(f"{ROBOT_ROOT_PRIM}/{link}/collisions")
    ]
    if not missing:
        print("All forearm/wrist colliders already present; nothing to do.")
        return

    if not backup_path.exists():
        shutil.copy2(usd_path, backup_path)
        print(f"backup -> {backup_path}")

    ref = stage.GetPrimAtPath(f"{ROBOT_ROOT_PRIM}/left_shoulder_yaw_link/collisions")
    if not ref:
        sys.exit(f"reference collider not found under {ROBOT_ROOT_PRIM}/left_shoulder_yaw_link")
    ref_img = UsdGeom.Imageable(ref)
    ref_purpose = ref_img.GetPurposeAttr().Get()
    ref_visibility = ref_img.GetVisibilityAttr().Get()

    for link in missing:
        radius, height, axis, translate = CAPSULES[link]
        if not stage.GetPrimAtPath(f"{ROBOT_ROOT_PRIM}/{link}"):
            sys.exit(f"link not found: {link}")
        capsule = UsdGeom.Capsule.Define(stage, f"{ROBOT_ROOT_PRIM}/{link}/collisions")
        capsule.CreateRadiusAttr(radius)
        capsule.CreateHeightAttr(height)
        capsule.CreateAxisAttr(axis)
        capsule.AddTranslateOp().Set(Gf.Vec3d(*translate))
        img = UsdGeom.Imageable(capsule.GetPrim())
        img.GetPurposeAttr().Set(ref_purpose)
        img.GetVisibilityAttr().Set(ref_visibility)
        UsdPhysics.CollisionAPI.Apply(capsule.GetPrim())
        print(f"ADDED {link}/collisions: r={radius} h={height} axis={axis} t={translate}")

    stage.GetRootLayer().Save()
    print("saved:", usd_path)


if __name__ == "__main__":
    main()
