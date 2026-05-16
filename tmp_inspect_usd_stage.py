from __future__ import annotations

import argparse
from collections import defaultdict

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from pxr import Usd, UsdGeom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("usd_path")
    parser.add_argument("--contains", nargs="*", default=["Kitchen", "Table", "Coffee", "Sink", "Cabinet", "Shelf"])
    args = parser.parse_args()

    stage = Usd.Stage.Open(args.usd_path)
    if stage is None:
        raise RuntimeError(f"Failed to open stage: {args.usd_path}")

    default_prim = stage.GetDefaultPrim()
    print(f"defaultPrim={default_prim.GetPath() if default_prim else None}")
    print("rootPrims=", [prim.GetName() for prim in stage.GetPseudoRoot().GetChildren()])

    if default_prim:
        print("defaultPrimChildren=", [prim.GetName() for prim in default_prim.GetChildren()])

    xform_cache = UsdGeom.XformCache()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    matches: dict[str, list[str]] = defaultdict(list)
    for prim in stage.Traverse():
        name = prim.GetName()
        for token in args.contains:
            if token.lower() in name.lower():
                try:
                    bbox = bbox_cache.ComputeWorldBound(prim)
                    box = bbox.ComputeAlignedBox()
                    min_pt = tuple(round(float(v), 3) for v in box.GetMin())
                    max_pt = tuple(round(float(v), 3) for v in box.GetMax())
                except Exception:
                    min_pt = None
                    max_pt = None
                try:
                    world_tf = xform_cache.GetLocalToWorldTransform(prim)
                    translation = world_tf.ExtractTranslation()
                    pos = tuple(round(float(v), 3) for v in translation)
                except Exception:
                    pos = None
                matches[token].append(f"{prim.GetPath()} pos={pos} bbox_min={min_pt} bbox_max={max_pt}")
                break

    for token in args.contains:
        print(f"\n[{token}]")
        for line in matches[token][:40]:
            print(line)


if __name__ == "__main__":
    main()
    simulation_app.close()
