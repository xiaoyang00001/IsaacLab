"""Offline pxr probe: print world-space bounding boxes of the pushcart and the
cardboard box USDs so the grasp script can stop guessing the geometry."""

import os

from pxr import Usd, UsdGeom

BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "source", "isaaclab_tasks", "isaaclab_tasks", "manager_based",
    "locomanipulation", "pick_place", "props",
)
BASE = os.path.abspath(BASE)


def probe(fname: str, depth: int = 2):
    path = os.path.join(BASE, fname)
    print("=" * 70)
    print(fname)
    print("=" * 70)
    stage = Usd.Stage.Open(path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide", "proxy"])
    root = stage.GetDefaultPrim() or stage.GetPseudoRoot()
    bbox = cache.ComputeWorldBound(root).ComputeAlignedRange()
    mn, mx = bbox.GetMin(), bbox.GetMax()
    print(f"TOTAL bbox min=({mn[0]:.3f},{mn[1]:.3f},{mn[2]:.3f}) max=({mx[0]:.3f},{mx[1]:.3f},{mx[2]:.3f})")
    print(f"      size=({mx[0]-mn[0]:.3f},{mx[1]-mn[1]:.3f},{mx[2]-mn[2]:.3f})")
    n = 0
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if not prim.IsA(UsdGeom.Boundable):
            continue
        b = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if b.IsEmpty():
            continue
        bn, bx = b.GetMin(), b.GetMax()
        print(f"  [{prim.GetTypeName():10s}] {p:60s} min=({bn[0]:+.3f},{bn[1]:+.3f},{bn[2]:+.3f}) max=({bx[0]:+.3f},{bx[1]:+.3f},{bx[2]:+.3f})")
        n += 1
        if n > 40:
            print("  ... (truncated)")
            break


probe("pushcart_physics.usda", depth=3)
probe("cart_box_d05_physics.usda", depth=2)
