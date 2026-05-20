from pxr import Usd


STAGE_PATH = r"D:\reboot\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\manager_based\locomanipulation\pick_place\warehouse.usd"


def main():
    stage = Usd.Stage.Open(STAGE_PATH)
    if stage is None:
        print("FAILED_TO_OPEN_STAGE")
        return

    root_layer = stage.GetRootLayer()
    print(f"STAGE={STAGE_PATH}")
    print(f"ROOT_LAYER={root_layer.realPath}")
    print(f"DEFAULT_PRIM={stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else 'NONE'}")
    print(f"SUBLAYERS={list(root_layer.subLayerPaths)}")

    found = []
    for prim in stage.Traverse():
        name = prim.GetName()
        path = str(prim.GetPath())
        if "Conveyor" in name or "Conveyor" in path or "A08_06" in name or "A08_06" in path:
            refs = prim.GetMetadata("references")
            payload = prim.GetMetadata("payload")
            found.append((path, name, refs, payload))

    print(f"FOUND_COUNT={len(found)}")
    for path, name, refs, payload in found[:100]:
        print(f"PRIM={path}")
        print(f"NAME={name}")
        print(f"REFERENCES={refs}")
        print(f"PAYLOAD={payload}")


if __name__ == "__main__":
    main()
