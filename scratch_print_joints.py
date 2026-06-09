from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from pxr import Usd, UsdPhysics
import os

usd_path = r"D:\Omniverse\isaacsim_assets\Assets\Isaac\5.1\Isaac\Robots\Agibot\A2D\configuration\A2D_physics.usd"
out_path = r"D:\Omniverse\IsaacLab\scratch_bbox.txt"

def print_prim(prim, indent=0):
    apis = prim.GetAppliedSchemas()
    type_name = prim.GetTypeName()
    res = "  " * indent + f"{prim.GetName()} ({type_name}) - APIs: {apis}\n"
    for child in prim.GetChildren():
        res += print_prim(child, indent + 1)
    return res

with open(out_path, "w") as f:
    if os.path.exists(usd_path):
        stage = Usd.Stage.Open(usd_path)
        root_prim = stage.GetPrimAtPath("/A2D")
        if root_prim.IsValid():
            f.write(print_prim(root_prim))
        else:
            f.write("Error: /A2D is not valid\n")
    else:
        f.write("Error: USD path does not exist\n")

simulation_app.close()
