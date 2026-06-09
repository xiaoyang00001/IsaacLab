from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from isaaclab.sim.utils import get_current_stage
from pxr import UsdPhysics, PhysxSchema, UsdGeom
import os

from isaaclab_tasks.manager_based.manipulation.place.config.agibot.place_toy2box_rmp_rel_env_cfg import spawn_agibot_floating, RmpFlowAgibotPlaceToy2BoxEnvCfg

stage = get_current_stage()
world = UsdGeom.Xform.Define(stage, "/World")

cfg = RmpFlowAgibotPlaceToy2BoxEnvCfg()
cfg.scene.robot.spawn.func("/World/Robot", cfg.scene.robot.spawn)

out_file = r"D:\Omniverse\IsaacLab\scratch_spawn_debug.txt"
with open(out_file, "w") as f:
    f.write("Checking prims after spawning:\n")
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if "Robot" in path:
            has_art = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
            has_physx_art = prim.HasAPI(PhysxSchema.PhysxArticulationAPI)
            if has_art or has_physx_art or prim.GetName() in ["Robot", "root_joint", "base_link"]:
                f.write(f"Prim: {path}, Type: {prim.GetTypeName()}, Has ArticulationRootAPI: {has_art}, Has PhysxArticulationAPI: {has_physx_art}\n")

simulation_app.close()
