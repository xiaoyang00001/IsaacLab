from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import isaaclab_tasks
import isaaclab_tasks.manager_based.manipulation.place.config.agibot.place_toy2box_rmp_rel_env_cfg as cfg_mod

with open("scratch_import_debug.txt", "w") as f:
    f.write(f"isaaclab_tasks path: {isaaclab_tasks.__file__}\n")
    f.write(f"cfg_mod path: {cfg_mod.__file__}\n")

simulation_app.close()
