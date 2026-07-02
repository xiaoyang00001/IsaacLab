# GR00T Whole-Body Control Viewer

This folder contains a minimal Isaac Lab viewer for mirroring GR00T / MuJoCo /
SONIC G1 whole-body-control states.

## Launch From Isaac Lab

```bash
cd /home/nolovr/IsaacLab

TERM=xterm ./isaaclab.sh -p \
  scripts/gr00t_wbc/isaaclab_g1_sim2sim_viewer.py
```

The repository reference CSV files use Isaac Lab joint order, so the viewer
uses `--csv-joint-order isaaclab` by default and converts the first 29 columns
to MuJoCo / hardware order internally.

By default the script runs in Isaac Lab physical mode:

```text
physics_step=True
robot_collisions=True
robot_gravity=True
robot_self_collisions=False
drive_mode=state
```

This lets Isaac/PhysX contacts, gravity, and collision shapes participate in
physical tasks while preserving the original state-stream replay behavior.
Self-collision is still disabled by default because humanoid collision meshes
often overlap and can destabilize the robot; enable it only when the USD
collision filtering has been checked:

```bash
--robot-self-collisions
```

For a more physical Isaac-side task where joints are driven by Isaac implicit
PD actuators instead of directly teleporting the joint state, use:

```bash
--drive-mode position --no-follow-root --no-root-zmq
```

In this mode, incoming MuJoCo/control joint values are treated as joint position
targets. The floating base is left to PhysX when root following is disabled.
Tune `--actuator-stiffness`, `--actuator-damping`,
`--actuator-effort-limit`, and `--actuator-velocity-limit` for your task.

For a pure non-physical MuJoCo pose viewer, launch with:

```bash
--visual-mirror
```

That disables PhysX stepping, robot collisions, self-collisions, and robot
gravity so the displayed pose is not changed by Isaac contacts.

## Launch With Real-Time ZMQ State

```bash
cd /home/nolovr/IsaacLab

TERM=xterm ./isaaclab.sh -p \
  scripts/gr00t_wbc/isaaclab_g1_sim2sim_viewer.py \
  --source zmq \
  --zmq-host 127.0.0.1 \
  --zmq-port 5557 \
  --zmq-topic g1_debug \
  --zmq-pose-source measured \
  --root-motion-mode auto \
  --root-zmq-host 127.0.0.1 \
  --root-zmq-port 5558 \
  --root-zmq-topic g1_root \
  --zmq-warmup-sec 1.0
```

The default ZMQ launch above uses Isaac physics. If the robot looks different
from MuJoCo, that is expected: the scene is now using Isaac contact/gravity
rather than being a strict visual mirror.

The deploy ZMQ output uses MuJoCo / hardware order by default. The viewer reads
MuJoCo / hardware-order joint fields and maps them into the Isaac USD by joint
name.

By default the viewer mirrors measured control state:

```bash
--zmq-pose-source measured
```

This reads `body_q_measured` / `body_q` and `left_hand_q` / `right_hand_q`.
Use it when you want to inspect the actual MuJoCo/control feedback state.

To match the repository MuJoCo realtime-debug visualizer, which reads the
reference target fields, launch with:

```bash
--zmq-pose-source target
```

This reads `body_q_target`, `base_trans_target`, and `base_quat_target`.
Walking translation in Isaac Lab requires the ZMQ payload to contain a moving
root position, such as `root_pos_w`, `base_trans_target`, or
`base_trans_measured`. The current C++ `base_trans_measured` field is a fixed
default position, so measured mode can mirror the body pose but cannot move the
robot through world space unless the publisher is changed to send the MuJoCo
floating-base `qpos[0:3]`.

The MuJoCo Python bridge now also publishes the true floating-base pose on:

```text
tcp://*:5558/g1_root
```

The Isaac viewer subscribes to this stream by default in ZMQ mode. After
restarting the MuJoCo simulation, the viewer should log `root_zmq=fresh`, and
`root_pos_w` / `root_quat_w` from MuJoCo will drive the Isaac root directly.
The camera is fixed by default so manually selected viewport angles are not
reset. Use `--camera-follow` only when you explicitly want the script to keep
moving the camera behind the robot.

As a fallback when this root stream is not available, the viewer provides
support-foot root estimation:

```bash
--root-motion-mode stance
```

This keeps the detected support foot fixed in world space and dead-reckons the
root xy translation from the mirrored leg motion. The default `auto` mode uses
real MuJoCo/source root motion when it exists and falls back to support-foot
estimation when the source root xy is static. Use `--no-root-zmq` to disable the
extra MuJoCo root stream subscription, or `--root-motion-mode source` when you
want no support-foot estimation.

If the source root height is intentionally different from the Isaac USD model,
use:

```bash
--ground-lock-clearance 0.032
```

or disable the protection for debugging:

```bash
--no-ground-lock
```

## Repository Path

By default the script looks for:

```text
/home/nolovr/GR00T-WholeBodyControl
```

If the repository is moved, set:

```bash
export GR00T_WBC_ROOT=/path/to/GR00T-WholeBodyControl
```
