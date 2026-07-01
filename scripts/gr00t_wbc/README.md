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

The viewer is a state mirror, not a PhysX control simulation. Robot gravity is
disabled by default and `--ground-lock` is enabled so that the feet do not drift
below the visual ground after the mirrored pose is written.

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
  --zmq-warmup-sec 1.0
```

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
