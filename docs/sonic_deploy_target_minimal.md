# SONIC deploy target minimal bridge

This branch keeps the minimum useful split between the two machines:

- GR00T-WholeBodyControl machine: Pico input, SONIC encoder/decoder/deploy loop.
- IsaacLab machine: virtual G1 target, physics, and optional Unitree DDS sim topics.

The goal is to make IsaacLab look like either:

- a direct ZMQ consumer of GR00T deploy debug targets; or
- a Unitree G1 low-level simulator that talks over `rt/lowcmd` and `rt/lowstate`.

## Data architecture

Default direct path:

```text
Pico / pose input
  -> GR00T-WholeBodyControl deploy
  -> SONIC encoder + decoder
  -> ZMQ PUB g1_debug.body_q_target, MuJoCo order, 29 DoF
  -> IsaacLab SonicDeployTargetAction
  -> sonic_robot joint position targets
```

DDS hardware-like path:

```text
Pico / pose input
  -> GR00T-WholeBodyControl deploy
  -> SONIC encoder + decoder
  -> Unitree DDS PUB rt/lowcmd
  -> IsaacLab UnitreeDdsLowCmdAction
  -> sonic_robot joint position targets
  -> IsaacLab publishes rt/lowstate + rt/secondary_imu
  -> GR00T deploy reads lowstate like a real/sim G1
```

Native Windows IsaacLab cannot reliably install/use CycloneDDS. For that case,
use the proxy path instead:

```text
Pico / pose input
  -> GR00T-WholeBodyControl deploy
  -> SONIC encoder + decoder
  -> GR00T/Linux DDS rt/lowcmd
  -> scripts/tools/sonic_unitree_dds_proxy.py
       publishes fake rt/lowstate + rt/secondary_imu back to GR00T
       forwards LowCmd.q as ZMQ g1_debug.body_q_target
  -> Windows IsaacLab SonicDeployTargetAction
  -> sonic_robot joint position targets
```

In this branch, IsaacLab does not run SONIC encoder/decoder in the DDS path. It only receives the final low-level command and publishes simulated state. That is the closest deployment shape to the future physical robot, because the GR00T side can later switch from IsaacLab DDS to the real Unitree DDS network without changing the policy boundary.

## Branch

Use:

```powershell
git checkout sonic-deploy-target-minimal
```

This branch was created from `gr00t-sonic-debug`.

## IsaacLab machine

Install runtime dependencies in the IsaacLab Python environment:

```powershell
.\isaaclab.bat -p -m pip install pyzmq msgpack
```

For DDS mode, also install Unitree SDK2 Python into the same environment. The GR00T repo already vendors it:

```powershell
.\isaaclab.bat -p -m pip install -e D:\src\Isaac\GR00T-WholeBodyControl\external_dependencies\unitree_sdk2_python
```

If this fails on Windows with:

```text
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

then the Python binding found no native CycloneDDS install. For the current two-machine bring-up, prefer one of these:

- stay on direct ZMQ mode on native Windows;
- run `scripts/tools/sonic_unitree_dds_proxy.py` on the GR00T/Linux machine and let Windows IsaacLab consume its ZMQ output;
- run the IsaacLab DDS bridge from Linux/WSL2 where CycloneDDS builds and selects network interfaces reliably;
- only use native Windows DDS after installing CycloneDDS C/C++ and exporting `CYCLONEDDS_HOME` or `CMAKE_PREFIX_PATH` before pip installing `unitree_sdk2_python`.

Direct ZMQ mode:

```powershell
$env:SONIC_DEPLOY_TRANSPORT="zmq"
$env:SONIC_DEPLOY_ENDPOINT="tcp://<GR00T_MACHINE_IP>:5557"
$env:SONIC_DEPLOY_TOPIC="g1_debug"
.\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0
```

DDS mode:

```powershell
$env:SONIC_DEPLOY_TRANSPORT="dds"
$env:UNITREE_DDS_DOMAIN_ID="0"
$env:UNITREE_DDS_INTERFACE="<isaaclab_network_interface>"
.\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0
```

Leave `UNITREE_DDS_INTERFACE` empty if CycloneDDS auto-selects the right interface. On Windows, DDS loopback/interface selection can be fragile; WSL2 or Linux is usually easier for DDS.

## GR00T machine

Direct ZMQ mode publishes the debug target stream that IsaacLab subscribes to:

```bash
cd /path/to/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh sim --input-type zmq_manager --zmq-host <pico_or_pose_source_host>
```

Make sure the deploy output/debug ZMQ port is reachable by the IsaacLab machine and matches `SONIC_DEPLOY_ENDPOINT`.

Proxy mode is the recommended path when GR00T requires DDS but IsaacLab is on native Windows. Run this proxy on the GR00T/Linux machine in an environment where `unitree_sdk2py`, `pyzmq`, and `msgpack` are installed:

```bash
cd /path/to/IsaacLab
python scripts/tools/sonic_unitree_dds_proxy.py \
  --domain-id 0 \
  --interface <gr00t_dds_network_interface> \
  --zmq-bind tcp://*:5557 \
  --zmq-topic g1_debug
```

Then run GR00T deploy normally in simulator mode so CRC checks are disabled:

```bash
cd /path/to/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh sim --input-type zmq_manager --zmq-host <pico_or_pose_source_host>
```

On the Windows IsaacLab machine, consume the proxy ZMQ stream:

```powershell
$env:SONIC_DEPLOY_TRANSPORT="zmq"
$env:SONIC_DEPLOY_ENDPOINT="tcp://<GR00T_MACHINE_IP>:5557"
$env:SONIC_DEPLOY_TOPIC="g1_debug"
.\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0
```

Pure DDS mode runs GR00T deploy as if IsaacLab itself were a simulator on the DDS network:

```bash
cd /path/to/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh sim --input-type zmq_manager --zmq-host <pico_or_pose_source_host>
```

Use the same DDS domain and network interface as IsaacLab. The `sim` target is important because GR00T deploy disables LowState CRC checking for simulator compatibility. IsaacLab currently publishes structurally correct `LowState`, but does not compute Unitree CRC yet.

## Topic contract

IsaacLab DDS mode:

- subscribes `rt/lowcmd` using `unitree_hg.msg.dds_.LowCmd_`;
- publishes `rt/lowstate` using `unitree_hg.msg.dds_.LowState_`;
- publishes `rt/secondary_imu` using `unitree_hg.msg.dds_.IMUState_`;
- uses G1 29-DoF hardware/MuJoCo motor order at the DDS boundary;
- remaps internally to IsaacLab/SONIC joint order before writing `sonic_robot`.

Useful environment variables:

```text
SONIC_DEPLOY_TRANSPORT=zmq|dds
SONIC_DEPLOY_ENDPOINT=tcp://<host>:5557
SONIC_DEPLOY_TOPIC=g1_debug
UNITREE_DDS_DOMAIN_ID=0
UNITREE_DDS_INTERFACE=<interface-name>
UNITREE_LOWCMD_TOPIC=rt/lowcmd
UNITREE_LOWSTATE_TOPIC=rt/lowstate
UNITREE_SECONDARY_IMU_TOPIC=rt/secondary_imu
UNITREE_G1_MODE_MACHINE=5
```

## Current limitations

- DDS LowState CRC is not generated in IsaacLab yet; run GR00T deploy in `sim` mode.
- Hand DDS topics are not bridged in IsaacLab yet; body 29-DoF is the first minimum target.
- LowState acceleration is zero-filled for now. Joint position, joint velocity, estimated torque, base quaternion, and base angular velocity are populated from IsaacLab.
