# SONIC deploy target 最小桥接方案

本文档说明 `sonic-deploy-target-minimal` 分支里的最小可用桥接形态。它把系统拆成两个边界清楚的部分：

- GR00T-WholeBodyControl 机器：负责 Pico 输入、SONIC encoder/decoder，以及 deploy 主循环。
- IsaacLab 机器：负责虚拟 G1 目标、物理仿真，以及可选的 Unitree DDS 仿真 topic。

目标是让 IsaacLab 对 GR00T deploy 看起来像下面两种对象之一：

- 直接消费 GR00T deploy 调试目标的 ZMQ subscriber；
- 通过 `rt/lowcmd` 和 `rt/lowstate` 通信的 Unitree G1 低层仿真器。

## 数据架构

默认直接 ZMQ 路径：

```text
Pico / pose input
  -> GR00T-WholeBodyControl deploy
  -> SONIC encoder + decoder
  -> ZMQ PUB g1_debug.last_action / body_q_target / base_quat_target
  -> IsaacLab SonicDeployTargetAction
  -> sonic_robot joint position targets
```

类硬件 DDS 路径：

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

原生 Windows 上的 IsaacLab 很难稳定安装和使用 CycloneDDS。遇到这种情况时，推荐使用代理路径：

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

在这个分支里，IsaacLab 的 DDS 路径不运行 SONIC encoder/decoder。IsaacLab 只接收最终低层命令，并发布仿真的状态。这种边界最接近未来真机部署：之后 GR00T 侧可以从 IsaacLab DDS 切到真实 Unitree DDS 网络，而不用改变 policy 边界。

## 分支

使用：

```powershell
git checkout sonic-deploy-target-minimal
```

该分支从 `gr00t-sonic-debug` 创建。

## IsaacLab 机器

在 IsaacLab Python 环境里安装运行依赖：

```powershell
.\isaaclab.bat -p -m pip install pyzmq msgpack
```

如果要使用 DDS 模式，还需要把 Unitree SDK2 Python 安装进同一个环境。GR00T 仓库已经 vendored 了该依赖：

```powershell
.\isaaclab.bat -p -m pip install -e D:\src\Isaac\GR00T-WholeBodyControl\external_dependencies\unitree_sdk2_python
```

如果 Windows 上安装时报错：

```text
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

说明 Python binding 没找到本机 CycloneDDS。当前双机 bring-up 阶段建议优先选择：

- 原生 Windows 上继续使用直接 ZMQ 模式；
- 在 GR00T/Linux 机器上运行 `scripts/tools/sonic_unitree_dds_proxy.py`，让 Windows IsaacLab 消费代理转出的 ZMQ；
- 在 Linux/WSL2 里运行 IsaacLab DDS bridge，因为 CycloneDDS 构建和网卡选择更稳定；
- 只有在安装 CycloneDDS C/C++ 并在 pip 安装 `unitree_sdk2_python` 前设置好 `CYCLONEDDS_HOME` 或 `CMAKE_PREFIX_PATH` 后，再尝试原生 Windows DDS。

直接 ZMQ 模式：

```powershell
$env:SONIC_DEPLOY_TRANSPORT="zmq"
$env:SONIC_DEPLOY_ENDPOINT="tcp://<GR00T_MACHINE_IP>:5557"
$env:SONIC_DEPLOY_TOPIC="g1_debug"
$env:SONIC_DEPLOY_TARGET_FIELD="last_action"
$env:SONIC_DEPLOY_TARGET_RATE_LIMIT="0.16"
$env:SONIC_DEPLOY_REFERENCE_TARGET_FIELD="body_q_target"
$env:SONIC_DEPLOY_BLEND_REFERENCE_LOWER_BODY="1"
$env:SONIC_DEPLOY_HOLD_LAST_REFERENCE="1"
$env:SONIC_DEPLOY_FOLLOW_BASE_YAW="1"
$env:SONIC_DEPLOY_FOLLOW_BASE_TRANSLATION="1"
$env:SONIC_DEPLOY_BASE_TRANSLATION_RATE_LIMIT="0.08"
$env:SONIC_DEPLOY_BASE_TRANSLATION_SCALE="2.0"
$env:SONIC_DEPLOY_SYNTHETIC_BASE_MOTION="1"
.\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0
```

DDS 模式：

```powershell
$env:SONIC_DEPLOY_TRANSPORT="dds"
$env:UNITREE_DDS_DOMAIN_ID="0"
$env:UNITREE_DDS_INTERFACE="<isaaclab_network_interface>"
.\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0
```

如果 CycloneDDS 能自动选择正确网卡，可以让 `UNITREE_DDS_INTERFACE` 为空。Windows 上 DDS loopback 和网卡选择比较脆弱；通常 WSL2 或 Linux 更适合 DDS。

## GR00T 机器

直接 ZMQ 模式会发布 IsaacLab 订阅的调试目标流：

```bash
cd /path/to/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh sim --input-type zmq_manager --zmq-host <pico_or_pose_source_host>
```

需要确认 deploy 的输出/调试 ZMQ 端口能被 IsaacLab 机器访问，并且与 `SONIC_DEPLOY_ENDPOINT` 一致。

当 GR00T 需要 DDS、但 IsaacLab 跑在原生 Windows 上时，推荐使用代理模式。代理应运行在 GR00T/Linux 机器上，并且所在环境需要安装 `unitree_sdk2py`、`pyzmq`、`msgpack`：

```bash
cd /path/to/IsaacLab
python scripts/tools/sonic_unitree_dds_proxy.py \
  --domain-id 0 \
  --interface <gr00t_dds_network_interface> \
  --zmq-bind tcp://*:5557 \
  --zmq-topic g1_debug
```

然后正常以 simulator 模式运行 GR00T deploy，这样 CRC 检查会被关闭：

```bash
cd /path/to/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh sim --input-type zmq_manager --zmq-host <pico_or_pose_source_host>
```

Windows IsaacLab 机器消费代理转出的 ZMQ 流：

```powershell
$env:SONIC_DEPLOY_TRANSPORT="zmq"
$env:SONIC_DEPLOY_ENDPOINT="tcp://<GR00T_MACHINE_IP>:5557"
$env:SONIC_DEPLOY_TOPIC="g1_debug"
$env:SONIC_DEPLOY_TARGET_FIELD="body_q_target"
.\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent.py --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0
```

纯 DDS 模式下，GR00T deploy 会把 IsaacLab 当成 DDS 网络上的仿真器：

```bash
cd /path/to/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh sim --input-type zmq_manager --zmq-host <pico_or_pose_source_host>
```

GR00T 和 IsaacLab 需要使用相同的 DDS domain 和网络接口。`sim` target 很重要，因为 GR00T deploy 会为 simulator 兼容性关闭 LowState CRC 检查。IsaacLab 当前发布结构正确的 `LowState`，但还没有计算 Unitree CRC。

## Topic 合同

IsaacLab DDS 模式：

- 使用 `unitree_hg.msg.dds_.LowCmd_` 订阅 `rt/lowcmd`；
- 使用 `unitree_hg.msg.dds_.LowState_` 发布 `rt/lowstate`；
- 使用 `unitree_hg.msg.dds_.IMUState_` 发布 `rt/secondary_imu`；
- 在 DDS 边界使用 G1 29DoF hardware/MuJoCo motor order；
- 写入 `sonic_robot` 前，内部 remap 到 IsaacLab/SONIC joint order。

常用环境变量：

```text
SONIC_DEPLOY_TRANSPORT=zmq|dds
SONIC_DEPLOY_ENDPOINT=tcp://<host>:5557
SONIC_DEPLOY_TOPIC=g1_debug
SONIC_DEPLOY_TARGET_FIELD=last_action
SONIC_DEPLOY_TARGET_RATE_LIMIT=0.16
SONIC_DEPLOY_REFERENCE_TARGET_FIELD=body_q_target
SONIC_DEPLOY_BLEND_REFERENCE_LOWER_BODY=1
SONIC_DEPLOY_HOLD_LAST_REFERENCE=1
SONIC_DEPLOY_FOLLOW_BASE_YAW=1
SONIC_DEPLOY_FOLLOW_BASE_TRANSLATION=1
SONIC_DEPLOY_BASE_QUAT_FIELD=base_quat_target
SONIC_DEPLOY_BASE_TRANS_FIELD=base_trans_target
SONIC_DEPLOY_BASE_YAW_RATE_LIMIT=0.12
SONIC_DEPLOY_BASE_TRANSLATION_RATE_LIMIT=0.08
SONIC_DEPLOY_BASE_TRANSLATION_SCALE=2.0
SONIC_DEPLOY_SYNTHETIC_BASE_MOTION=1
SONIC_DEPLOY_SYNTHETIC_BASE_MOTION_GAIN=0.35
SONIC_DEPLOY_SYNTHETIC_BASE_MOTION_DEADZONE=0.002
SONIC_DEPLOY_SYNTHETIC_BASE_MOTION_MAX_STEP=0.035
UNITREE_DDS_DOMAIN_ID=0
UNITREE_DDS_INTERFACE=<interface-name>
UNITREE_LOWCMD_TOPIC=rt/lowcmd
UNITREE_LOWSTATE_TOPIC=rt/lowstate
UNITREE_SECONDARY_IMU_TOPIC=rt/secondary_imu
UNITREE_G1_MODE_MACHINE=5
```

在固定 root 的 ZMQ 验证模式下，`base_trans_target` 如果没有持续变化，机器人只会原地摆腿。为了便于观察 whole-body target，IsaacLab 默认会在这种情况下根据下半身 reference 的变化合成一个小的可视 root 平移，并在日志里显示 `root_src=synthetic`。如果 deploy 已经提供有效位移，则日志会显示 `root_src=base_trans`，并优先使用 deploy 的 `base_trans_target`。

## 当前限制

- IsaacLab 还没有生成 DDS LowState CRC；GR00T deploy 需要运行在 `sim` 模式。
- IsaacLab 还没有桥接手部 DDS topic；当前最小目标先覆盖 body 29DoF。
- LowState acceleration 当前先填 0。joint position、joint velocity、estimated torque、base quaternion、base angular velocity 会从 IsaacLab 状态填入。
