# G1 XR 全身遥操作实时镜像框架

> 分支：`xiaoyang0704`
> 对应启动脚本：[scripts/environments/teleoperation/teleop_se3_agent.py](../scripts/environments/teleoperation/teleop_se3_agent.py)
> 对应任务：`Isaac-PickPlace-Locomanipulation-G1-Abs-v0`
> 环境配置：[locomanipulation_g1_env_cfg.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py)
> 动作实现：[mdp/actions.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/mdp/actions.py) 中的 `MuJoCoG1MirrorAction`

## 1. 这条链路在做什么

这不是一个让 Isaac Lab 自己用 IK / RL 策略去控制 G1 走路和抓取的任务。它的实际分工是：

- **远端主机**（Ubuntu，`192.168.10.230`）跑 `GR00T-WholeBodyControl`（SONIC + MuJoCo）全身控制器，实时把机器人身体关节角、根节点位姿通过 UDP 广播出来。
- **本机 Isaac Lab**（Windows，`192.168.10.46`）只订阅这些 UDP 包，把收到的关节角 / 根节点位姿**直接写入**仿真中 G1 的关节状态和刚体位姿——相当于一个实时镜像播放器，不做任何解算。
- **VR 头显 + 手柄**挂在 Isaac Lab 这一侧，提供操作员的第一人称视角（锚定在 G1 头部），并且**只**负责左右手夹爪的开合意图，不参与身体姿态控制。

简言之：**全身姿态由远端全身控制器决定，Isaac Lab 负责镜像显示 + 提供 XR 第一人称视角 + 采集夹爪开合指令**，两路控制权在关节层面是分开写入、互不覆盖的（见第 6 节）。

## 2. 启动命令

```powershell
$env:GR00T_WBC_ROOT="F:\ISAACWholeBody\GR00T-WholeBodyControl"
$env:ISAACLAB_G1_ZMQ_HOST="192.168.10.230"
$env:ISAACLAB_G1_ROOT_ZMQ_HOST="192.168.10.230"

.\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent.py `
  --xr `
  --device cuda:0 `
  --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 `
  --teleop_device motion_controllers `
  --enable_pinocchio
```

`teleop_se3_agent.py` 针对这个任务名做了几处特殊处理（见文件开头 `G1_LOCOMANIP_TASK_ID` 相关逻辑）：

| 命令行参数 | 实际效果 |
|---|---|
| `--task Isaac-PickPlace-Locomanipulation-G1-Abs-v0` | 也接受不带 `-v0` 的别名 `Isaac-PickPlace-Locomanipulation-G1-Abs`，会自动补全 |
| `--teleop_device motion_controllers` | 对该任务这其实是默认值：只要没显式传 `--teleop_device`，脚本会自动把默认的 `keyboard` 改成 `motion_controllers` |
| `--enable_pinocchio` | 对该任务会被脚本强制设为 `True`，即使不传这个参数也一样（Pink IK / GR1T2 retargeter 依赖 Pinocchio） |
| `--xr` | 只要 `--teleop_device` 是 `handtracking` 或 `motion_controllers`，脚本会自动把 `xr` 打开，因此这里显式传 `--xr` 是冗余但无害的写法 |

也就是说，最短的等价命令其实是：

```powershell
.\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent.py --device cuda:0 --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0
```

保留显式参数只是为了可读性，不是必需的。

## 3. 网络拓扑

```
┌─────────────────────────────┐         UDP 5557 (topic: g1_debug)          ┌──────────────────────────────┐
│ Ubuntu 主机 192.168.10.230   │  ───────────body 关节状态──────────────▶   │ Windows Isaac Lab 192.168.10.46│
│ GR00T-WholeBodyControl       │                                              │ (bind 0.0.0.0:5557 / 0.0.0.0:5558)│
│ (SONIC 全身控制 + MuJoCo)     │         UDP 5558 (topic: g1_root)           │ MuJoCoG1MirrorAction           │
└─────────────────────────────┘  ───────────根节点位姿───────────────────▶   └──────────────────────────────┘
```

默认网络参数集中写在 [scripts/gr00t_wbc/g1_udp_network.env](../scripts/gr00t_wbc/g1_udp_network.env)，`locomanipulation_g1_env_cfg.py` 里的 `_load_default_network_config()` 会在模块导入时自动读取这个文件并用 `os.environ.setdefault(...)` 灌入环境变量——**显式设置的环境变量优先级更高**，不会被这个文件覆盖。

## 4. 环境变量清单

| 变量 | 默认值 | 作用 |
|---|---|---|
| `GR00T_WBC_ROOT` | 依次尝试 `F:/ISAACWholeBody/GR00T-WholeBodyControl`、仓库上级目录、当前工作目录 | 用来定位 G1 43-DoF USD 资产（`gear_sonic/data/robots/g1/*.usd`），找不到会抛 `FileNotFoundError` |
| `ISAACLAB_G1_TRANSPORT` | `udp` | 传输方式，`udp` 或 `zmq` |
| `ISAACLAB_G1_ZMQ_HOST` / `ISAACLAB_G1_ROOT_ZMQ_HOST` | `192.168.10.230` | **仅当 `ISAACLAB_G1_TRANSPORT=zmq` 时才生效**（见下方说明） |
| `ISAACLAB_G1_UDP_BIND_HOST` / `PORT` / `TOPIC` / `RCVBUF` | `0.0.0.0` / `5557` / `g1_debug` / `262144` | body 关节状态 UDP 订阅参数 |
| `ISAACLAB_G1_ROOT_UDP_BIND_HOST` / `PORT` / `TOPIC` / `RCVBUF` | `0.0.0.0` / `5558` / `g1_root` / `262144` | 根节点位姿 UDP 订阅参数 |
| `ISAACLAB_G1_NETWORK_CONFIG` / `G1_NETWORK_CONFIG` | 未设置时用仓库内 `g1_udp_network.env` | 可指向自定义的网络配置文件 |

**需要特别注意的坑**：当前默认 `transport=udp`，而 `MuJoCoG1MirrorAction`（[actions.py:307-322](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/mdp/actions.py#L307-L322)）在 UDP 模式下只是在本地 `bind(udp_bind_host, udp_port)` 监听、**不会向任何“host”发起连接**——谁往 `0.0.0.0:5557` / `0.0.0.0:5558` 发包，Isaac Lab 就收谁的。所以启动命令里设置的 `ISAACLAB_G1_ZMQ_HOST` / `ISAACLAB_G1_ROOT_ZMQ_HOST=192.168.10.230` **在当前 UDP 传输下实际不起作用**，只有把 `ISAACLAB_G1_TRANSPORT` 显式改成 `zmq` 时（ZMQ SUB socket 会 `connect()` 到 `tcp://<host>:<port>`）这两个变量才会生效。保留这两个变量目前更多是为将来切回 ZMQ 传输做准备，或作为链路来源的文档说明。

真正决定“谁能把包发给 Isaac Lab”的是防火墙放行 + Ubuntu 侧发送目标是否填对了 `192.168.10.46:5557/5558`（对应 `g1_udp_network.env` 里的 `G1_UDP_OUT_HOST` / `G1_ROOT_UDP_HOST`，这两个是发送端配置，Isaac Lab 侧不使用）。

## 5. 机器人与场景资产

- 机器人使用 `G1_43DOF_GR00T_CFG`（[locomanipulation_g1_env_cfg.py:100](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py#L100)），USD 路径由 `_find_gr00t_g1_43dof_usd()` 在 `GR00T_WBC_ROOT/gear_sonic/data/robots/g1/` 下按以下候选文件名顺序查找：`g1_43dof_isaaclab_nomdl.usd` → `g1_43dof.usd` → `g1_43dof_isaaclab_no_material.usda` → `g1_43dof_isaaclab_nomdl.usda` → `g1_43dof_s3.usda`。
- 场景（`LocomanipulationG1SceneCfg`）里还定义了 `packing_table` 和 `object`（待抓取蓝色方块），但当前初始位置 `z` 分别为 `-1000.66` / `-100.76`，明显在地面以下，实际不会出现在场景中——如果之后要恢复拾取任务，需要先把这两个 z 坐标改回合理数值。
- `num_envs` 固定为 `1`：`MuJoCoG1MirrorAction` 的 `__init__` 里有 `self._enabled = cfg.enabled and self.num_envs == 1`，多环境会静默关闭镜像功能（只打印一条 `[WARN]`），这个任务的设计前提就是单机单人 XR 第一人称。

### 5.1 可选场景：SonicSolo / SonicFullscene（2026-07-07 从 sonic-windows-xr-ar-anchor 移植）

两个新任务 id，把 `--task` 换掉即可使用，其余启动参数与主任务完全相同（动作/观测/XR/teleop 均继承主配置 `LocomanipulationG1EnvCfg`，只换场景）：

| 任务 id | 场景内容 |
|---|---|
| `Isaac-SonicSolo-Locomanipulation-G1-v0` | 主场景 + 抱取演示物体（HugBox 台座 + 纸箱），适合轻量调试 |
| `Isaac-SonicFullscene-Locomanipulation-G1-v0` | 主场景 + warehouse.usd 仓库背景 + USD 打包台 + 转向盘可抓道具 + HugBox |

移植原则是**只要 USD 场景资产**，源分支的 SONIC 机器人配置和 events 事件系统（HugBox 物理补齐 / 传送带滚轮驱动 / 地板摩擦补绑 / viewer 对齐）均未移植，因此：

- 两个场景里机器人出生点被移到源分支的仓库通道坐标 `(-2.0, 11.008, 0.78)`（`ROBOT_SPAWN_POS`，[sonic_solo_locomanipulation_env_cfg.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/sonic_solo_locomanipulation_env_cfg.py)），warehouse/HugBox 等道具坐标与源分支原样共用；`root_position_mode="relative"` 只镜像位移增量，出生点任意可行；
- HugBox 在出生点正前方 **+X** 1.05m（源分支布局假定机器人朝向 +X；根朝向由 UDP 源绝对拷贝，若实际朝向不同调整 `_DEMO_OBJECT_POS` / `_DEMO_STAND_POS`）；纸箱用 USD 默认质量（源分支靠 prestartup 事件调轻到 0.8kg，未移植）；
- fullscene 的传送带只有视觉模型（warehouse.usd 自带），不会转动、无流水箱子；warehouse 地板碰撞体保持默认摩擦 μ=0.5（镜像路线根姿态直写，地面摩擦不影响效果）。

## 6. 控制权划分：身体镜像 vs. 手柄夹爪

`ActionsCfg` 里只有一个动作项 `mujoco_g1_mirror`（[action_cfg.py:37](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/configs/action_cfg.py#L37)），代码注释也特别强调：**这个任务不要再加 IK 或 locomotion 动作项，否则会覆盖镜像状态**。它同时做两件事：

1. **身体姿态镜像**（来自远端 UDP）：`mirror_joint_names` 覆盖髋/膝/踝/腰/肩/肘/腕关节，每个仿真步把收到的 29-DoF MuJoCo 关节角直接 `write_joint_state_to_sim` + 设为位置/速度目标；根节点位姿通过独立的 `g1_root` 流镜像（`root_motion_mode="source"`，不启用支撑脚兜底估计）。
   - `root_position_mode="relative"`：第一次收到根节点包时记录远端起始位置和 Isaac 本地起始位置，之后世界位置 = **本地起始位置 + (远端当前位置 − 远端起始位置)**，即只镜像位移增量而不是绝对坐标，避免两侧坐标系原点不一致导致机器人瞬移。
   - `root_zmq_required=True`：如果收到了 body 包但还没收到 root 包，会打印一次性 `[WARN] ... the robot will walk in place until root_pos_w/root_quat_w arrive.`——机器人原地摆动但不移动，通常就是这个原因。
2. **手部/夹爪由 VR 手柄控制，不走 MuJoCo 镜像**：`mirror_hands_from_mujoco = cfg.mirror_hands and not cfg.controller_gripper_enabled`（[actions.py:405](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/mdp/actions.py#L405)）。当前配置里 `controller_gripper_enabled` 默认为 `True`，所以这个表达式恒为 `False`——手指关节永远不会被 MuJoCo 的手部数据覆盖，全部由 `G1GripperMotionControllerRetargeter` 从 VR 手柄扳机 / 侧握实时生成（action 维度为 4：`[left_index, left_middle, right_index, right_middle]`）。

夹爪相关参数（对应「解决夹爪不能完全闭合问题」提交 `ec1732ddc`）：

| 参数 | 值 | 说明 |
|---|---|---|
| `controller_gripper_finger_close_angle` | `1.8` | 食指/中指全握角度上限（弧度） |
| `controller_gripper_thumb_1_angle` / `_thumb_2_angle` | `1.1` / `1.8` | 拇指两个关节的全握角度上限 |
| `controller_gripper_action_alpha` | `1.0` | 低通平滑系数，`1.0` 表示不做平滑，手柄输入直接生效 |
| `controller_gripper_use_soft_limits` | `False` | 用硬关节限位而不是软限位夹紧目标角，避免限位过紧夹不拢 |
| `controller_gripper_write_joint_state` | `True` | 除了下发位置目标外，还直接写关节状态，避免高刚度 PD 追不上目标导致「看起来没完全闭合」 |

## 7. XR 第一人称视角与朝向校准

`self.xr`（`XrCfg`）配置：

- `anchor_prim_path = ".../Robot/head_link"`：头显视角锚定在 G1 头部。
- `anchor_rotation_prim_path = ".../Robot/pelvis"`：朝向的参考点用骨盆而不是头部，避免头显自身旋转带偏视角朝向。
- `anchor_rotation_mode = FOLLOW_PRIM_SMOOTHED`：朝向用 slerp 平滑跟随骨盆偏航角，而不是硬跳。
- `fixed_anchor_height = False`：锚点高度跟随机器人实际高度变化（蹲下/站起时视角会跟着变化）。
- `recenter_yaw_button = ("/user/hand/right", "b")`，`recenter_yaw_button_event = "release"`：松开右手柄 **B 键**时，把当前头显朝向重新和骨盆朝向对齐，用来消除长时间 VR 使用后累积的偏航漂移（对应提交 `f3fa7ecea` 「实现 RecenterYaw」）。
- `recenter_anchor_forward_axis=(-1,0,0)`、`recenter_headset_forward_axis=(0,-1,0)`、`recenter_headset_fallback_axis=(1,0,0)`：分别是 G1 USD 骨盆本地系和头显本地系里代表「前方」的轴，这是针对当前 G1 模型和头显标定出来的值——换模型或换头显设备需要重新标定。

**注意**：正因为右手 **B 键**被占用做视角回正，`G1GripperMotionControllerRetargeterCfg` 在本任务里显式传了 `use_right_b_button=False`（[locomanipulation_g1_env_cfg.py:512](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py#L512)），否则默认库行为下右 B 键会同时触发「中指闭合」和「视角回正」两个动作冲突。左手 X 键固定绑定环境 reset，同样不参与夹爪控制。

## 8. teleop_devices 配置

```python
self.teleop_devices = DevicesCfg(
    devices={
        "motion_controllers": OpenXRDeviceCfg(
            retargeters=[
                G1GripperMotionControllerRetargeterCfg(
                    sim_device="cpu",
                    use_right_b_button=False,
                ),
            ],
            sim_device="cpu",
            xr_cfg=self.xr,
        ),
    }
)
```

对应命令行的 `--teleop_device motion_controllers`，与 `teleop_se3_agent.py` 里 `create_teleop_device` 的查找逻辑对应（`env_cfg.teleop_devices.devices["motion_controllers"]`）。

## 9. 最近相关提交（便于排障时对照代码演进）

| 提交 | 内容 |
|---|---|
| `f3fa7ecea` 实现RecneterYaw | 新增 `recenter_yaw_button` 系列配置和 `xr_anchor_utils.py` 的朝向回正逻辑 |
| `ec1732ddc` 提交解决夹爪不能完全闭合问题 | 调大夹爪闭合角度、新增 `controller_gripper_write_joint_state` 等直写关节状态的开关 |
| `c2cd0dbc1` 提交UDP | 大幅重写 `scripts/gr00t_wbc/isaaclab_g1_sim2sim_viewer.py` 的 UDP 支持 |
| `8c56e85d8` 提交 | 新增 `g1_udp_network.env`，把 UDP 设为默认传输并让主机/端口可通过环境变量配置 |

## 10. 排障 Checklist

- **机器人原地不动/不走动**：看终端日志 `[INFO] MuJoCo G1 root mirror: ...`，如果 `root_source=none`，说明还没收到 `g1_root` 流；确认 Ubuntu 端 SONIC/MuJoCo 已经在往 `192.168.10.46:5558` 发送 UDP，且 Windows 防火墙放行了入站 UDP `5557`/`5558`。
- **启动报 `FileNotFoundError: Could not locate GR00T G1 43-DoF USD`**：检查 `GR00T_WBC_ROOT` 路径下是否真的存在 `gear_sonic/data/robots/g1/` 目录及候选 USD 文件之一。
- **夹爪不动作**：确认手柄扳机/侧握有数据上报；注意 `deadzone=0.04`、`full_press_threshold=0.85` 两个门限（低于 4% 视为无输入，高于 85% 视为全握）。
- **改了 `ISAACLAB_G1_ZMQ_HOST` 但没有效果**：默认 `transport=udp` 下这两个变量不生效，见第 4 节说明；要么改用 `ISAACLAB_G1_UDP_BIND_HOST`（本机监听地址，一般不需要改），要么显式设 `ISAACLAB_G1_TRANSPORT=zmq` 并让远端用 ZMQ PUB。
- **`--enable_pinocchio` 相关的导入顺序问题**：`teleop_se3_agent.py` 已经保证在 `AppLauncher` 启动前先 `import pinocchio`（避免用到 Isaac Sim 自带的 pinocchio 版本），不需要手动调整。
- **传了 `--num_envs` 大于 1**：`MuJoCoG1MirrorAction` 会静默禁用镜像（只打印一次 WARN），本任务目前只支持单环境 XR 第一人称。

## 11. 相关文件索引

- [scripts/environments/teleoperation/teleop_se3_agent.py](../scripts/environments/teleoperation/teleop_se3_agent.py) — 启动入口
- [source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py) — 任务/场景/XR/teleop_devices 配置
- [source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/configs/action_cfg.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/configs/action_cfg.py) — `MuJoCoG1MirrorActionCfg` 全部字段
- [source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/mdp/actions.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/mdp/actions.py) — `MuJoCoG1MirrorAction` 实现（UDP/ZMQ 订阅、身体镜像、夹爪合成）
- [source/isaaclab/isaaclab/devices/openxr/retargeters/humanoid/unitree/g1_gripper_motion_controller.py](../source/isaaclab/isaaclab/devices/openxr/retargeters/humanoid/unitree/g1_gripper_motion_controller.py) — 手柄到夹爪的 retargeter
- [source/isaaclab/isaaclab/devices/openxr/xr_cfg.py](../source/isaaclab/isaaclab/devices/openxr/xr_cfg.py) — XR 锚点与朝向回正的完整字段定义
- [scripts/gr00t_wbc/g1_udp_network.env](../scripts/gr00t_wbc/g1_udp_network.env) — 网络参数默认值（双机 IP、端口、topic）
- [scripts/gr00t_wbc/README.md](../scripts/gr00t_wbc/README.md) — 配套的 sim2sim viewer 说明（同一套 UDP/ZMQ 协议，用于纯查看而非本任务的 XR 遥操作）
