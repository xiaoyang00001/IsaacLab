# 双 G1 程序化搬运编排（行走 → 抬箱 → 搬运行走）

> 分支：`feat/dual-g1-scripted-carry`（基于 `feat/pickplace-g1-collision-test`）
> 编排脚本：[scripts/gr00t_wbc/g1_dual_carry_choreography.py](../scripts/gr00t_wbc/g1_dual_carry_choreography.py)
> 场景道具：[locomanipulation_g1_env_cfg.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py) 中的 `carry_stand` / `carry_crate`
> 姊妹文档：[GR00T_G1_XR全身遥操作实时镜像框架.md](GR00T_G1_XR全身遥操作实时镜像框架.md)

> **⚠️ 前置依赖**：物理夹持依赖镜像手臂的 PD 弹簧驱动链路，该链路不在基分支
> `feat/pickplace-g1-collision-test` 上，需先合入以下提交（位于 `fix/zmq-object-sync-bind` /
> `backup0710` 谱系）后本演示才能真正夹住箱子：
> `28133c877`（镜像手臂改 PD 弹簧驱动，`pd_drive_joint_names`）、
> `cd711000b`（力矩上限改回真实 G1 量级）、
> `1114ca2e2`→`80868cba4`（手臂增益/阻尼调参链，终态 K=200 D=11）、
> `8cb5f36c4`（G1 前臂/手腕胶囊碰撞体脚本；本地 g1_43dof.usd 已打过补丁则不受影响）。
> 缺少这些提交时手臂是运动学硬写，掌心会穿模并把箱子顶飞（07-08 已知问题）。

## 1. 这条链路在做什么

XR 镜像框架里，两台 G1 的全身姿态来自远端 Ubuntu 的 SONIC/MuJoCo 发送端。本编排把**发送端换成一个本地 Python 脚本**：脚本按预排的时间线程序化生成两台机器人的根轨迹 + 29 DoF 关节轨迹，通过同一套 UDP 镜像协议（`MuJoCoG1MirrorAction`）驱动仿真，完成"两机器人行走 → 相向转身 → 合力抬箱 → 抬着箱子侧步行走"的完整演示。

Isaac Lab 侧**零改动复用** XR 遥操的启动命令；不需要远端主机，不需要 SONIC。

控制通道复用（与镜像框架第 6 节一致）：

| 部位 | 通道 | 效果 |
|---|---|---|
| 根位姿 + 腿/腰关节 | 运动学硬写（`write_*_to_sim`） | 步态是动画回放；两机器人间距被脚本刚性锁定 |
| 手臂关节 | PD 目标（`pd_drive_joint_names`） | roll 内收目标的跟踪误差转成持续夹持力，物理夹住箱子 |
| 手指 | XR 手柄（`G1GripperSyncAction`，可不用） | 搬运不依赖手指，靠掌心/前臂夹持 |

## 2. 启动方法

```powershell
# 终端 1：Isaac Lab（与 XR 遥操命令完全一致）
.\isaaclab.bat -p scripts\environments\teleoperation\teleop_se3_agent.py `
  --xr `
  --device cuda:0 `
  --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 `
  --teleop_device motion_controllers `
  --enable_pinocchio

# 终端 2：编排脚本（等仿真窗口出来、机器人站定后再启动）
D:\miniconda3\envs\env_isaaclab\python.exe scripts\gr00t_wbc\g1_dual_carry_choreography.py
```

要点：

- `ISAACLAB_G1_TRANSPORT=udp` 由 `g1_udp_network.env` 默认给出，两台机器人分别监听 5557/5558 与 5567/5568，脚本默认发到 `127.0.0.1`。
- 双 Windows 主机部署时加 `--hosts 192.168.10.46,192.168.10.47`。
- **不要让远端 SONIC 发送端和本脚本同时发包**，同端口会互相覆盖。
- 常用参数：`--carry-distance 1.5`（搬运距离）、`--carry-speed 0.2`、`--walk-speed 0.35`、`--dry-run`（只打印时间线不发包）。

## 3. 编排时间线（默认参数）

| 相位 | 时间 (s) | 内容 |
|---|---|---|
| settle | 0 – 2.0 | 站定，等镜像接管平滑 |
| advance | 2.0 – 5.1 | 双双沿 +Y 前行 1.09 m 到箱子两端（前向步态） |
| turn | 5.1 – 7.6 | 原地相向转身：R1 → 朝 +X，R2 → 朝 −X（踏步步态） |
| reach | 7.6 – 9.6 | 双臂前伸，掌心从外侧滑到箱子两侧面旁 |
| squeeze | 9.6 – 11.1 | shoulder_roll 内收，掌心压入箱侧 ~1 cm，PD 转夹持力 |
| lift | 11.1 – 13.1 | shoulder_pitch 加深前抬，箱底离台约 5 cm |
| carry | 13.1 – 20.6 | 两机器人保持面对面，沿世界 +Y 侧步平移 1.5 m |
| hold | 20.6 – ∞ | 保持终态持续发流 |

## 4. 行走驱动原理：运动学镜像回放，不是物理行走

机器人"行走"不是动力学控制（无 RL 策略、无 SONIC、无平衡解算），而是**运动学镜像回放**：
脚本算出每一时刻"骨盆应该在哪、腿摆什么角度"，喂给 `MuJoCoG1MirrorAction` 每个物理步硬写进仿真。
机器人本质上是被脚本牵着走的木偶，不存在平衡问题、不会摔倒。链路分三层：

### 4.1 脚本侧：程序化生成轨迹

- **根轨迹（位移的真正来源）**：`eval_robot()` 按相位时间线用 smoothstep 插值出骨盆世界位置
  (x, y, z) 与朝向 yaw——"前进 1.09 m""原地转身 90°""侧移 1.5 m"都是在根位置层面完成的。
- **腿部步态（装饰性动画）**：`gait_overlay()` 用正弦函数生成摆腿：hip_pitch 前后摆、
  knee 在摆动相抬起（`max(0, sin(φ))`）、ankle 反向补偿保持脚掌大致水平，左右腿相位差 π。
  它只负责"看起来在走"，与根位移只是节奏匹配（速度 ≈ 步频 × 步长），脚底轻微打滑是固有现象。
- 以 100 Hz 打包发 UDP：29 关节角发 5557/5567（`body_q`），根位姿发 5558/5568
  （`root_pos_w` / `root_quat_w`）。

### 4.2 环境侧：镜像动作接收执行（`MuJoCoG1MirrorAction.apply_actions`）

| 部位 | 写入方式 | 含义 |
|---|---|---|
| 根位姿 | `write_root_link_pose_to_sim()` 每步硬写 | 机器人位移由此产生 |
| 腿/腰关节 | `write_joint_state_to_sim()` 运动学硬写 | 步态是精确回放 |
| 手臂 | 只发 PD 位置目标（`pd_drive_joint_names`） | 跟踪误差经执行器转成真实接触力，夹箱是**真物理** |

### 4.3 设计取舍

这本来就是 XR 遥操框架的通道——平时由远端 SONIC/MuJoCo 算全身姿态发过来镜像，本编排只是把
发送端换成了本地脚本。好处：Isaac Lab 侧零改动；两台机器人的间距被脚本刚性锁定，搬箱时夹持
距离不会漂。代价：行走没有动力学意义——若要真正的物理行走，需走 SONIC 跟踪或 RL 策略路线
（即"场景1"框架中 P 物理跟踪的路线）。

## 5. 几何设计与 FK 校准依据

核心巧合：两机器人出生点 `(-3.8, 19.008)` / `(-2.3, 19.008)` 间距恰 1.5 m，箱子放中点 `x = -3.05` 时双方骨盆到箱心各 0.75 m，**全程无需 x 向走位**。

手臂关键帧用 pinocchio + `g1_29dof.urdf`（`d:/src/Isaac/GR00T-WholeBodyControl`）离线 FK 校准（pelvis 系，x 前 y 左 z 上）：

| 关键帧 | sp / sr / el | 掌心位置 (m) | 用途 |
|---|---|---|---|
| ready | +0.20 / ±0.20 / 0.60 | [0.13, ±0.22, −0.05] | 与 `init_state` 一致，接管无跳变 |
| reach | −0.35 / ±0.15 / 0.40 | [0.31, ±0.17, +0.10] | 掌心在箱侧外 ~4 cm，滑入无碰撞 |
| squeeze | −0.35 / ∓0.06 / 0.40 | y → ±0.11 | 压入箱侧（半宽 0.11）约 1 cm，每掌 ~15-20 N |
| lift | −0.65 / ∓0.06 / 0.55 | z → +0.14 | 抬高 ~5 cm，掌心同时更深入箱端 |

关键符号结论：**shoulder_pitch 负值 = 手臂前伸抬起**；left_shoulder_roll 正值 = 左臂外展；roll 每 −0.1 rad 掌心内收 ~2.8 cm。

道具几何（改动必须与脚本常量同步）：

- `carry_crate`：1.0 × 0.22 × 0.24 m，1.5 kg，摩擦 1.4/1.1，中心 `(-3.05, 20.10, 0.865)`。箱长 1.0 → 两端伸到距骨盆 0.25 m，掌心（前伸 0.31）握入端部约 6 cm。
- `carry_stand`：0.35 × 0.35 × 0.74 m 运动学高台，台顶 0.74 = 箱底；台面窄于箱长，不挡两端夹持位。
- 箱心高 0.86 = pelvis 0.78 + squeeze 掌心高 0.083。

夹持力预算：squeeze 内收干涉 ~1 cm → Δθ≈0.03 rad → τ≈6 N·m → 每掌 ~15 N 法向力；μ=1.2 下四掌摩擦容量 ~70 N ≫ 箱重 15 N（裕度 4×）。lift 关键帧额外内收，力上限受 `effort_limit_sim=25 N·m` 封顶，不会失稳。

## 6. 调参指南

| 症状 | 调整 |
|---|---|
| 掌心够不到箱侧 / 夹空 | 加深 `ARM_SQUEEZE.shoulder_roll_left`（更负 → 更用力内收）；或把 `carry_crate` 尺寸 y 加大 |
| 夹持过猛箱子弹飞 | squeeze roll 从 −0.06 回调到 −0.03；或降低 `physics_material` 摩擦 |
| 抬升时箱子蹭台面 | 加大 `ARM_LIFT.shoulder_pitch` 绝对值（更负 = 抬更高）；或把 `carry_stand` 高度调低 |
| 搬运途中滑落 | 降低 `--carry-speed`；lift/squeeze 的 roll 更负一点 |
| 步态脚滑严重 | 属运动学回放固有现象，调 `WALK_FREQ_HZ` 与 `--walk-speed` 匹配观感 |
| 想重演 | 仿真端按 `R` 复位箱子回高台，重启脚本（机器人瞬移回出生点重演） |

## 7. 排障 Checklist

1. 机器人不动：确认 `ISAACLAB_G1_TRANSPORT=udp`（`g1_udp_network.env` 默认）；确认 Isaac Lab 控制台出现 `MuJoCo G1 mirror received first packet`。
2. 只有一台动：检查 5567/5568 端口未被防火墙拦截；两台的包都由本脚本发出，不需要设置 `ISAACLAB_G1_2_*` 环境变量。
3. 机器人抖动/姿态跳变：确认远端 SONIC 发送端已关（同端口冲突）。
4. 箱子没被夹住直接掉落：按第 6 节调 squeeze/lift 关键帧；确认场景里 `CarryCrate` 在高台上（按 `R` 复位）。
5. XR 端 RESET（env.reset）会把机器人拉回出生点但脚本时间线不回退——重启脚本即可。

## 8. 相关文件索引

- 编排脚本（UDP 发布端）：`scripts/gr00t_wbc/g1_dual_carry_choreography.py`
- 镜像动作实现：`source/.../pick_place/mdp/actions.py`（`MuJoCoG1MirrorAction`）
- 动作配置默认值：`source/.../pick_place/configs/action_cfg.py`（`pd_drive_joint_names` 手臂 PD 通道）
- 场景道具：`source/.../pick_place/locomanipulation_g1_env_cfg.py`（`carry_stand` / `carry_crate` / `carry_crate_sync`）
- 网络端口约定：`scripts/gr00t_wbc/g1_udp_network.env`
