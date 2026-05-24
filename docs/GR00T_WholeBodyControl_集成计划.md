# GR00T-WholeBodyControl 集成计划

> **进度：阶段 1+2+3.1+3.2+3.3+3.4 集成通过；闭环反馈循环未消除，本阶段选 C 收尾，A/B 转后续 TODO（2026-05-24）**
> 完成：FK body_pos / 时间窗对齐 / reset 同步 robot 到 mocap[0] + yaw align / scale=1.0。部分成功（step 1 raw=2.56 起点修对）但 step 3→20 仍放大到 17.67、joint_pos 撞限位 3.09。**决策：接受当前为半成品停手**，A（改 actuator 匹配训练）/B（接 motion_lib + PyTorch ckpt）/C（SONIC 微调）/D（接管 robot/remote_robot）转为后续 TODO。pipeline 地基已牢固足以作微调切入点。详见"阶段 3.4 决策：选 C 收尾"。

## 概述

将 NVIDIA GR00T-WholeBodyControl 集成到当前 IsaacLab 项目中，将基于 SONIC 的全身控制**应用到 `robot` / `remote_robot`**（这两个目前 `fix_root_link=True`，需要先解除）；`walker_robot` 保留现有 `AutoWalkAction`（CPG 解析步态）作为 baseline 进行对比。

---

## 阶段 1 完成纪要（2026-05-23）

### 环境与版本
| 项 | 值 |
|---|---|
| IsaacLab 仓库版本 | **2.3.2**（[VERSION](../VERSION)，与 GR00T 要求完全匹配） |
| env_isaaclab Python | 3.11.15 |
| PyTorch | 2.7.0+cu128 |
| GPU / CUDA | RTX 3060 Laptop / 12.8 |
| gear_sonic | 0.1.0（editable，无 extras） |
| huggingface_hub | 0.36.2 |

### 已下载模型（位于 `D:/src/Isaac/GR00T-WholeBodyControl/`）

| 文件 | 大小 | 用途 |
|---|---|---|
| `gear_sonic_deploy/policy/release/model_encoder.onnx` | 48 MB | SONIC 策略编码器 |
| `gear_sonic_deploy/policy/release/model_decoder.onnx` | 40 MB | SONIC 策略解码器 |
| `gear_sonic_deploy/policy/release/observation_config.yaml` | 2.3 KB | 154D 观测定义 |
| `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx` | 739 MB | 运动学规划器（target_vel → motion ref） |
| `sonic_release/last.pt` | 448 MB | PyTorch checkpoint（微调起点） |
| `sonic_release/config.yaml` | 28 KB | 训练配置 |

### 与本文档原版的关键修正
1. **HF 模型仓库不是 `nvidia/g1_locomanip_finetune`**——那是老的 **Decoupled WBC**（GR00T N1.5/N1.6 用的下肢 RL + 上肢 IK 解耦控制器）。新一代 **GEAR-SONIC** 在 [`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC)，本项目已选择走 GEAR-SONIC 路线。
2. **SONIC 不是 cmd-driven walking policy**：观测 **154D**（详见下方"挑战 1"），输出 29D 关节目标。必须有 **motion reference 流**作为输入，不能像普通 ONNX walking policy 那样直接吃 `vx/vy/wz`。
3. **Windows 坑（已踩）**：
   - `conda run -n env_isaaclab python download_from_hf.py` 在 conda 25.5.1 上会崩 → 改用 `D:/miniconda3/envs/env_isaaclab/python.exe` 直调
   - `check_environment.py` 的 `disk_space()` 用了 `os.statvfs`（POSIX 专有），Windows 上必崩；其余 5 项检查正常

### 阶段 2 前置依赖
- ✅ `onnxruntime-gpu` 1.26.0 已安装（CPU provider 跑 dual-pass 6.45ms，~150 FPS，单 env 足够 50Hz；CUDA provider 缺 cublasLt64_12.dll/cuDNN 9，暂未启用）

### 已跳过
- 30 GB Bones-SEED SMPL 数据（仅训练用）
- `gear_sonic[training]` 的 trl / accelerate / smpl_sim（仅训练用）
- `gear_sonic[inference]` 的 Isaac-GR00T VLA 客户端（VLA 场景再装）

---

## 阶段 2 完成纪要（2026-05-23）

### 关键调研发现（**与原文档严重不一致**）

**SONIC 部署 release 模型 ≠ observation_config_example.yaml 描述的 154D**：

| ONNX | 输入 | 输出 |
|---|---|---|
| `model_encoder.onnx` | **1762D** `obs_dict` | **64D** `encoded_tokens` |
| `model_decoder.onnx` | **994D** `obs_dict` | **29D** `action` |

实际 [observation_config.yaml](D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml) 是 **multi-frame + multi-mode**（g1/teleop/smpl 三种 encoder mode），decoder 端含 `token_state` (64D) + `his_base_angular_velocity_10frame_step1` + `his_body_joint_positions_10frame_step1` + `his_body_joint_velocities_10frame_step1` + `his_last_actions_10frame_step1` + `his_gravity_dir_10frame_step1` 五块 10 帧历史。**复现这套观测构造需要时间窗 buffer + mode 切换逻辑，工作量大**。

**GR00T 仓库未提供 SONIC 的 Python 推理参考代码**：部署只有 C++（`gear_sonic_deploy/src/g1/` + TRTInference）。Python 侧仅有 Decoupled WBC 的 ONNX 加载示例（[g1_gear_wbc_policy.py](D:/src/Isaac/GR00T-WholeBodyControl/decoupled_wbc/control/policy/g1_gear_wbc_policy.py)）。

### 选定的路径：**C - 最小骨架**

不复现真实观测，**用 zero-fill 跑通 dual-pass ONNX 推理 → 写关节目标**，验证 IsaacLab → ONNX → joint write 整条 pipeline。后续阶段再补观测构造。

### 关键工程决策（与原意图的偏离）

原意图："SONIC 应用到 `robot`/`remote_robot`（解除 fix_root_link）"。

实际做法：**新增第 4 个机器人 `sonic_robot`**，与 `walker_robot` 平级，专门给 SONIC 用。

偏离原因：`robot` / `remote_robot` 与 `upper_body_ik` + ZMQ 紧耦合 — 原 `__post_init__` 里直接 `self.actions.upper_body_ik.controller.urdf_path = ...` 和 `hand_joint_names = self.actions.upper_body_ik.hand_joint_names`。强行替换会破坏 XR/teleop 流程。下阶段把 SONIC 真正接管 robot/remote_robot 时需剥离 IK + ZMQ（工作量数倍）。

### 代码改动汇总（3 个文件，< 200 行新增）

**[source/.../pick_place/mdp/actions.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/mdp/actions.py)**
- 顶部加 `SONIC_G1_29DOF_JOINT_ORDER` 常量（29 个 G1 关节，与 SONIC 训练 MJCF `g1_29dof_rev_1_0.xml` 树遍历顺序一致）
- 文件尾加 `SONICWholeBodyAction(ActionTerm)`：
  - `_load_policies()`：加载 encoder/decoder ONNX，缓存输入/输出 name 和 dim
  - `_run_sonic_zero()`：zero-fill encoder obs → tokens → 拼到 decoder obs 前 token_dim 维 → 29D action（per env 循环，无 batch dim）
  - `process_actions()`：`target = default + action_scale × action_rel[:, :n_resolved]`
  - `apply_actions()` / `reset()`：标准三件套

**[source/.../pick_place/configs/action_cfg.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/configs/action_cfg.py)**
- 新增 `SONICWholeBodyActionCfg`：
  - `encoder_path` / `decoder_path` (MISSING)
  - `joint_names` (MISSING，建议传 `list(SONIC_G1_29DOF_JOINT_ORDER)`)
  - `sonic_action_dim: int = 29`
  - `action_scale: float = 0.25`（保守 — zero-fill 推理输出曾到 ±2 rad，避免剧烈晃动；接入真实观测后调回 1.0）

**[source/.../pick_place/locomanipulation_g1_env_cfg.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py)**
- 新增常量：
  - `SONIC_G1_29DOF_CFG`（解除 fix_root_link、启用重力，pos=(-2.0, 1.5, 0.75) 与 walker 分开放）
  - `SONIC_ENCODER_PATH` / `SONIC_DECODER_PATH`（硬编码路径，下阶段可外置为环境变量或 cfg）
- `LocomanipulationG1SceneCfg` 加 `sonic_robot: ArticulationCfg`（prim `SONICRobot`）
- `ActionsCfg` 加 `sonic_wholebody = SONICWholeBodyActionCfg(...)`

### 验证
- 3 个改动文件 AST parse 全通过
- ONNX dual-pass 已在 env_isaaclab 独立验证：encoder 3.37ms + decoder 2.55ms = 6.45ms，输出 29D 数值合理

### 如何手动 Play 验证（用 sonic_verify.py，**不能用 zero_agent.py**）

⚠️ **不要用 `scripts/environments/zero_agent.py`** — `pick_place` 在 `isaaclab_tasks/__init__.py` 的 `_BLACKLIST_PKGS` 里（注释 `TODO: Remove pick_place from the blacklist once pinocchio from Isaac Sim is compatibility`），自动注册会跳过它，导致 `gymnasium.error.NameNotFound`。

为此项目新增了 [scripts/tools/sonic_verify.py](../scripts/tools/sonic_verify.py)，基于 zero_agent + 手动 import pick_place 触发 `gym.register`。每帧用 zero action 驱动 `SONICWholeBodyAction` 的 dual-pass 推理。

```powershell
# 项目根目录 d:\src\Isaac\xiaoyangIssacLab\IsaacLab\ 下执行
.\isaaclab.bat -p scripts/tools/sonic_verify.py --num_envs 1

# 如果首次启动渲染太慢/黑屏，可加 --headless 跳过 GUI，只验证场景能否构建 + SONIC 能否推理
.\isaaclab.bat -p scripts/tools/sonic_verify.py --num_envs 1 --headless --max_steps 200

# 关闭 crashreporter 旧 dump 上报（节省启动时间）
$env:CARB_CRASHREPORTER_DISABLED=1
```

弹出 Isaac Sim GUI 后点 viewport 左下 Play ▶（不自动开始的话）。

**终端预期日志**：
```
[IsaacLab] [SONIC] asset=sonic_robot resolved=29/29 joints action_scale=0.25 enc_in=1762D dec_in=994D
```
- `resolved=29/29`：29 个关节全部在 USD 中匹配到，未 skip
- `enc_in=1762D dec_in=994D`：ONNX 加载尺寸正确
- 若出现 `[SONIC] skip joint '...'`：USD 关节名与 SONIC MJCF 不一致，需建 perm 映射（阶段 3 任务）

**Viewport 预期现象**：
1. 场景出现 **4 个 G1**：`robot` / `remote_robot`（IK 站立）、`walker_robot`（在 (-2.0, 0.0, 0.75)，CPG 步态）、**`sonic_robot`（在 (-2.0, 1.5, 0.75)，walker 旁边）**
2. `sonic_robot` 摆出**固定但奇怪的姿态**（zero-fill obs → decoder 输出恒定 → 关节目标恒定）
3. 可能**缓慢倾倒**（姿态非为站立设计，且无真实反馈观测）—— 正常现象，不是失败信号

**眼测通过标准**：①日志有 `resolved=29/29` 且无 ImportError ②`sonic_robot` 出现在场景中 ③做出某种姿态（哪怕摔倒）。三项都满足 = 阶段 2 验证通过。

### 阶段 2 验证调试纪要（2026-05-23）

首次 `sonic_verify.py` 启动连续踩到 3 个独立 bug，全部修复后 SONIC 骨架核心验证通过。完整时间线如下，留作未来排错参考。

#### Bug 1：`gymnasium.error.NameNotFound: Environment Isaac-PickPlace-Locomanipulation-G1-Abs doesn't exist`

**症状**：用 `scripts/environments/zero_agent.py` 启动，task 找不到。

**根因**：[isaaclab_tasks/__init__.py:37](../source/isaaclab_tasks/isaaclab_tasks/__init__.py#L37) 的 `_BLACKLIST_PKGS` 包含 `"pick_place"`，注释为 `TODO(@ashwinvk): Remove pick_place from the blacklist once pinocchio from Isaac Sim is compatibility`。这意味着 `import_packages()` 自动注册时跳过 pick_place，必须手动 import 该子包才会触发 `gym.register`。

**为什么以前没遇到**：项目里 `record_demos.py` / `replay_demos.py` / `teleop_se3_agent.py` 等脚本都在文件头部手动 `import isaaclab_tasks.manager_based.locomanipulation.pick_place  # noqa: F401`。`zero_agent.py` 是上游通用脚本，没有这个 import。

**修复**：新建 [scripts/tools/sonic_verify.py](../scripts/tools/sonic_verify.py)，基于 zero_agent 模板 + 加上手动 import + SONIC 进度日志 + `--max_steps` 参数。

#### Bug 2：`TypeError: AutoWalkActionCfg.__init__() got an unexpected keyword argument 'forward_speed'`

**症状**：sonic_verify 在 import pick_place 时报 cfg 字段不存在。

**根因**：v3 物理驱动改造（commit `93b8e66`）废弃了 `forward_speed` 字段（脚地接触自然推进，不再需要根节点强制平移），但 [locomanipulation_g1_env_cfg.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py) 中 `walker_skeletal_walk = AutoWalkActionCfg(..., forward_speed=0.3, ...)` 调用没同步删除。

**为什么以前没遇到**：Bug 1 让 zero_agent 在 `gym.make` 之前就退出，根本到不了 import env_cfg；其他能跑这个 task 的脚本只在 teleop 测试时被偶尔启动，可能也没及时撞到。

**修复**：从 `walker_skeletal_walk = AutoWalkActionCfg(...)` 调用中删除 `forward_speed=0.3` 参数。

#### Bug 3：`AttributeError: 'AutoWalkActionCfg' object has no attribute 'forward_speed'`

**症状**：env 构造到 ActionManager._prepare_terms 时崩，行号 [mdp/actions.py:343](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/mdp/actions.py#L343)。

**根因**：Bug 2 的同源遗漏 —— `AutoWalkAction.__init__` 的启动日志 print 里也引用了 `cfg.forward_speed:.2f`。废弃字段时只删了 cfg 类定义，没全局清理引用。

**修复**：从 print f-string 删除 `speed={cfg.forward_speed:.2f}m/s` 那一段。

#### Bug 4（GUI 阶段）：`sonic_robot` 在 viewport 默认视角看不到

**症状**：headless 跑通后切 GUI，场景里没看到 sonic_robot。

**根因**：场景对齐事件（`align_robots_from_conveyor` / `align_walker_robot_to_conveyor`）把 robot/remote_robot/walker 运行时移动到 conveyor 周围 (Y≈11~18)，但 **sonic_robot 没有对应的对齐事件**，仍停留在 `SONIC_G1_29DOF_CFG.init_state.pos` 写死的 `(-2.0, 1.5, 0.75)`，距主场景 ~10m，viewport 默认相机看不到。

实际坐标（从 headless 日志摘）：
| 机器人 | 实际位置 | 来源 |
|---|---|---|
| robot | (-4.987, 14.508, 0.75) | align 事件 |
| remote_robot | (-6.237, 14.508, 0.75) | align 事件 |
| walker_robot | (-4.987, 11.008, 0.75) | align_walker 事件 |
| sonic_robot | (-2.0, 1.5, 0.75) | 配置写死，无对齐事件 |

**诊断（无需改代码）**：打开 Isaac Sim 左边 **Stage** 面板（菜单 Window → Stage 或 F4），搜 `SONIC` 找到 `/World/envs/env_0/SONICRobot` → 右键 **Frame Selected** 或选中后按 `F` 键聚焦。

**修复（最小骨架阶段足够）**：把 `SONIC_G1_29DOF_CFG.init_state.pos` 直接改到 walker 旁边，让两个并排出现在主场景视角：
```python
SONIC_G1_29DOF_CFG.init_state.pos = (-2.0, 11.008, 0.75)  # walker 同 Y，X 错开约 3m
```
后续如果场景对齐事件改了 walker 位置，需要同步更新这个写死值；终极方案是在 [mdp/events.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/mdp/events.py) 加一个 `align_sonic_robot_to_conveyor` 事件（仿照 `align_walker_robot_to_conveyor`）。

#### 经验

1. **删除 cfg 字段时全文搜索引用**：`grep -rn 'forward_speed' source/.../locomanipulation/` 会一次性暴露所有遗漏点（cfg 调用方 + 实现类的所有读取）。本项目 v3 改造时漏了 2 处。
2. **blacklist 机制的隐性约束**：`isaaclab_tasks/__init__.py` 的 `_BLACKLIST_PKGS` 是关键文件，任何 task 不在自动注册列表里 = 必须手动 import。新加 task 时记得检查。
3. **AppLauncher 的崩溃日志噪音**：每次启动 Isaac Sim 可能上报历史 `[previous crash]` dump，与本次会话无关。诊断时按 End 看最新输出，或设 `$env:CARB_CRASHREPORTER_DISABLED=1` 关掉。
4. **场景里有运行时对齐事件**：写死的 `init_state.pos` 在场景对齐事件运行后会被覆盖（robot/remote_robot/walker 都是这样）；新增机器人时要么仿照写一个对齐事件，要么调整初始位置到对齐后场景的预期范围内，否则 viewport 看不到。
5. **BAR1 显存可独立于主显存耗尽**：GUI 模式下 PhysX/render 初始化可能因 BAR1 不足而 access violation 崩溃（headless 反而没事，因为不动渲染）。如果 GUI 起不来而 headless 能跑，多半是其他应用（QQ、浏览器等）占了 BAR1，重启电脑或退出 GPU 密集应用可释放。

#### 验证通过的标志

```
[IsaacLab] [SONIC] asset=sonic_robot resolved=29/29 joints action_scale=0.25 enc_in=1762D dec_in=994D
[IsaacLab] [AutoWalkAction] asset=walker_robot freq=0.80Hz resolved_joints=...
[sonic_verify] env created; action_space=Box(-inf, inf, (1, 30), float32)
[sonic_verify] reset done; entering step loop
[sonic_verify] step=100
...
```

`action_space=(1, 30)` 拆解：SONIC 1 + walker 1 + IK 28 = 30，符合预期。

### 常见故障排查

| 报错 / 现象 | 根因 | 处理 |
|---|---|---|
| `gymnasium.error.NameNotFound: Environment Isaac-PickPlace-Locomanipulation-G1-Abs doesn't exist` | 用了 `zero_agent.py` 而非 `sonic_verify.py`；pick_place 在 isaaclab_tasks blacklist 里不会自动注册 | 改用 `scripts/tools/sonic_verify.py`（手动 import 触发注册） |
| `ImportError: SONIC requires onnxruntime` | env_isaaclab 未激活 | `isaaclab.bat` 通常自动激活 env_isaaclab，确认是否用了正确的入口脚本 |
| `FileNotFoundError: ...model_encoder.onnx` | `retrieve_file_path` 可能不识别纯本地 Windows 路径（它原本设计来处理 Omniverse Nucleus URL） | 把 [mdp/actions.py](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/mdp/actions.py) 里 `retrieve_file_path(self.cfg.encoder_path)` 直接改为 `self.cfg.encoder_path`；decoder 同理 |
| `find_joints` 报 0 match → `[SONIC] skip joint` | USD 关节命名与 SONIC MJCF 略有差异 | 临时不影响 pipeline 验证；阶段 3 要建立完整 perm 映射 |
| GUI 黑屏几分钟仍未渲染 | 首次启动 shader 编译 + 4 个 G1 USD 加载，慢是常态 | 等 3-5 分钟；不放心可同时另起一个 `--headless` 终端做诊断 |
| 终端只刷 `[previous crash]` 日志 | Isaac Sim 在上报历史 crash dump，与本次会话无关 | 设 `$env:CARB_CRASHREPORTER_DISABLED=1` 后启动可禁用 |

### 阶段 3 必做（按优先级）
1. **真实多帧 buffer**：维护过去 10 帧的 `base_angular_velocity` / `joint_pos` / `joint_vel` / `last_actions` / `gravity_dir`，按 observation_config.yaml 偏移写入 decoder 输入
2. **encoder 输入**：先选最简单的 **g1 mode**（mode_id=0，要 `encoder_mode_4` + `motion_joint_positions_10frame_step5` + `motion_joint_velocities_10frame_step5` + `motion_anchor_orientation_10frame_step5`），从固定 motion 文件回放或 `planner_sonic.onnx` 输出取
3. **关节顺序映射验证**：运行时打印 `articulation.joint_names`，确认与 `SONIC_G1_29DOF_JOINT_ORDER` 一致（USD 顺序可能不同，需建立 perm 索引）
4. **action_scale 回到 1.0**：观测真实后

---

## 参考资源

- [GR00T-WholeBodyControl 官方文档](https://nvlabs.github.io/GR00T-WholeBodyControl/index.html)
- [快速开始指南](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/quickstart.html)
- [训练指南](https://nvlabs.github.io/GR00T-WholeBodyControl/user_guide/training.html)
- [VLA 推理教程](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/vla_inference.html)
- [G1 预训练模型](https://huggingface.co/nvidia/g1_locomanip_finetune)
- [SONIC 论文](https://nvlabs.github.io/SONIC/)

## 当前状态 vs 目标状态

### 当前实现（AutoWalkAction）
- ✅ 基于解析公式的 CPG 步态
- ✅ 全身协调（腿+腰+手臂+手）
- ✅ 物理驱动
- ❌ 无学习能力
- ❌ 无视觉输入
- ❌ 无任务导向

### 目标实现（GR00T + SONIC）
- ✅ 基于 RL 训练的策略
- ✅ 全身控制（locomotion + manipulation）
- ✅ 物理驱动
- ✅ 支持 VLA（视觉-语言-动作）
- ✅ 任务导向（pick & place）
- ✅ 可微调

## 集成步骤

### 阶段 1：环境准备

1. **安装 GR00T-WholeBodyControl**
   ```bash
   # 克隆仓库
   git clone https://github.com/NVlabs/GR00T-WholeBodyControl.git
   cd GR00T-WholeBodyControl
   
   # 安装依赖（已有 Isaac Lab）
   pip install -e .
   ```

2. **验证 Isaac Lab 兼容性**
   - 检查当前 Isaac Lab 版本
   - 确认与 GR00T 要求的版本匹配

### 阶段 2：创建 SONIC Action Term

创建新的 Action Term：`source/isaaclab_tasks/.../mdp/actions.py`

```python
class SONICWholeBodyAction(ActionTerm):
    """基于 SONIC 的全身控制 Action Term。
    
    集成 NVIDIA GR00T-WholeBodyControl 的 SONIC 控制器，
    实现 50Hz 全身关节控制（locomotion + manipulation）。
    """
    
    cfg: SONICWholeBodyActionCfg
    _asset: Articulation
    
    def __init__(self, cfg: SONICWholeBodyActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._env = env
        
        # 加载 SONIC 策略
        self._policy = self._load_sonic_policy(cfg.policy_path)
        
        # 初始化观测缓冲区
        self._obs_buffer = self._init_observation_buffer()
        
    def _load_sonic_policy(self, policy_path: str):
        """加载预训练的 SONIC 策略"""
        # TODO: 实现策略加载逻辑
        pass
    
    def process_actions(self, actions: torch.Tensor):
        """处理动作：观测 → 策略 → 关节目标"""
        # 1. 收集观测
        obs = self._collect_observations()
        
        # 2. 运行策略
        joint_targets = self._policy(obs)
        
        # 3. 更新关节目标
        self._processed_actions = joint_targets
    
    def _collect_observations(self) -> torch.Tensor:
        """收集 SONIC 所需的观测"""
        # 关节位置、速度
        # 根节点状态
        # IMU 数据
        # 接触力
        pass
```

### 阶段 3：配置文件

创建 `SONICWholeBodyActionCfg`：

```python
@configclass
class SONICWholeBodyActionCfg(ActionTermCfg):
    """SONIC 全身控制配置"""
    
    class_type: type[ActionTerm] = SONICWholeBodyAction
    
    policy_path: str = MISSING
    """SONIC 策略文件路径（.pt 或 .onnx）"""
    
    control_frequency: float = 50.0
    """控制频率（Hz），SONIC 默认 50Hz"""
    
    use_vla: bool = False
    """是否使用 VLA（视觉-语言-动作）模式"""
    
    vla_model_path: str | None = None
    """VLA 模型路径（如果 use_vla=True）"""
```

### 阶段 4：环境配置

修改 `locomanipulation_g1_env_cfg.py`：

```python
# 使用 SONIC 控制的 walker
from .configs.action_cfg import SONICWholeBodyActionCfg

walker_sonic_control = SONICWholeBodyActionCfg(
    asset_name="walker_robot",
    policy_path="{ISAAC_LAB_NUCLEUS_DIR}/Policies/sonic_g1_locomanip.pt",
    control_frequency=50.0,
    use_vla=False,
)
```

### 阶段 5：下载预训练模型

> **已完成于 2026-05-23**——见顶部"阶段 1 完成纪要"。
> 实际仓库是 `nvidia/GEAR-SONIC`（不是 `g1_locomanip_finetune`，后者属于老的 Decoupled WBC）。下载方式不用 `huggingface-cli`，用 GR00T 仓库自带脚本：

```bash
cd D:/src/Isaac/GR00T-WholeBodyControl
# 部署 ONNX（model_encoder/decoder + planner，~830 MB）
D:/miniconda3/envs/env_isaaclab/python.exe download_from_hf.py
# 训练 PyTorch ckpt（~448 MB，跳过 30 GB SMPL）
D:/miniconda3/envs/env_isaaclab/python.exe download_from_hf.py --training --no-smpl
```

### 阶段 6：测试与验证

1. **基础测试**
   ```bash
   python scripts/run_env.py --task Isaac-Locomanipulation-G1-v0
   ```

2. **验证指标**
   - 控制频率是否达到 50Hz
   - 机器人是否稳定行走
   - 全身协调性
   - 任务完成率（pick & place）

### 阶段 7：微调（可选）

如果需要针对特定任务微调：

```bash
# 使用 Isaac Lab 训练
python scripts/train_sonic.py \
    --task Isaac-Locomanipulation-G1-SONIC-v0 \
    --num_envs 4096 \
    --headless
```

## 技术挑战与解决方案

### 挑战 1：观测空间匹配
**问题**：SONIC 需要 **154D** 观测，远比普通 walking policy 复杂。

SONIC G1 默认观测构成（来自 `gear_sonic_deploy/policy/release/observation_config.yaml`）：

| 字段 | 维度 | 偏移 | 来源 | IsaacLab 对接方式 |
|---|---|---|---|---|
| `motion_joint_positions` | 29 | 0 | motion reference 帧 | 需 mocap / 规划器输出，**关键依赖** |
| `motion_joint_velocities` | 29 | 29 | motion reference 时间导数 | 同上 |
| `motion_anchor_orientation` | 6 | 58 | motion anchor 旋转矩阵前两列 | 由 motion 数据 + 根 quat 计算 |
| `base_angular_velocity` | 3 | 64 | IMU 角速度 | `articulation.data.root_ang_vel_b` |
| `body_joint_positions` | 29 | 67 | 当前关节角 | `articulation.data.joint_pos` |
| `body_joint_velocities` | 29 | 96 | 当前关节速度 | `articulation.data.joint_vel` |
| `last_actions` | 29 | 125 | 上一帧策略输出 | Action Term 自维护缓存 |
| **总计** | **154** | | | |

**解决**：
1. **关节顺序对齐**——SONIC 的 29 DoF 关节顺序需与 IsaacLab `G1_29DOF_CFG` 对齐（必须按 SONIC 训练时的顺序排列输入/输出）
2. **motion reference 来源**——三选一：
   - (a) `planner_sonic.onnx`（target_vel → motion 帧）——最接近"速度命令"接口
   - (b) 离线 mocap 文件回放（BVH/SMPL/CSV）
   - (c) 外部 ZMQ pose 流（已有 `G1TriHandUpperBodyZeroMqRetargeter` 可参考）
3. **观测顺序与 enabled 标志**可在 `observation_config.yaml` 调整（offset 自动重算）

### 挑战 2：控制频率
**问题**：SONIC 需要 50Hz，当前环境可能不同
**解决**：调整环境 `decimation` 参数，或在 Action Term 中实现频率转换

### 挑战 3：ZMQ 通信（部署阶段）
**问题**：真实机器人部署需要 ZMQ 通信
**解决**：实现 ZMQ 桥接，连接 Isaac Lab 仿真与 C++ 部署

### 挑战 4：VLA 集成
**问题**：VLA 需要视觉输入和语言指令
**解决**：
- 添加相机传感器到场景
- 实现语言指令解析
- 集成 Isaac-GR00T N1.7 模型

## 与现有代码的兼容性

### 保留的部分
- ✅ 环境配置结构
- ✅ 场景设置（三个机器人）
- ✅ 事件系统（对齐、重置）
- ✅ 观测系统

### 替换的部分
- ❌ `AutoWalkAction` → `SONICWholeBodyAction`
- ❌ CPG 步态生成 → RL 策略
- ❌ 解析公式 → 神经网络

### 共存方案
可以保留 `AutoWalkAction` 作为 baseline：
```python
# 配置中提供两种选择
walker_baseline = AutoWalkActionCfg(...)  # 原有实现
walker_sonic = SONICWholeBodyActionCfg(...)  # GR00T 实现
```

## 时间估算

| 阶段 | 预计时间 | 说明 |
|------|---------|------|
| 环境准备 | 1-2 天 | 安装、配置、验证 |
| Action Term 实现 | 3-5 天 | 核心逻辑、观测适配 |
| 配置与集成 | 1-2 天 | 环境配置、模型加载 |
| 测试与调试 | 2-3 天 | 功能验证、性能优化 |
| 文档更新 | 1 天 | 更新说明文档 |
| **总计** | **8-13 天** | 不含微调训练 |

## 下一步行动

1. **阶段 1：环境准备** ✅ 已完成（2026-05-23）
   - [x] 克隆 GR00T-WholeBodyControl 仓库到 `D:/src/Isaac/GR00T-WholeBodyControl/`
   - [x] env_isaaclab 安装 gear_sonic core + huggingface_hub
   - [x] 下载 GEAR-SONIC 部署版 ONNX + 训练版 PyTorch ckpt
   - [x] 确认 IsaacLab 2.3.2 / Python 3.11.15 / PyTorch 2.7.0 兼容

2. **阶段 2：最小骨架** ✅ 已完成（2026-05-23）
   - [x] 安装 `onnxruntime-gpu` 1.26.0 到 env_isaaclab（CPU provider 推理 OK，6.45ms/dual-pass）
   - [x] 验证 SONIC dual-pass ONNX：encoder 1762→64 → decoder 994→29，输出数值合理
   - [x] 提取 SONIC 训练用 G1 29 DoF 关节顺序（`g1_29dof_rev_1_0.xml` MJCF）
   - [x] 在 `mdp/actions.py` 创建 `SONICWholeBodyAction`（zero-fill 观测，dual-pass 推理）+ `SONIC_G1_29DOF_JOINT_ORDER` 常量
   - [x] 在 `configs/action_cfg.py` 创建 `SONICWholeBodyActionCfg`
   - [x] **偏离原意图**：未动 robot/remote_robot（紧耦合 IK+ZMQ），改为新增第 4 个机器人 `sonic_robot`
   - [x] 三个文件 AST parse 全通过
   - [x] 创建 `scripts/tools/sonic_verify.py`（pick_place 在 blacklist 里，必须手动 import 触发 gym.register）
   - [x] 修复 v3 物理驱动后 `forward_speed` 字段未清理的 2 处遗漏（env_cfg 调用方 + AutoWalkAction.__init__ print）
   - [x] **运行时通过**：`sonic_verify.py --headless --max_steps 200` 跑完 0 错（SONIC 加载 / env 构造 / reset / 200 step loop 全过）
   - [ ] *待 GUI 眼测*：去掉 `--headless` 看 `sonic_robot` 实际姿态表现

3. **阶段 3：真实观测构造**（按子阶段递进）
   - **3.1 真实 decoder 观测**（2026-05-23）
     - [x] `SONICWholeBodyAction` 新增 5 块 history buffer：`_hist_base_ang_vel(N,10,3)` / `_hist_joint_pos(N,10,29)` / `_hist_joint_vel(N,10,29)` / `_hist_last_actions(N,10,29)` / `_hist_gravity_dir(N,10,3)`，全部按 SONIC 关节顺序
     - [x] `_push_history()` 每步 FIFO 推入：`root_ang_vel_b` / `joint_pos[:, _joint_ids]` / `joint_vel[:, _joint_ids]` / `_last_action` / `projected_gravity_b`
     - [x] `_build_decoder_input()` 按 994D 偏移精确拼装（见下方"decoder 994D 偏移表"）
     - [x] `reset()` 重置 history：joint_pos 回 default，gravity 回 (0,0,-1)，其余清零
     - [x] **headless 通过**：`sonic_verify --headless --max_steps 200` 200 帧零错，日志含 `history_len=10`
     - [x] **GUI 暴露 frame-major layout 错误**：sonic_robot 摔倒后乱动；为隔离"训练分布外"问题，临时改 `fix_root_link=True + disable_gravity=True` 让机器人悬空，并加 `[SONIC] step=...` 每 50 步 debug 打印；数据显示 frame-major 时 `action absmax=25~27`（vs zero-fill 基线 1.9），明显 garbage。已切到 **dim-major** 重试。
     - [x] **dim-major 数值验证通过**（2026-05-23）：`action absmax=1.27~2.62`、`mean≈-0.1~-0.2`、`std=0.55~0.84`、`joint_pos absmax=1.14（远离限位）`。完全落在合理范围，模型在响应观测产生有意义输出。
     - [x] **物理解锁配置已落地**（2026-05-23）：`SONIC_G1_29DOF_CFG.spawn.articulation_props.fix_root_link = False`、`disable_gravity = False`；`action_scale = 1.0`（从 0.25 回到 SONIC 训练默认）
     - [x] **物理验证结果**（2026-05-23）：**立刻摔倒**。数据 `action absmax=14~16`（远超悬空 dim-major 时 1.27~2.62）、`mean=-2~-3` 持续负偏置、`joint_pos absmax=3.0+` 撞关节限位。机器人趴地后训练分布外 → 输出 garbage → 撞限位恶性循环。
     - [x] **结论：decoder 真观测不足以维持平衡，必须接 encoder 提供 motion reference**（进 3.2 A 路径）
     - [x] **回滚配置**：`fix_root_link = True + disable_gravity = True`，让 3.2 调试在站立姿态进行
   - **3.2 encoder g1 mode 输入**（难度评估完成 2026-05-23，待选路径）

     **难点**：encoder 1762D 由 14 个字段拼成，但 [observation_config.yaml](D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml) 不给每个字段的精确维度。按经验推测：

     | 字段 | 推测维度 | g1 mode 必需 |
     |---|---|---|
     | `encoder_mode_4` | 4D（mode one-hot 或 4 floats） | ✓ |
     | `motion_joint_positions_10frame_step5` | 29×5 = 145D | ✓ |
     | `motion_joint_velocities_10frame_step5` | 29×5 = 145D | ✓ |
     | `motion_root_z_position_10frame_step5` | 1×5 = 5D | |
     | `motion_root_z_position` | 1D | |
     | `motion_anchor_orientation` | 6D | |
     | `motion_anchor_orientation_10frame_step5` | 6×5 = 30D | ✓ |
     | `motion_joint_positions_lowerbody_10frame_step5` | 12×5 = 60D | （teleop mode 用） |
     | `motion_joint_velocities_lowerbody_10frame_step5` | 12×5 = 60D | （teleop mode 用） |
     | `vr_3point_local_target` | 9D（head/lhand/rhand × xyz） | （teleop mode 用） |
     | `vr_3point_local_orn_target` | 18D（3 点 × 6D rotation） | （teleop mode 用） |
     | `smpl_joints_10frame_step1` | **未知**（22 或 24 关节 × 3 × 10？） | （smpl mode 用） |
     | `smpl_anchor_orientation_10frame_step1` | 6×10 = 60D | （smpl mode 用） |
     | `motion_joint_positions_wrists_10frame_step1` | **未知**（wrists 关节数不明） | （smpl mode 用） |

     推测累加约 1200~1500D，离 1762 还差 200~500D。精确逆向需要读 [sonic_release/config.yaml](D:/src/Isaac/GR00T-WholeBodyControl/sonic_release/config.yaml) 中各 obs term 的 func 定义，或读 [gear_sonic_deploy/src/g1/](D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/) 的 C++ 部署代码。预计 1-2h。

     **路径已选 C → A**（2026-05-23）：3.1 物理验证已跑，机器人立刻摔倒 → **必须接 encoder，进 A 路径**。

     **D2 最小试探**（2026-05-24）失败：填 `enc[0,0]=1.0` 后 `action absmax=12~18`、`joint_pos absmax=3.0+ 撞限位` —— 与摔倒分布外几乎一致。证明假设全错，撤回到 zero-fill。

     **D1 重大解码**（2026-05-24，从 [gear_sonic/utils/inference_helpers.py:200-310](D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic/utils/inference_helpers.py) 的 ONNX export 代码逆向）：
     - Encoder input 真实 layout: `[encoder_index(1) | tokenizer_obs 按 sonic_release/config.yaml tokenizer 段顺序拼接]`
     - **encoder_index 是 1 个 long scalar**（不是 4D one-hot）：value=mode_id（0=g1, 1=teleop, 2=smpl），forward 时 `inputs[..., 0].long()` 取出，cast 后 0=g1
     - **3.1 encoder zero-fill 隐式选了 mode_id=0=g1**（因为 zero = 0.0 cast long = 0），这就是 dim-major 数值合理（absmax=1.27~2.62）的原因！
     - D2 失败：`enc[0,0]=1.0 → mode_id=1 → teleop`，与 g1 inputs 错位 → garbage
     - **训练 g1 encoder 只用 2 个字段**（来自 sonic_release/config.yaml line 75-78）：
       - `command_multi_future_nonflat`（offset 1-420，420D = 10 frames × 14 bodies × 3）
       - `motion_anchor_ori_b_mf_nonflat`（offset 431-490，60D = 10 frames × 6D rotation）

     **D1 实施**（2026-05-24，代码已落地）：
     - `_build_encoder_input(env_idx)`：显式 `enc[0,0]=0.0` 选 g1 mode；填 self-reference body_pos（14 个 SONIC body 在 pelvis 坐标系下的位置，10 帧重复）；motion_anchor_ori 填 identity 6D（reference orientation == robot orientation）
     - `_init_sonic_body_indices()`：用 `find_bodies` 解析 14 个 SONIC body link 到 USD 索引
     - `_compute_self_ref_body_pos_b()`：用 `body_link_pos_w - root_link_pos_w` + `quat_apply_inverse(root_quat_w, ...)` 转 body frame

     **D1 第一次验证（frame-major repeat）失败**（2026-05-24）：
     - `[SONIC INIT] body indices resolved: 14/14` ✅ — 14 个 body 全部解析（ids=`[0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]`）
     - `self_ref_body_pos absmax=0.7466 mean≈0` ✅ — body_pos_b 数据合理（~75cm 与 G1 几何一致）
     - 但 `action absmax=10~22, mean=-1~-3, joint_pos absmax=2.6~3.1（撞限位）` ❌ — 与 D2 失败模式一致
     - **根因**：body data 与 mode 都对，问题在 **flatten layout** —— 当前用 `np.tile(body_flat, 10)` 是 **frame-major repeat**（`[f0_b0..b13, f1_b0..b13, ...]`），但 decoder 端已验证 SONIC 用 **dim-major**

     **D1 修正 v2（dim-major repeat）**（2026-05-24）：
     - 改用 `np.repeat(body_flat, 10)` —— dim-major（每维 10 帧连续）
     - **关键区别**：`np.tile([a,b,c], 3)`=`[a,b,c,a,b,c,a,b,c]`（frame-major）；`np.repeat([a,b,c], 3)`=`[a,a,a,b,b,b,c,c,c]`（dim-major）
     - **结果**：仍 garbage（`absmax=12.75, mean=-1.79, std=4.52` vs frame-major `14.54, -2.07, 5.30` 几乎相同）

     **standalone encoder layout probe** ([scripts/tools/sonic_encoder_layout_probe.py](../scripts/tools/sonic_encoder_layout_probe.py))（2026-05-24）：

     直接加载 ONNX 跑遍 layout 组合（30 秒，不需要 IsaacSim）：

     | 测试 | encoder | decoder history | action absmax |
     |---|---|---|---|
     | baseline | 全 zero | zero | **1.92** ✅ |
     | body_pos dim-major + id6 dim-major | self-ref | zero | **2.75~2.93** ✅ |
     | body_pos frame-major + id6 frame-major | self-ref | zero | **3.21~3.32** ≈ OK |
     | mode_id=1 (teleop) only | zero | zero | 1.09 |
     | sonic_verify D1 v2 (实际) | self-ref dim-major | **真实 history** | **12~22** ❌ |

     **结论**：**问题不在 encoder layout**！encoder 单独填充时 absmax 总在 1.7~3.3 合理范围。
     **真实根因**：encoder token + decoder real history 的组合产生 garbage。

     **推测的本质问题**：sonic_robot 当前 `fix_root_link=True` 让 decoder history 是"完全静止"
     （joint_vel=0、base_ang_vel=0、joint_pos 不变、gravity 不变）——
     这是 **SONIC 训练分布外**（训练时机器人在跟踪 mocap 动作）。
     模型看到"encoder token 说追踪 self-ref + history 说机器人完全僵死"→ 解读为异常 → garbage。

     **E4 standalone history 模拟**（2026-05-24，复现失败 = 反向有效结论）：

     扩展 [scripts/tools/sonic_encoder_layout_probe.py](../scripts/tools/sonic_encoder_layout_probe.py) 加 `build_decoder_history()` 模拟真实静止 history（G1 default pose 静止 + gravity[0,0,-1]）：

     | 测试 | encoder | history | absmax |
     |---|---|---|---|
     | zero everything | zero | zero | 1.92 |
     | zero + static history (joint_pos absolute) | zero | static abs | **2.23** |
     | zero + static history (joint_pos_rel=0) | zero | static rel | **2.15** |
     | self-ref + static history (joint_pos absolute) | self-ref | static abs | **2.69** |
     | self-ref + static history (joint_pos_rel=0) | self-ref | static rel | **2.62** |
     | **sonic_verify 实际跑** | self-ref | **真实** | **12~22** |

     **结论**：standalone 无法复现 garbage。说明问题是 **反馈循环**：
     - 第 0 帧 standalone-level absmax=2.6 → action_scale=1.0 × 2.6 = 2.6 rad 偏移
     - 写关节 → 关节扭曲到极限 (3.0+) → history 进入 OOD 极端 pose
     - 第 1 帧 SONIC 看到 OOD history → 训练分布外 → garbage 变大
     - 正反馈循环放大到 absmax=12+

     **额外发现 joint_pos_rel bug**：sonic_release/config.yaml line 458 用 `joint_pos_rel`（相对 default），
     我代码传**绝对 joint_pos**。standalone 显示影响较小（2.23 vs 2.15），但应修复对齐训练。

     **更深层根因**：**self-reference 假设本身错**。SONIC 训练时 reference = mocap 帧（与 robot 略不同），
     robot 在追踪。让 reference=robot 自己是训练分布外，必须接真实 mocap (E3)。

     **E4 后立即修复**（2026-05-24，已落地）：
     - `_push_history()` 改用 `joint_pos - default` 传 joint_pos_rel
     - `_init_history()` / `reset()` 把 `_hist_joint_pos` 初始化从 default → zero
     - `action_scale: 1.0 → 0.2` 缓冲 self-ref 反馈循环

     **修复实测通过**（2026-05-24，sonic_verify GUI）：

     | 指标 | 修复前 garbage | **修复后** |
     |---|---|---|
     | action absmax | 12~22 | **1.65~2.73** ✅ |
     | action mean | -1~-3 | **-0.10~-0.20** ✅ |
     | action std | 4~5.7 | **0.58~0.83** ✅ |
     | joint_pos absmax (relative) | 3.0+ 撞限位 | **1.34** ✅ 远离限位 |

     数值范围与 encoder zero-fill 基线（absmax=1.27~2.62）几乎一致，反馈循环被打破。
     但 self-ref 仍是 OOD 假设，sonic_robot 摆出"奇怪固定姿态"而非真有意义的目标跟踪。

     **E3 第一步 mocap 接入**（2026-05-24，已落地 + 实测通过）：
     - 下载 sample mocap：`D:/src/Isaac/GR00T-WholeBodyControl/sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl`（1202 帧、30 fps、40.1s walking）
     - **mocap 格式**：joblib + zlib 压缩；顶层 `{motion_name: motion_dict}`；含 root_rot(xyzw, 1202×4)、root_trans_offset(1202×3)、dof(1202×29)、pose_aa(1202×30×3)、smpl_joints(1202×24×3)
     - **关键 detail**：mocap quat 是 **xyzw 顺序**，IsaacLab 用 wxyz，需 `[3,0,1,2]` 重排
     - `_load_mocap()` 加载 + 转 wxyz + 缓存到 device tensor
     - `_build_encoder_input()` 的 `motion_anchor_ori_b_mf_nonflat`：取未来 10 帧（step=5）mocap root_rot → matrix_from_quat → `mat[..., :2]` row-major reshape → dim-major flatten
     - `_advance_mocap()` 每 process_actions 推进 1 帧
     - **GUI 实测结果**：absmax 1.91~3.02（vs E3 前 identity anchor 时 1.65~2.73），mean/std 微增——**mocap 信号确实通过 token 影响 decoder 输出**，但量级小（anchor_ori 仅占 encoder 1762D 的 3.4%，body_pos 420D 仍 self-ref 主导）
     - 仍未做：body_pos 用 mocap pose（需 forward kinematics）→ step 3b

     **E3 D 物理验证**（2026-05-24，✅ 突破性结果）：
     - 解 fix_root_link 让 sonic_robot 物理生效
     - **站立时长 > 5 秒**（vs 3.1 立刻摔倒），`self_ref_body_pos absmax 从 0.7466 → 0.6822` = robot 在物理下真的在动 + body 相对 pelvis 位置在改变
     - mocap anchor 信号在物理下提供了有意义的平衡基准，证明这条路是对的
     - 但 absmax 仍 12.69（OOD body_pos self-ref 仍主导），需要 step 3b 提供 mocap body_pos

     **E3 step 3b（SMPL→SONIC body_pos 近似映射，2026-05-24，失败）**：
     - 假设：mocap PKL 的 `smpl_joints (1202, 24, 3)` 可作为 14 body 近似
     - **实测失败原因**：robot_filtered 文件的 smpl_joints **全是 0**（placeholder）！只有 `dof (1202, 29)` 是真实关节角数据
     - 真实 SMPL 数据在 smpl_filtered/ 目录，但 SMPL 是给 mode=2 用的，不是 g1 mode 需要的 G1 body geometry
     - g1 mode 的 motion_lib 实际上用 dof + URDF 做 forward kinematics 算 14 body in pelvis frame；deploy 时 mocap PKL **不预存 body_pos**

     **pinocchio Windows 安装失败**（2026-05-24）：
     - `pip install pin` 卡在 cmeel-boost 编译，Windows 缺 boost C++ 库
     - 替代方案待选：
       - IsaacLab Articulation 临时 set joint state + sim step 1202 次预算 body_pos（约 24s 启动延迟）
       - 离线写 IsaacSim 启动脚本预算 body_pos 到 .npy 缓存
       - 切换 WSL2 装 pinocchio

     **当前状态（fallback 到 self-ref body_pos）**：
     - SMPL 全零检测 + fallback 逻辑已加：smpl absmax < 1e-6 → `_mocap_body_pos_b = None` → 走 self-ref 分支
     - 等同于 E3 D 物理验证状态：站立 > 5 秒 + mocap anchor_ori 60D 时变信号
     - **真正的下一步是 FK 方案**（让 body_pos 跟 mocap dof）—— 见上方"替代方案待选"

     **A 路径调研进展（D1 部分）**：
     - 部署字段 ↔ 训练 obs term ↔ func 映射已找全（在 gear_sonic/envs/manager_env/mdp/observations.py）
     - 关键事实：`num_future_frames=10`、`num_bodies=14`（pelvis + 6 leg + torso + 6 arm，关键骨架点）
     - 维度修正（已确认）：
       - `motion_joint_positions_10frame_step5` = **420D**（10×14×3，body positions in body-local frame，非 joint angles！命名误导）
       - `motion_anchor_orientation_10frame_step5` = **60D**（10×6）
       - `motion_anchor_orientation` = **6D**（单帧）
       - `motion_root_z_position_10frame_step5` = **10D**、`motion_root_z_position` = **1D**
     - 累加已确认 ~921D，剩 ~841D 给 lower_body / vr / smpl / wrists 6 字段（待 D1 精确）

     备选路径：
     - **A. 硬解 1762**：读 sonic_release/config.yaml 中 obs term 的 func 定义 / C++ 部署代码；构造 self-reference 填 g1 mode 必需 4 字段，其余 zero
     - **B. 全 encoder zero-fill**：token_state 全 0（5min），作为下界 baseline
   - **3.3 真正的 motion reference 源**
     - [ ] 选 A：`planner_sonic.onnx`（target_vel → motion 帧）— 速度命令接口
     - [ ] 选 B：从 GR00T 仓库的 sample_data/ 加载 mocap 文件回放（`download_from_hf.py --sample` 下载）
   - **3.4 收尾**
     - [ ] 运行时打印 `articulation.joint_names` 与 `SONIC_G1_29DOF_JOINT_ORDER` 对照，确认无 perm 错位
     - [ ] action_scale 0.25 → 1.0（接入真实观测后）

#### decoder 994D 输入偏移表（按 observation_config.yaml release 段，frame-major flatten）

| 偏移 | 大小 | 字段 | IsaacLab 来源 |
|---|---|---|---|
| [0:64] | 64 | `token_state` | encoder 当帧输出 |
| [64:94] | 30 | `his_base_angular_velocity_10frame_step1` | `root_ang_vel_b` × 10 帧 |
| [94:384] | 290 | `his_body_joint_positions_10frame_step1` | `joint_pos[:, _joint_ids]` × 10 帧 |
| [384:674] | 290 | `his_body_joint_velocities_10frame_step1` | `joint_vel[:, _joint_ids]` × 10 帧 |
| [674:964] | 290 | `his_last_actions_10frame_step1` | `_last_action` × 10 帧 |
| [964:994] | 30 | `his_gravity_dir_10frame_step1` | `projected_gravity_b` × 10 帧 |
| **总计** | **994** | | |

> ✅ flatten layout 实测为 **dim-major**（`[d0_f0..f9, d1_f0..f9, ...]`），即每维的 10 帧时间序列连续存放。代码实现：`tensor[env_idx].t().flatten()`（(10, K) → (K, 10) → 1D）。
>
> **判别证据**（2026-05-23 GUI 实测，sonic_robot 悬空状态）：
> | layout | `action absmax` | `action std` | `joint_pos absmax` | viewport |
> |---|---|---|---|---|
> | zero-fill 基线 (阶段 2) | ~1.9 | 0（恒定） | 不撞限位 | 恒定姿态 |
> | **frame-major** ❌ | **25~27** | **6.5** | **1.97（撞限位）** | 关节抽搐 |
> | **dim-major** ✅ | **1.27~2.62** | **0.55~0.84** | **1.14（远离限位）** | 关节小幅微动（待确认） |

4. **长期目标**：
   - [ ] 集成 VLA 功能（需补 `gear_sonic[inference]`）
   - [ ] 针对特定任务微调（需补 30 GB SMPL + `gear_sonic[training]`，64+ GPU 训练）
   - [ ] 部署到真实 G1 机器人（C++ 推理栈 + TensorRT）

---

## 已完成 / 剩余工作梳理（2026-05-24）

### ✅ 已完成

#### 阶段 1 — 环境准备
- GR00T-WholeBodyControl 仓库克隆（2.17 GiB）
- `env_isaaclab` 装 `gear_sonic` core（IsaacLab 2.3.2 + Py 3.11.15 + Torch 2.7.0+cu128）
- GEAR-SONIC ONNX（encoder 48MB + decoder 40MB）+ PyTorch ckpt + observation_config.yaml + planner ONNX 全部下载
- `onnxruntime-gpu` 1.26.0 装好（CPU provider dual-pass 6.45ms，单 env 50Hz 足够）
- Windows 坑已记录：`conda run` 在 conda 25.5.1 上崩 → 直调 python.exe；`check_environment.py` 的 `os.statvfs` Windows 不可用

#### 阶段 2 — 最小骨架
- 新增 `SONICWholeBodyAction` + `SONICWholeBodyActionCfg`（< 200 行）
- 新增 `sonic_robot`（第 4 个机器人），不动 robot/remote_robot 的 IK + ZMQ 耦合
- `sonic_verify.py` 启动脚本（手动 import 触发 gym.register，绕过 `_BLACKLIST_PKGS`）
- zero-fill encoder 跑通 dual-pass ONNX → 29D action → joint write 整条 pipeline
- headless 200 帧零错

#### 阶段 3.1 — 真实 decoder 观测（994D）
- 10-frame history buffer 5 块（base_ang_vel / joint_pos_rel / joint_vel / last_actions / gravity_dir）
- **dim-major flatten** 实测确认（`tensor.t().flatten()`，vs frame-major absmax 25→1.27 差距明显）
- action absmax 稳定 1.27~2.62（vs encoder zero-fill 基线一致）
- `joint_pos_rel` = `joint_pos - default_joint_pos`（对齐 sonic_release/config.yaml line 458）

#### 阶段 3.2 — Encoder 1762D 真实布局
- mode_id=0（g1）确认是 1D 标量 long，不是 4D one-hot（D2 失败教训）
- body_pos 420D layout = 10×14×3，**body in pelvis frame**，dim-major flatten（`np.repeat`）
- anchor_ori 60D = 10×6（rotation matrix 前 2 列）
- 14 body 名单 `SONIC_BODY_NAMES` 已定义并 resolve 通过
- standalone 探测脚本 `sonic_encoder_layout_probe.py` 验证 layout 假设

#### 阶段 3.3 — Mocap anchor_ori 接入 + 物理验证
- sample mocap：`walk_forward_amateur_001__A001.pkl`（1202 帧、30 fps、40.1s walking）
- 格式：**joblib + zlib 压缩**；root_rot **xyzw → wxyz**（`[3,0,1,2]` 重排）
- `_advance_mocap` 每 process_actions 推进 1 帧
- mocap-based anchor_ori 60D 时变信号已通过 encoder token 影响 decoder
- **关键突破**：解 `fix_root_link` 让 sonic_robot 物理生效 → **站立时长 > 5 秒**（vs 3.1 立刻摔倒）
- `action_scale: 1.0 → 0.2` 缓冲 self-ref 反馈循环 → action absmax 1.65~3.02 稳定

#### 工程化基础设施
- `sonic_verify.py` headless / GUI 验证脚本
- 每 50 步 debug print：action mean/absmax/std + joint_pos absmax + self_ref_body_pos 统计
- 文档 + memory 全部同步

### ❌ 未完成

#### 核心阻塞：body_pos FK
- 现状：mocap PKL 的 `smpl_joints (1202, 24, 3)` 是 placeholder 全 0，自动 fallback 到 self-ref
- self-ref 是训练分布外（SONIC 训练 reference=mocap 帧 ≠ robot 自身），导致 body_pos 420D 仍是主导误差源
- 三个待选方案：
  - **F1** IsaacLab Articulation runtime 临时驱动 + sim step 预算（30-60 min，可能 sim step 时序卡顿）
  - **F2** 独立 IsaacSim 启动脚本预算 body_pos 到 `.npy` 缓存（45 min，最干净，推荐）
  - **F3** 手写 G1 FK 链（60-90 min，复杂、易出错）
- pinocchio Windows 安装失败（cmeel-boost 编译缺 boost），暂不走 pinocchio 路线

#### 次要清理项
- [ ] `action_scale: 0.2 → 1.0` 调回（body_pos FK 落地后再尝试）
- [ ] `sonic_robot.init_state.pos` 当前硬编码 `(-2.0, 11.008, 0.75)`，应改成 event 对齐 walker_robot
- [ ] mocap 帧率：当前 1 frame/process_actions，实际播放 1.67x 真实速度（应每 1/0.6 ≈ 1.67 step 推一帧）
- [ ] 运行时打印 `articulation.joint_names` 与 `SONIC_G1_29DOF_JOINT_ORDER` 对照，确认无 perm 错位
- [ ] CUDA provider 启用（缺 cublasLt64_12.dll / cuDNN 9）— 单 env 不急

#### 原意图未完成
- [ ] SONIC 真正接管 `robot` / `remote_robot`（剥离 upper_body_ik + ZMQ 紧耦合，工作量数倍）
- [ ] mocap 切换 / 多动作支持
- [ ] target_vel → planner_sonic.onnx → motion ref（命令驱动接口，B 路径）

#### 长期目标（不在本阶段范围）
- [ ] VLA 集成（需 `gear_sonic[inference]` + Isaac-GR00T 客户端）
- [ ] 任务微调（需 30 GB SMPL + 64+ GPU）
- [ ] 部署到真实 G1（C++ + TensorRT）

### 关键状态判断

整个 pipeline 的"地基"已经牢固：ONNX 加载、994D decoder 完整观测、1762D encoder layout、mocap 加载与 anchor 信号、物理验证、反馈循环已破。**核心剩余卡点只有一个：body_pos FK**。FK 解决后预期 action 数值范围进一步降低 + 行走动作真正可识别，整个 pipeline 进入"训练分布内的真实推理"。

下一步推荐：**F4（gear_sonic 自带 torch FK）**——见下方"BONES-SEED + F4 调研"。原 F2 独立 IsaacSim 预算脚本降级为备选。

---

## BONES-SEED 与 F4 方案调研（2026-05-24）

### BONES-SEED 数据集判断

调研背景：知识库 [[BONES-SEED数据集]] 笔记暗示 motion_lib 目录格式自带 `body_pos.csv` + `body_quat.csv`，可能直接解 body_pos FK 卡点。

**实测核实结论：BONES-SEED 不直接解 FK 卡点**。

| 路径 | 真相 |
|---|---|
| motion_lib 三件套（joint_pos / body_pos / body_quat.csv） | 是 SOMA retargeter 的**输出**（已 FK 过），不是 BONES-SEED 原始格式 |
| BONES-SEED `g1/csv/` 平 CSV 文件 | 只有 `Frame + 6 root cols + 29 joint angles`，**没有 body_pos** |
| 转换器 [convert_soma_csv_to_motion_lib.py:209-214](D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py#L209-L214) | `body_pos_w` 写成 zeros 占位，只填了 root；输出 PKL 的 `smpl_joints` 也是 zeros |

→ BONES-SEED CSV 与当前 sample PKL 完全同款问题：dof 真、body_pos 假。**单靠 BONES-SEED 仍需 FK**。

### BONES-SEED 的真实价值（中长期，非当前阶段）

| 时间窗 | 用途 | 紧迫度 |
|---|---|---|
| 当前阶段 | 不解 FK 卡点 | ⛔ 不下载 |
| F4 落地后 | `g1.tar.gz`（估 10-30GB）→ 转换器 → F4 补 body_pos → 74K locomotion 序列替代当前 1 个 walking | 🟡 锦上添花 |
| 语言条件控制 | `seed_metadata_v002_temporal_labels.jsonl` 6 万行时序文本标注 → 替代 planner_sonic.onnx 的 B 路径 | 🟡 命令驱动可选路 |
| SONIC 微调 | `gear_sonic[training]` 路径必需训练数据源（含 SMPL 30GB） | 🟢 当前阶段外 |

### F4 方案：gear_sonic 自带纯 torch FK

调研意外发现 `Humanoid_Batch.fk_batch()` 是纯 PyTorch G1 FK，**无 pinocchio 依赖**——绕开了之前 F1/F2/F3 三选一的痛点。

[torch_humanoid_batch.py:360 `Humanoid_Batch.fk_batch()`](D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic/utils/motion_lib/torch_humanoid_batch.py#L360)：

| 项 | 值 |
|---|---|
| 输入 | `pose (B, T, 30, 3) axis-angle` + `trans (B, T, 3)` + fps |
| 输出 | `wbody_pos (B, T, N_bodies, 3) 世界坐标` + `wbody_rot` |
| 依赖 | torch + numpy + scipy + lxml + open3d（顶部 import；fk_batch 本身是否用到待确认） |
| MJCF | `gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml`（仓库自带） |
| 当前 env_isaaclab 缺 | `lxml`（必装，wheel 几 MB） + `open3d`（500MB+，可能可 stub） |

### 落地步骤（F4，2026-05-24 已全部完成）

1. **依赖安装** ✅：`D:/miniconda3/envs/env_isaaclab/python.exe -m pip install lxml`（lxml-6.1.1，4 MB）。`open3d` 不装——验证发现它只在 `torch_humanoid_batch.py` line 793 `o3d.io.read_triangle_mesh` 和 line 873（注释掉的 write）用到，**fk_batch 路径完全不依赖**。
2. **预算脚本** ✅：[scripts/tools/precompute_mocap_body_pos.py](../scripts/tools/precompute_mocap_body_pos.py)
   - `_stub_open3d()`：把 `open3d` / `open3d.io` 注册为返回 None 的 stub module
   - `Humanoid_Batch.load_mesh = lambda self: None`：monkey-patch 跳过 mesh 加载（fk_batch 不需要）
   - 配置 MJCF 路径 `gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml`
   - 输入：sample PKL 的 `pose_aa (T, 30, 3)` axis-angle + `root_trans_offset (T, 3)`
   - `fk_batch(pose, trans, fps=30)` → `global_translation (1, T, 30, 3)` 世界系
   - `SONIC_BODY_NAMES` 14 个名字 resolve 到 `Humanoid_Batch.body_names` indices `[0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29]`（注意：与 IsaacLab USD 的 indices `[0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29]` **不同**，因为 Humanoid_Batch 的 body order 是 MJCF 树遍历，与 USD 不一致——这正是按 name resolve 而非 index 透传的原因）
   - 减 pelvis (index 0) → pelvis frame → 落 `walk_forward_amateur_001__A001__body_pos14_pelvis.npy`（**197 KB**）
   - 实测 1202 帧 pelvis-frame body_pos absmax = **0.7531**，与 self-ref 的 0.6822 量级一致
3. **`SONICWholeBodyAction._load_mocap()` 集成** ✅：自动检测 `{pkl_stem}__body_pos14_pelvis.npy`，加载到 `self._mocap_body_pos_b`；shape 不匹配则 fallthrough SMPL，最后 self-ref。优先级 = **F4 .npy > SMPL approx > self-ref**。
4. **headless smoke test** ✅：`sonic_verify.py --headless --max_steps 60` 日志确认：
   ```
   loaded mocap from ... body_pos=F4 FK npy (absmax=0.753) @ walk_forward_amateur_001__A001__body_pos14_pelvis.npy
   body indices resolved: 14/14
   step=50  action mean=-2.2038 absmax=7.6925 std=2.7820 | joint_pos absmax=1.6144
   ```
5. **GUI 视觉验证 + 反馈循环回潮**（2026-05-24 实测）

### F4 落地后数值变化 + GUI 实测

| 状态 | action absmax | action std | joint_pos absmax |
|---|---|---|---|
| 阶段 3.3 self-ref body_pos | 1.65~3.02 | 0.58~0.83 | 1.34 |
| 阶段 3.4 F4 headless step 50 | 7.69 | 2.78 | 1.61 |
| **阶段 3.4 F4 GUI step 300 峰值** | **18.58** | **5.46** | **1.97** |

**GUI 全 step 演化**（30 秒，500 帧）：

| step | action absmax | std | joint_pos absmax | 趋势 |
|---|---|---|---|---|
| 50 | 7.69 | 2.78 | 1.61 | — |
| 100 | 10.08 | 3.04 | 1.97 | ↑ 撞限位 |
| 150 | 10.92 | 3.55 | 1.86 | ↑ |
| 200 | 10.06 | 3.54 | 1.96 | 停顿 |
| 250 | 12.72 | 4.85 | 2.13 | ↑↑ |
| **300** | **18.58** | **5.46** | 1.97 | **峰值** |
| 350 | 12.41 | 3.49 | 1.97 | 抖动 |
| 400 | 12.38 | 4.32 | 1.97 | 抖动 |
| 450 | 10.48 | 3.98 | 1.61 | 抖动 |

**用户眼测：三种现象都有**（部分像 walking 腿摆动 + 部分关节抽搐 + 部分立刻摔倒）。

**诊断（反馈循环占主导）**：
- joint_pos absmax 持续黏在 1.97 ≈ G1 大多数关节限位 **2.0**，意味着关节被打到极限
- 极端姿态进 10-frame history → decoder 看到 OOD 输入 → action absmax 持续放大
- 这与阶段 3.2 D1 v2 的 garbage 状态同质（当时也是 absmax 12+），只是现在 mocap 信号比 self-ref 提供了部分有效追踪（→ 三种现象并存）

**第一轮修复**（[locomanipulation_g1_env_cfg.py:298](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/locomanipulation_g1_env_cfg.py#L298)）：
- `action_scale: 0.2 → 0.05`
- 峰值偏移：18.58 × 0.05 = **0.93 rad / 单帧**，避开关节限位

### 阶段 3.4 GUI 第二轮（2026-05-24 实测，scale=0.05）

| step | action absmax | std | joint_pos absmax |
|---|---|---|---|
| 50-300 | 3.2~3.5 稳定 | 1.15 稳定 | 0.19~0.54 |

✅ **反馈循环已切断** + **关节远离限位**。
❌ **但腿部不动 + 摔倒**：action_scale 0.05 × 3.5 ≈ 0.175 rad/帧 ≈ 10°/帧追不上 30fps walking 的髋膝 30~60° 摆动。robot 由重力被动摔倒（self_ref_body_pos 0.72 是 ragdoll 倒下不是有意 walking）。

scale 在 **[0.05 太保守不动 / 0.2 反馈爆炸]** 没有 stable 中间区 → 说明 SONIC 收到的 mocap 信号本身就 OOD，scale 调参治标不治本。

### 阶段 3.4 FK rotation 修复（2026-05-24，验证为非根因）

候选根因 1：**FK pelvis frame 只去了 translation 没去 root rotation**，14 body 残留 world rotation 让 SONIC 看到 OOD body_pos。

[precompute_mocap_body_pos.py:115-130](../scripts/tools/precompute_mocap_body_pos.py#L115-L130) 修复：
```python
rel_w = body14_t - pelvis_t            # 减 translation
pelvis_R_T = pelvis_R.transpose(-1,-2) # pelvis world rotation transpose
rel_b = einsum("tij,tnj->tni", pelvis_R_T, rel_w)  # 旋到 pelvis local frame
```

**实测**：`world-rel absmax=0.7531 → pelvis-local absmax=0.7552`（差 0.3%）。

**结论**：当前 sample mocap 是直线行走（`root_rot absmax=0.9996` ≈ identity quaternion），pelvis world R ≈ I，去不去 rotation 数值几乎一样。**rotation 修复本身正确**（未来转弯 mocap 必须用），但**不是当前问题的根因**。

### 阶段 3.4 上游验证工具（mocap_playback.py，2026-05-24）

scale 调参无效 + rotation 修复无效，需要判定责任在 **mocap 数据上游** 还是 **SONIC 调用下游**。

[scripts/tools/mocap_playback.py](../scripts/tools/mocap_playback.py)：绕过 SONIC 直接 kinematic 播放 mocap dof → 看是否合理 walking。

- 复用 sonic_verify 的 env 启动（保留场景 + sonic_robot），但 step loop 里跳过 `env.step()`，直接 `asset.write_joint_state_to_sim(mocap_dof) + asset.write_root_pose_to_sim(mocap_root) + sim.step()`
- 关键事实：mocap PKL 的 `dof (T, 29)` 顺序 = MJCF actuator order = **与 `SONIC_G1_29DOF_JOINT_ORDER` 完全一致**（确认自 [convert_soma_csv_to_motion_lib.py:130-162 `BONES_CSV_JOINT_NAMES`](D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic/data_process/convert_soma_csv_to_motion_lib.py#L130-L162)），可直接逐帧覆写
- `frame_step=0.6`（30fps mocap → 50Hz sim），匹配真实速度
- headless smoke test 通过：dof absmax=1.692、root_trans Y 方向位移 8.19m（向前走 8 米）= 数值层面 mocap 数据合理

**判定三分支**：

| GUI 现象 | 含义 | 下一步 |
|---|---|---|
| 腿前后摆 + 整体向前位移 ~8m 流畅 walking | mocap 数据 OK，问题在 SONIC 调用层（action 应用方式 / 帧率 / 时间窗 / 字段 unit） | 查 [sonic_release/config.yaml](D:/src/Isaac/GR00T-WholeBodyControl/sonic_release/config.yaml) 训练 action term 定义 |
| 关节穿模 / 抖动 / 不像 walking | mocap 数据本身坏（关节单位 / 顺序 / retarget 错） | mocap 数据是死路，需要别的 walking 源（planner_sonic.onnx 或 BONES-SEED 重新 retarget） |
| 卡在原地 / 关节几乎不变 | dof 字段是 placeholder（virtual） | 看具体哪个 joint 在变 + 切别的 mocap 序列 |

**用户实测**：✅ **腿前后摆 + 整体向前位移 ~8m 流畅 walking**（2026-05-24）= **mocap 数据上游完全 OK**，责任明确在 SONIC 调用下游。进入"下游修复 1"。

### 阶段 3.4 下游修复 1：训练配置查证 + 时间窗对齐（2026-05-24）

从 [sonic_release/config.yaml](D:/src/Isaac/GR00T-WholeBodyControl/sonic_release/config.yaml) 提取 SONIC 训练时的精确参数：

```yaml
actions:
  joint_pos:
    _target_: isaaclab.envs.mdp.actions.JointPositionActionCfg
    joint_names: [".*"]
    use_default_offset: true          # → target = default + scale × raw_action
    # scale 未显式 → 默认 1.0

config:
  action_clip_value: 20.0             # raw_action 硬 clip 上限
  actor_actions_history_length: 10

commands.motion:
  target_fps: 50                       # SONIC 训练 sim 50Hz
  dt_future_ref_frames: 0.1            # future ref 帧间隔 = 0.1 秒
  num_future_frames: 10                # 取未来 10 帧 = 跨度 1.0 秒
  encoder_sample_probs: {g1: 1, teleop: 1, smpl: 1}   # 三 mode 等概率训练
  motion_lib_cfg:
    motion_file: data/motion_lib_bones_seed/robot_filtered/  # 与我们 sample 同源
```

#### 已查证的对齐项 ✅

| 项 | 训练 | 我们 | 状态 |
|---|---|---|---|
| action 公式 | `default + 1.0 × raw` | `default + 0.05 × raw` | ✅ 公式一致，scale 差异详见下方 |
| `use_default_offset` | True | 等价（`policy_output_offset = default`） | ✅ |
| `actor_actions_history_length` | 10 | 10 | ✅ |
| encoder mode_id | 0 (g1) | 0 (g1) | ✅ |

#### 发现的三处偏差 ❌ → 已修

| 项 | 训练 | 修前（我们） | 修后 |
|---|---|---|---|
| **dt_future_ref_frames** | 0.1s（50Hz × 5 sim step） | 5 mocap 帧 = 5/30 = **0.167s** | `_mocap_step = round(0.1 × fps)` = **3 mocap 帧 = 0.1s** ✅ |
| **mocap 推进速率** | 50Hz sim 同步 mocap_lib (target_fps=50) | 1 mocap 帧 / sim step（30fps mocap 当 50fps 播 = 1.67× 慢） | `_mocap_advance_per_step = mocap_fps / 50.0` = **0.6 frame / sim step** ✅ |
| **reset 时 mocap 指针** | 重新采样新动作 | 没重置（继续上次位置） | `_mocap_frame = 0; _mocap_frame_f = 0.0` ✅ |

代码位置：[mdp/actions.py `_load_mocap` / `_advance_mocap` / `reset`](../source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/mdp/actions.py)。

#### action_scale 偏差备忘

我们当前 `action_scale = 0.05`，训练是 `1.0`（差 20 倍）。理论上应等于训练值，但实测 scale=1.0 / 0.2 都会反馈循环爆炸（absmax 18+）。说明 SONIC 真实推理 raw_action 在我们环境里就比训练分布大几倍。先保留 0.05 看 GUI 时间窗修复后视觉，若开始 walking 再慢慢往 0.1~0.2 加。

#### headless smoke 验证

修后 step=50: `action absmax=3.76 std=1.13 joint_pos absmax=0.53`，数值与修前同量级（数值层面 dt_future 改动差异小，关键看 GUI 视觉是否出现 walking 动作）。

#### 待 GUI 再验

| GUI 现象 | 含义 | 下一步 |
|---|---|---|
| 出现 walking 腿前后摆动（即使最后摔） | dt_future + advance 是真根因 | 调 scale 0.05 → 0.1~0.2 加幅度 |
| 仍只是站着 / 缓慢倒 | 时间窗不是关键，查 obs commands.motion 其它字段（teleop / smpl 字段我们 zero） | 接下游修复 2 |
| 抽搐 / 爆炸 | _mocap_step=3 取太密或 advance 太快 | 回 step=5 或 advance=0.3 |

### 阶段 3.4 GUI 第三轮（dt_future 修后，2026-05-24）

**用户实测**：仍然只是站着 / 缓慢倒，**手部有动作但腿部没有动作**。

#### 诊断打印（每 50 step 加输出 4 段 absmax + 一次性打印 SONIC index → USD joint name 映射）

| step | legs[0:12] | waist[12:15] | l_arm[15:22] | r_arm[22:29] | 总 absmax |
|---|---|---|---|---|---|
| 50 | 2.26 | 2.60 | 2.36 | 3.76 | 3.76 |
| 100 | 1.57 | 1.59 | 1.58 | 2.35 | 2.35 |
| 150 | 1.56 | 1.73 | 2.22 | 2.65 | 2.65 |
| 200 | 1.84 | 2.00 | 2.79 | 3.28 | 3.28 |

**关键发现**：
- joint mapping **完全正确**（SONIC[0]→USD[0]=left_hip_pitch, SONIC[12]→USD[2]=waist_yaw, ...）
- SONIC **确实在给腿部输出动作**，量级与手部相当
- scale=0.05 × 2 ≈ 5-7° / 关节：手部 5-7° = 整只手摆十几 cm 看得见；**腿部 5-7° = 大腿微动几 cm + 承重摩擦 = 看起来不动**
- walking 真正需要的髋/膝幅度 30-60°，scale=0.05 远远不够

#### scale 调参的反馈循环陷阱（headless smoke 实测）

| action_scale | step 50 absmax | step 200 absmax | 状态 |
|---|---|---|---|
| 0.05 | 3.76 | 3.28 | ✅ 稳定，但腿幅度不够 |
| 0.15 | **10.59** | **14.56** | ❌ 反馈循环爆炸 (joint_pos 撞限位 1.97) |
| 0.20 | 7.69（早期）| 18.58 峰值 | ❌ 反馈循环爆炸（与之前同） |

时序修复 (dt_future + advance) **没有消除反馈循环**。scale 提到 0.15 就爆。

#### 真正根因诊断（reset 时 robot 与 mocap 初始姿态不同步）

| 项 | 训练时 | 我们 |
|---|---|---|
| reset 后 robot joint_pos | `motion_lib` 把 robot 设为 `mocap.dof[0]`（与 mocap 同步） | `default_joint_pos`（站立） |
| reset 后 robot root_pose | `mocap.root_trans[0]` + `mocap.root_rot[0]` | scene init_state 默认 |
| 第一帧 SONIC 看到的 obs | "robot 当前姿态 ≈ mocap 当前帧" → raw_action ≈ 0 | "robot 静止站立 + mocap 在 walking" → raw_action 想追 mocap → 大 |
| obs.joint_pos rel | raw 本身（scale=1.0） | scale × raw（永远小于训练分布） |

**反馈循环机制**：robot 与 mocap 初始姿态不同步 → SONIC 输出大 raw 想追上 → scale 小则 obs 反馈"我没动多少" → SONIC 输出更大 raw → 爆炸。

### 阶段 3.4 下游修复 2 落地（2026-05-24）

实施：
1. `_load_mocap` 缓存 `_mocap_dof (T, 29)` + `_mocap_root_trans (T, 3)`
2. `_load_mocap` 把 `_mocap_root_rot_wxyz` 整体减去 root_rot[0]，使 `aligned[0] = identity`（消掉初始 yaw -88°）
3. `_sync_robot_to_mocap_frame0()`：
   - `asset.write_joint_state_to_sim(mocap_dof_0, joint_ids)` ✅ post-write actual absmax = 1.692 与 mocap.dof[0] 一致
   - `asset.write_root_pose_to_sim([current_root_pos, mocap_rot0 = identity])`
   - `asset.write_root_velocity_to_sim(zero)`
   - `_processed_actions = mocap_dof_0`
4. `reset()` 调 `_sync_robot_to_mocap_frame0()`
5. `action_scale 0.15 → 1.0`（训练值）

### 阶段 3.4 GUI 第四轮（reset 同步 + scale 1.0 后，headless 实测）

| step | absmax | std | joint_pos absmax | legs[0:12] |
|---|---|---|---|---|
| 1 | **2.56** | 1.01 | 1.69 | 1.84 |
| 3 | 5.93 | 2.01 | 1.08 | 5.42 |
| 5 | 10.10 | 3.24 | 1.34 | 6.29 |
| 10 | 11.25 | 3.90 | 2.03 | 8.97 |
| 20 | 17.67 | 6.49 | 3.09 | 11.89 |
| 50 | 16.49 | 4.67 | 2.61 | 6.67 |

**部分成功**：step 1 raw=2.56 ≈ mocap.dof.absmax=1.69（合理量级），证明 reset 同步把"起点"修对了。

**仍失败**：step 3 → 20 仍持续放大到 17.67，joint_pos 撞限位 3.09。反馈循环没消除。

**对照表**（不同修复阶段 step 1 / step 50 absmax）：
| 状态 | step 1 | step 50 | step 200 |
|---|---|---|---|
| 阶段 3.3 self-ref scale=0.2 | — | 7.69 | 18.58 |
| 阶段 3.4 mocap body_pos scale=0.15 | — | 10.59 | 14.56 |
| 阶段 3.4 + reset 同步 + history=mocap[0] | — | 18.47 | 23.26 |
| **阶段 3.4 + reset 同步 + history=zero + yaw align** | **2.56** | 16.49 | (爆炸) |

### 阶段 3.4 已尽快速修复路径，剩余可疑根因

reset 同步 + 时间窗 + FK 都做了，**反馈循环仍未消除**。剩余可疑（按代价从小到大）：

| 候选 | 可疑度 | 检验代价 |
|---|---|---|
| **actuator stiffness/damping 不匹配训练** | 高 | 看 [gear_sonic/.../robots/g1.py:258-275](D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/robots/g1.py#L258) ImplicitActuator NATURAL_FREQ=10Hz vs IsaacLab DCMotor stiffness=100 — 改 G1_29DOF_CFG 用同款 ImplicitActuator |
| **sim_dt 不匹配训练** | 中 | 训练用 50Hz control，physics 通常 200-400Hz。我们 IsaacLab 默认未知。改 sim dt |
| **base_ang_vel / gravity_dir 坐标系不匹配** | 中 | 训练 obs func 在 [gear_sonic/.../observations.py](D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/mdp/observations.py)，需要看每个字段的 frame 转换 |
| **mocap PKL 没经过 motion_lib.target_fps=50 重采样** | 中 | 训练时 motion_lib 把 mocap 重采样到 50fps；我们用 30fps 原始数据。需要预先重采样 |
| **PyTorch checkpoint vs ONNX 输出有差异** | 低 | 试用 sonic_release/last.pt 验证 |

### 阶段 3.4 当前状态总结

- ✅ Pipeline 完整（ONNX 加载、994D decoder obs、1762D encoder layout、mocap FK body_pos、时序对齐、reset 同步）
- ✅ joint mapping 正确（SONIC[i] → USD joint 验证通过）
- ✅ 上游 mocap 数据合格（mocap_playback kinematic 验证）
- ❌ **闭环 SONIC 推理仍反馈循环爆炸**，最可能原因是 **actuator stiffness/damping 不匹配训练**（我们 IsaacLab DCMotor stiffness=100，训练用 ImplicitActuator stiffness=99 + armature=0.025 + 不同 PD 配方）

### 阶段 3.4 决策：选 C 收尾（2026-05-24）

用户决策：**接受当前为半成品停下**，A / B 写入下方"后续 TODO"，本分支阶段 3.4 工作收尾。

理由：闭环反馈循环根因深入 actuator/sim_dt/obs 单位等多层，每个候选都需小时级深挖且不保证解决；当前 pipeline 地基已牢，足以作为 SONIC 微调或后续切入点。

### 后续 TODO（不在本阶段执行）

#### A. actuator 配置匹配训练（最高优先）
- 改 [G1_29DOF_CFG](../source/isaaclab_assets/isaaclab_assets/robots/unitree.py#L388) actuator：从 IsaacLab `DCMotor` 切换到 `ImplicitActuator` + SONIC 训练同款 PD 配方
- 参考 [gear_sonic/.../robots/g1.py:10-26](D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic/envs/manager_env/robots/g1.py#L10-L26)：
  - `NATURAL_FREQ = 10 * 2π = 62.83 rad/s`，`DAMPING_RATIO = 2.0`
  - 各 actuator armature：5020=0.0036、7520_14=0.0102、7520_22=0.0251、4010=0.00425
  - stiffness = armature × NATURAL_FREQ²、damping = 2 × DAMPING_RATIO × armature × NATURAL_FREQ
- 注意：需要给 sonic_robot 独立的 ArticulationCfg（不动 robot/walker_robot/remote_robot 的 G1_29DOF_CFG），通过 `cfg.actuators` 覆盖
- 预期工作量：1-2 小时

#### B. 接 motion_lib 数据加载 + PyTorch ckpt 验证
- 安装 gear_sonic [training] extras（trl / accelerate / smpl_sim）
- 用 motion_lib 真正的加载链路替换我们的 joblib + 手 FK pipeline（与训练完全对齐）
- 试用 [sonic_release/last.pt](D:/src/Isaac/GR00T-WholeBodyControl/sonic_release/last.pt) PyTorch checkpoint 替换 ONNX dual-pass，验证两者数值是否一致
- 预期工作量：3-4 小时

#### C. SONIC 微调（远期）
- 用 BONES-SEED `g1.tar.gz`（10-30GB）+ `gear_sonic[training]` 在我们的 task 环境下微调 SONIC checkpoint
- 需先解空间约束（D 盘清出 30+GB）+ GPU 时长预算
- 远期目标，不在本分支范围

#### D. 把 SONIC 接管 robot / remote_robot（原始意图）
- 当前 SONIC 作为第 4 个机器人 sonic_robot 平级落地，原始意图是替换 robot/remote_robot 的 AgileBasedLowerBodyAction
- 需先剥离 upper_body_ik + ZMQ 紧耦合（__post_init__ 里直接 self.actions.upper_body_ik.controller.urdf_path = ...）
- 取决于 A / B 是否让 sonic_robot 真正能 walking

### 当前空间约束（2026-05-24 实测）

- D 盘剩余 **15.3 GB**、C 盘剩余 **13.1 GB**
- F4 实际开销（已落地）：lxml 4 MB + .npy 197 KB ≈ **< 5 MB**（open3d stub 省了 500 MB）
- BONES-SEED `g1.tar.gz`：估 10-30 GB → **当前不可承受**
- BONES-SEED 完整 114 GB → **完全超出**

→ F4 已经把 FK 卡点解掉；BONES-SEED 留到清理出 30+ GB 后再考虑。

---

## 参考文献

- [Building Generalist Humanoid Capabilities with NVIDIA Isaac GR00T N1.6](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-GR00T-n1-6-using-a-sim-to-real-workflow)
- [SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control](https://nvlabs.github.io/SONIC/)
