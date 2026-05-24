# GR00T-WholeBodyControl 集成计划

> **进度：阶段 1（环境准备）+ 阶段 2（最小骨架）+ 阶段 3.1（真实 decoder 观测）✅ headless 全部通过于 2026-05-23**
> 仓库已克隆、`gear_sonic` core 已装入 `env_isaaclab`、GEAR-SONIC ONNX + PyTorch ckpt 已下载、`SONICWholeBodyAction` 骨架已落地为场景中的第 4 个机器人 `sonic_robot`、994D decoder 端按真实 10-frame history 拼装（encoder 仍 zero-fill），均以 `sonic_verify.py --headless --max_steps 200` 实测 200 帧零错。GUI 视觉变化眼测尚未做。详见下方"阶段 1 / 阶段 2 完成纪要"。

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
     - 待验证：absmax 应被压制（2.6 × 0.2 = 0.52 rad），但 self-ref 仍是 OOD 假设，最终必须接 mocap

     **下一步路径**：
     - **优先验证修复**：跑 sonic_verify，看 absmax 是否被压制 + GUI 是否减弱关节扭曲
     - **E3 接 mocap**（紧接其后）：`python download_from_hf.py --sample`（4MB 含 walking 序列），加载真实 mocap 帧替代 self-ref，回归 SONIC 训练范式

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

## 参考文献

- [Building Generalist Humanoid Capabilities with NVIDIA Isaac GR00T N1.6](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-GR00T-n1-6-using-a-sim-to-real-workflow)
- [SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control](https://nvlabs.github.io/SONIC/)
