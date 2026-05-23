# GR00T-WholeBodyControl 集成计划

> **进度：阶段 1（环境准备）✅ 完成于 2026-05-23**
> 仓库已克隆、`gear_sonic` core 已装入 `env_isaaclab`、GEAR-SONIC ONNX + PyTorch ckpt 已下载。详见下方"阶段 1 完成纪要"。

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

### 阶段 2 前置依赖（**必装**）
- `onnxruntime`（或 `onnxruntime-gpu`）——推理 ONNX 必需，当前未安装

### 已跳过
- 30 GB Bones-SEED SMPL 数据（仅训练用）
- `gear_sonic[training]` 的 trl / accelerate / smpl_sim（仅训练用）
- `gear_sonic[inference]` 的 Isaac-GR00T VLA 客户端（VLA 场景再装）

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

2. **阶段 2：骨架代码（待启动）**
   - [ ] 安装 `onnxruntime`（或 `onnxruntime-gpu`）到 env_isaaclab
   - [ ] 核对 SONIC G1 29 DoF 关节顺序 vs `G1_29DOF_CFG`，建立映射表
   - [ ] 在 `mdp/actions.py` 创建 `SONICWholeBodyAction`，先用**固定姿态**作 motion reference（不动 planner），验证 ONNX 推理跑通 + 关节写入正确
   - [ ] 在 `configs/action_cfg.py` 创建 `SONICWholeBodyActionCfg`，参考 [GEAR-SONIC observation_config.yaml](D:/src/Isaac/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/observation_config.yaml)
   - [ ] 解除 `robot` / `remote_robot` 的 `fix_root_link`，在新环境配置里启用 SONIC

3. **阶段 3：motion reference 对接**
   - [ ] 接入 `planner_sonic.onnx`（target_vel → motion 帧），实现"速度命令 → SONIC 行走"
   - [ ] 替代方案：mocap 文件 / ZMQ pose 流

4. **长期目标**：
   - [ ] 集成 VLA 功能（需补 `gear_sonic[inference]`）
   - [ ] 针对特定任务微调（需补 30 GB SMPL + `gear_sonic[training]`，64+ GPU 训练）
   - [ ] 部署到真实 G1 机器人（C++ 推理栈 + TensorRT）

## 参考文献

- [Building Generalist Humanoid Capabilities with NVIDIA Isaac GR00T N1.6](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-GR00T-n1-6-using-a-sim-to-real-workflow)
- [SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control](https://nvlabs.github.io/SONIC/)
