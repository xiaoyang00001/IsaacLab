# GR00T-WholeBodyControl 目标文档

> 目标：让 G1 人形机器人在 IsaacLab 中用 SONIC AI 政策自主行走。

## 最终目标

G1 机器人在 IsaacLab 仿真环境中，通过 SONIC AI 控制器实现稳定的步行运动。

**成功标准**：
- 机器人能持续行走超过 10 秒（500+ 步）不摔倒
- 关节角度不超出安全限位
- 行走姿态接近 mocap 运动捕捉数据

---

## 技术架构

```
mocap 运动数据（walking 序列）
    ↓
encoder (1762D → 64D token)
    ↓ + decoder history (10帧)
decoder (64D token + 994D history → 29D 关节偏移)
    ↓ × action_scale
绝对关节目标 → 写入 G1 29个关节
    ↓
机器人物理响应
    ↓
新关节位置/速度 → 反馈到 decoder history
```

---

## 现状（2026-05-25）

**已验证通过**：
- ONNX dual-pass 推理 pipeline（encoder + decoder）
- 994D decoder 观测字段（joint_pos/vel、last_action、gravity 等 10 帧 history）
- mocap 运动数据加载 + 50fps 时钟同步
- 关节映射（G1 USD 29 joints ↔ SONIC 29DOF）
- actuator PD 配置（与 SONIC 训练对齐）

**卡点**：
- step 3+ 后 action absmax 从 ~2.5 爆炸到 22+
- 根因：**decoder history（特别是 his_last_actions）与训练分布不匹配**
  - 训练时：`Normal(mean, std).sample()` — stochastic action
  - 推理时：ONNX 直接输出 deterministic mean
  - 10 帧 history 累积误差 → decoder OOD → garbage 放大

---

## 修复路径

### B2b（正在进行）：action noise 注入
在 ONNX 输出的 29D action 上叠加 `Normal(0, per_joint_std)`，模拟训练时 stochastic policy。
- 已实现：ckpt 提取 per-joint std (29,) → npy
- 效果：腿部有动作，但 r_arm (index 25-28) absmax 仍 17-20

### B3（下一步）：motion_lib 随机 reset
reset 时把机器人对齐到 mocap 随机帧（不是固定第 0 帧），与训练时的随机初始化对齐。

### C（接受现状）：
sonic_robot 作为 SONIC 微调起点，不强求现在 walk。pipeline 地基已牢。

---

## 当前分支状态

- `gr00t-sonic-bodypos-probe`：探针测试分支（含 B2b action noise 实现）
- `sonic_release/last.pt`：PyTorch checkpoint（微调用，已可加载）
- `gear_sonic_deploy/policy/release/`：ONNX 部署模型

---

## 快速参考

| 文件 | 用途 |
|------|------|
| `scripts/tools/sonic_verify.py` | headless 验证脚本 |
| `scripts/tools/compare_pytorch_vs_onnx.py` | PyTorch vs ONNX 对比 |
| `scripts/tools/extract_sonic_action_std.py` | 从 ckpt 提取 per-joint std |
| `source/.../mdp/actions.py` | SONICWholeBodyAction 实现 |
| `source/.../configs/action_cfg.py` | SONICWholeBodyActionCfg 配置 |