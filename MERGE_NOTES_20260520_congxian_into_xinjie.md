# Merge 记录：`congxian` → `xinjie`（2026-05-20）

## 背景

- 本地分支：`xinjie`（个人开发分支）
- 远端目标：`origin/xinjie`（最新的工作状态，包含 15 个个人提交）
- 待合并：`congxian`（同事分支，一个大的 squash commit `Sync current 40.36 IsaacLab version`）
- Merge base：`4df6560e1 docs: add NCCL troubleshooting notes for multi-GPU training (#5195)`

## 准备工作

发现本地 `xinjie` 与 `origin/xinjie` 分歧严重：

- 本地多 3 个提交（openxr / ZeroMQ 网络配置 refactor，疑似与远端的 15 个提交重复）
- `origin/xinjie` 多 15 个提交（传送带、多机器人配置、IP 自动检测等新功能）

**处理方式：**

1. 备份本地状态：`git branch xinjie-local-backup-20260520 xinjie`
2. 重置到远端：`git reset --hard origin/xinjie`
3. 从干净的 `origin/xinjie` 出发 merge `congxian`

> 备份分支 `xinjie-local-backup-20260520` 保留了本地多出的 3 个 openxr 提交，如有需要可恢复。

## Merge 结果

- 合并提交：`e5b51e2c2 Merge branch 'congxian' into xinjie`
- 共 23 个文件改动，3836 行新增，175 行删除
- 8 个冲突文件，全部已手动解决

## 冲突解决明细

### 1. `source/isaaclab_tasks/.../pick_place/mdp/__init__.py`

仅空行差异。保留 HEAD（xinjie）版本。

### 2. `scripts/tools/record_demos.py`（5 处冲突）

**最复杂的冲突，两边都改了同一片逻辑。**

| 冲突块 | xinjie 内容 | congxian 内容 | 解决方式 |
|---|---|---|---|
| `import h5py` | 无 | 在 AppLauncher 之前导入 h5py，避免 conda DLL 冲突 | **取 congxian**（合理的环境修复） |
| `MultiTeleopDevice.__init__` | 仅 `devices: list` | 增加 `device_names` 参数 | **取 congxian**（扩展能力） |
| `advance_primary_robot_only` 方法 | 无 | 新增方法，按设备名优先路由到主机器人 | **取 congxian**（多机器人支持） |
| `MultiTeleopDevice(...)` 实例化 | `(interfaces)` | `(interfaces, valid_devices)` | **取 congxian** |
| 主循环 action 处理 | 中性姿态填充未激活的 IK 动作组（修复 PhysX AABB 错误） | 检测 MultiTeleopDevice → 调用 `advance_primary_robot_only` → 双机器人时补零给从机器人 | **两者合并** ⭐ |

合并后的主循环逻辑：

```python
if isinstance(teleop_interface, MultiTeleopDevice):
    primary_action = teleop_interface.advance_primary_robot_only()
    total_action_dim = int(env.action_space.shape[-1])
    primary_action_dim = int(primary_action.shape[-1])
    if total_action_dim == primary_action_dim * 2:
        zero_remote_action = torch.zeros_like(primary_action)
        action = torch.cat((primary_action, zero_remote_action), dim=-1)
    else:
        action = primary_action
else:
    action = teleop_interface.advance()

# 之后接 xinjie 的中性姿态填充/裁剪逻辑
expected_dim = env.action_manager.total_action_dim
if action.shape[-1] < expected_dim:
    # 用中性手臂姿态填充避免 PhysX AABB 退化
    ...
```

### 3. `source/isaaclab/.../openxr/openxr_device.py`

- xinjie：`ZeroMqGameClient.init(self._cfg.zmq_game_server_endpoint, self._cfg.zmq_player_id)`
- congxian：`ZeroMqGameClient.init("tcp://192.168.40.30:14026", 1)`（硬编码）

**取 xinjie**（配置化版本更可维护）。

### 4. `source/isaaclab/.../trihand/g1_upper_body_zeromq_retargeter.py`

- xinjie：调试 `print` 已注释
- congxian：调试 `print` 启用中

**取 xinjie**（减少日志噪音）。

### 5. `source/isaaclab/.../openxr/zeromq_game_client.py`（3 处冲突）

xinjie 加了完整的调试日志：

- 初始化时 print endpoint + player_id
- 连接成功 / 失败 print + logger
- send 成功一次后静默，send 失败一次后静默（避免日志泛滥）

congxian 是更早的简版日志。

**统一取 xinjie**（对应 commit `cb17602c0 feat(openxr): 添加ZeroMQ客户端和服务设备的调试日志`）。

### 6. `source/isaaclab/.../openxr/zeromq_game_sub_device.py`（11 处冲突）

冲突贯穿全文件，xinjie 整体改进幅度大：

- 加入 XRCore 消息总线订阅以支持 `send_teleop_command("start/stop/reset")`
- 大量调试日志（初始化、连接、第一条数据、每 200 条打印一次、控制器数据变化时打印）
- 配置项默认值改为延迟读取 `NETWORK_CFG`（避免循环导入）

**直接 `git checkout --ours`** 取 HEAD 全版本。

### 7. `source/isaaclab_tasks/.../pick_place/locomanipulation_g1_env_cfg.py`（9 处冲突）

差异点：

| 主题 | xinjie | congxian |
|---|---|---|
| `ZMQ_SYNC_ROLE` / 资产名 | 来自 `NETWORK_CFG` | 硬编码 `"subscriber"` |
| `FIXED_G1_29DOF_CFG` 初始姿态 | 仅设 `rot`，pos 走默认 | 显式设 pos 和 rot |
| `test_box` 坐标注释 | 描述 y∈[0.98, 3.69]、z≈0.85、x=0.62（与实际 pos 匹配） | 描述 y∈[-11.42, -8.70]、x=15.30（旧场景坐标） |
| `remote_upper_body_ik` / `object_sync` | 单次定义、用变量 | congxian 这里有重复定义 + 硬编码 endpoint |
| `ObservationsCfg` | 用 `LOCAL_ROBOT_ASSET_NAME` | congxian 末尾又重复了一遍 |
| `TerminationsCfg.success` | 用 `LOCAL_ROBOT_ASSET_NAME` 配置 robot_cfg | 仅 task_link_name |
| `EventsCfg` | 包含 conveyor 对齐 + viewer 锁定（用 NETWORK_CFG） + 改用 PhysxSurfaceVelocityAPI（去掉 drive_test_box 周期事件） | 旧版（带 drive_test_box 周期事件） |
| XR anchor 绑定 | 通过 NETWORK_CFG 决定挂哪个机器人 | 硬编码 pelvis 路径 |
| `ZeroMqGameSubDeviceCfg(...)` 调用 | 仅 `topic="state"`（其它走 NETWORK_CFG 默认值） | 全部硬编码 endpoint/player_id |

**全部取 HEAD（xinjie）**，使用 `git checkout --ours`。

> ⚠️ **额外坑**：auto-merge 在不冲突的部分把 congxian 的 `EventsCfg` 类整段保留下来了，导致最终文件出现两个同名 `EventsCfg` 类。`git checkout --ours` 直接拿 HEAD 整版文件，避开了这个问题。如果是手工编辑冲突标记，需要额外去重。

### 8. `source/isaaclab_tasks/.../pick_place/zmq_object_sync.py`

- xinjie：`endpoint = NETWORK_CFG.zmq_object_sync_endpoint`
- congxian：`endpoint = "tcp://192.168.40.30:15555"`

**取 xinjie**（配置化）。

## congxian 引入的额外文件

`congxian` 的 squash commit 里夹带了一批看起来是调试/备份产物的文件。**根据用户决定全部保留**：

### A. 根目录乱放的脚本 / 备份文件（保留）

- `_conda_python.bat`
- `actions.py`
- `g1_head_pose_locomotion.py`
- `locomanipulation_g1_env_cfg.py`（注：与 `source/isaaclab_tasks/...` 下的同名文件路径不同）
- `pink_controller_cfg.py`（同上）
- `temp_inspect_usd_scene.py`
- `isaaclab.bat.bak_pin_bootstrap`

### B. 显式命名的备份文件（保留）

- `source/isaaclab/isaaclab/controllers/pink_ik/pink_ik.py.bak_daqp_fallback`
- `source/isaaclab_tasks/.../pick_place/locomanipulation_g1_env_cfg_backup_20260518_151946.py`
- `source/isaaclab_tasks/.../pick_place/mdp/events_backup_20260518_151946.py`

### C. head_pose locomotion 新功能（保留）

- `source/isaaclab/.../retargeters/humanoid/unitree/g1_head_pose_locomotion.py`（新 retargeter）
- `source/isaaclab_tasks/.../pick_place/g1_head_pose_locomotion.py`
- `source/isaaclab_tasks/.../pick_place/mdp/g1_head_pose_locomotion.py`
- `source/isaaclab_tasks/.../pick_place/pink_controller_cfg.py`（注意：与 `configs/pink_controller_cfg.py` 共存）

### D. 二进制模型权重（保留）

- `assets/model/policy.onnx`（约 2.5 MB）
- `assets/model/policy1.onnx`（约 2.5 MB）

> 💡 **建议**：未来如果有清理意愿，根目录的 `actions.py`、`g1_head_pose_locomotion.py`、`locomanipulation_g1_env_cfg.py`、`pink_controller_cfg.py`、`temp_inspect_usd_scene.py` 几乎肯定是误提交（路径错位），可以单独起个清理 PR 删除。`*.bak` / `*_backup_*` 系列也属于本不该提交的产物。

## 遇到的坑

### 1. 本地分支与远端 diverged

切到 `xinjie` 后发现本地与 `origin/xinjie` 分别领先 3 / 落后 15 个提交。本地 3 个提交看起来与远端 15 个里的部分提交主题重叠（都是 ZeroMQ 网络配置重构）。

**处理**：备份后 reset 到 origin/xinjie，从更完整的远端版本出发。

### 2. record_demos.py 主循环逻辑双向修改

两边解决的是不同问题（多机器人路由 vs 动作维度补齐），只能合并而非二选一。已询问用户确认合并方案。

### 3. locomanipulation_g1_env_cfg.py 重复类定义

git auto-merge 把两边的 `EventsCfg` 都保留了，导致同名类重复。`git checkout --ours` 解决。

### 4. 大量"垃圾"文件

congxian 的 squash commit 引入了多个明显误提交的备份/临时文件。已询问用户后保留。

## 后续建议

- ✅ Merge 已完成在本地 `xinjie` 分支
- ⏳ **尚未 push 到远端**
- 🧪 建议先验证关键入口能跑通后再 push：
  - `record_demos.py` 启动单设备 / 多设备场景
  - locomanipulation G1 环境实例化
- 🧹 后续考虑清理误提交的备份/根目录脚本（A、B 类文件）
- 💾 备份分支 `xinjie-local-backup-20260520` 可在确认无需恢复后删除
