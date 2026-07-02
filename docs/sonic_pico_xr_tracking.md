# SONIC-PICO 闭环 XR/AR 集成跟踪

本文档跟踪 IsaacLab Windows 侧 XR/AR 能力的集成进展：让 Isaac Sim viewport 出现 AR/VR 按钮，
供 PICO 头显通过 OpenXR 进入仿真场景（先观察，后续接手柄/手追踪遥操输入）。

文档主体使用中文说明；项目名、API 名、命令、环境变量、文件路径保持原样。

关联文档：

- `docs/gr00t_sonic_pico_isaaclab_framework.md` — 整体闭环框架
- `docs/sonic_deploy_target_minimal.md` — deploy target 最小场景
- GR00T 仓库 `docs/source/tutorials/windows_isaaclab_deploy_bridge.md` — Windows 启动命令与参数

## XR 启动机制（链路）

AR/VR 按钮不是默认 UI，它来自 XR experience kit 加载的扩展。完整链路：

```text
start_windows_isaaclab_sonic.ps1 -Xr
  -> teleop_se3_agent.py --xr          (AppLauncher argparse 标志)
  -> AppLauncher._xr = True            (app_launcher.py:628-641)
  -> experience = apps/isaaclab.python.xr.openxr.kit   (app_launcher.py:735-741, 非 headless)
  -> 加载 omni.kit.xr.system.openxr + omni.kit.xr.profile.ar, app.xr.enabled=true
  -> viewport 出现 AR/VR 按钮
```

对照：默认 `apps/isaaclab.python.kit` 内 **0 个 xr 扩展**，不可能有 AR 按钮。

`teleop_se3_agent.py` 有三条 XR 触发路径，本项目选 1：

| 路径 | 效果 | 备注 |
|------|------|------|
| 1. `--xr` CLI 标志（脚本 `-Xr`） | kit + DLSS + remove_camera_configs 全生效 | **当前采用** |
| 2. `--teleop_device` 含 `handtracking` | 同上（teleop_se3_agent.py:75-77 自动置 xr=True） | 语义是手追踪遥操，SonicSolo cfg 无 teleop_devices，不用 |
| 3. 环境变量 `XR=1` | 仅换 kit；`args_cli.xr` 仍 False，DLSS 等优化不生效 | 可用于隔离 DLSS 变量排障 |

## 启动命令

```powershell
powershell -ExecutionPolicy Bypass -File "<GR00T_ROOT>\scripts\start_windows_isaaclab_sonic.ps1" `
  -UbuntuIp "<ubuntu_ip>" `
  -WindowsIp "<windows_ip>" `
  -IsaacLabRoot "D:\path\to\IsaacLab" `
  -Xr
```

仅验证 AR 按钮时 `-UbuntuIp` 可用 `127.0.0.1` 占位（ZMQ SUB 静默重连，不影响启动）。

## 当前状态（2026-07-02）

已完成：

- [x] 根因分析：AR 按钮缺失 = 启动未走 XR kit（见上方链路）
- [x] `-Xr` 开关落地，两仓库同步提交：
  - IsaacLab `1f6e7c322`（分支 `sonic-pico-closed-loop-congxian`，rebase 前旧 hash 602e8ee1b）
  - GR00T `4b396da`（分支 `realtime-bvh-g1-retarget-0628`，含教程文档更新）
- [x] 启动验证（部分）：XR experience 加载成功（`[ext: isaaclab.python.xr.openxr-2.3.2] startup`），
  44s `app ready`，**AR 按钮已在 viewport 出现**
- [ ] 完整启动验证（SonicSolo 场景加载 + teleop 主循环）——被本机内存问题阻塞，见问题 3

## 已解决问题记录

### 问题 1：AR 按钮缺失

- 现象：按原命令启动，Isaac Sim 界面无 AR 按钮。
- 根因：脚本未传 `--xr`，`--teleop_device` 默认 `keyboard` 不含 `handtracking`，
  AppLauncher 走默认 `isaaclab.python.kit`（无任何 xr 扩展）。
- 修复：脚本新增 `-Xr` 开关 → 追加 `--xr`。注意 `--xr` 是 Python argparse 参数，
  塞进 `--kit_args` 不会被解析。

### 问题 2：PowerShell 5.1 编码坑（`-Xr` 参数消失）

- 现象：加参数后启动报 `NamedParameterNotFound: Xr`，但文件里参数明明存在。
- 根因：Windows PowerShell 5.1 按 ANSI(GBK) 读取无 BOM 的 `.ps1`；param 块内的
  UTF-8 中文注释被解码为乱码，**吞掉了 `[switch]$Xr` 参数**。
  PowerShell 7 (pwsh) 默认 UTF-8，语法检查通过，掩盖了问题。
- 修复：脚本注释全部改回纯 ASCII 英文。验证方法：
  `powershell -NoProfile -Command "(Get-Command '<脚本>').Parameters.ContainsKey('Xr')"`
- 约束：**该脚本永远保持 ASCII-only**（脚本内已留注释说明）。

### 问题 3：三连崩 = Windows 提交内存（commit limit）耗尽

三次启动失败死法不同但根因相同——commit charge 撞顶（当时 77.5 / 92.2 GB，
Isaac Sim 启动高峰需提交约 20 GB）：

| 次序 | 死亡点 | 表象 |
|------|--------|------|
| 1 | physics warm start（`initialize_physics`） | `Windows fatal exception: access violation` |
| 2 | `import torch` | `MemoryError` |
| 3 | 扩展加载 ~17s | 静默退出，日志截断，exit 0 |

- 判据：物理内存充足（free 28 GB）但 commit 余量 < 启动需求；`Memory Compression` 高
  （10.7 GB）说明内存压力大。查看命令：
  `Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory | % { "{0:N1}/{1:N1} GB" -f ($_.CommittedBytes/1GB), ($_.CommitLimit/1GB) }`
- 缓解（三选一）：重启机器；关闭 commit 大户（WindowsTerminal 旧窗口、多余 IDE/模拟器、
  `wsl --shutdown`）；扩大 pagefile（64 GB RAM 配 29 GB pagefile 偏小，建议系统管理或 ≥48 GB）。
- 澄清：第 1 次日志末尾的 `Out of memory.` 是 PowerShell 宿主被崩溃线程转储撑爆的次生错误，
  不是 Isaac Sim 死因；`-Xr` 功能本身与三次失败无关。

## 本机环境（2026-07-02 实测）

- GPU：NVIDIA RTX 3060 Laptop 6 GB（Kit 枚举 GPU 0 Active，5996 MB）+ Intel UHD；
  Oray/Todesk/MuMu 等虚拟显卡未进入 Kit 枚举，无影响
- RAM 64 GB；pagefile 29 GB → commit limit ≈ 92 GB
- 驱动 576.83，Graphics API D3D12
- conda env：`env_isaaclab`（miniconda3）

## 待办

- [ ] **P0** 清理提交内存后完整启动验证：场景加载、`Teleoperation started`、AR 按钮截图留档
- [ ] **P1** Windows OpenXR runtime 配置：PICO Connect 串流 → SteamVR，系统默认 OpenXR runtime
  设为 SteamVR；点击 Start AR 实测进会话（CloudXR runtime 容器为 Linux-only，Windows 只能走本地 runtime）
- [ ] **P1** XR 会话激活状态下复测 `env_hz` 是否仍钉 50 Hz 实时（闭环七条件之一；XR 渲染开销更高，
  掉速会导致步态相位畸变）
- [ ] **P2** `SonicSoloLocomanipulationEnvCfg` 无 `xr: XrCfg` 锚点配置，进 AR 后 anchor 在世界原点；
  视角不合适时补 XrCfg（anchor_pos/anchor_rot 对准 sonic_robot）
- [ ] **P2** PICO 手柄/手追踪作为遥操输入：需给环境配置 `teleop_devices`（现仅观察，不接输入）
- [ ] **P3** 收敛双份脚本拷贝（GR00T 为唯一源或反之），消除手动双写成本
