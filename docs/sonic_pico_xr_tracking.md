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
- [x] 完整启动验证（SonicSolo 场景加载 + teleop 主循环）：2026-07-02 二次实测在 commit
  余量仅 6GB（比三连崩时更紧）下顺利通过，`app ready` 41s，`Teleoperation started`，
  `SonicRobotStatePublisher` 稳定跑过 6000+ 步无崩溃——说明问题 3 不是必现，视当次系统负载而定
- [x] XR 锚点绑定 `sonic_robot/pelvis`：日志确认
  `Anchor Prim Path: /World/envs/env_0/SONICRobot/pelvis (Dynamic Anchoring)`，
  详见下方"XR 锚点"一节

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

### 问题 4：Windows 端 `xrCreateInstance failed`——SteamVR 在跑，但没有 PICO 客户端注册头显

2026-07-02 手动实测（锚点配置已生效，`Teleoperation started` 正常）时，仿真启动阶段打出：

```text
Error [GENERAL | xrCreateInstance | OpenXR-Loader] : LoaderInstance::CreateInstance chained CreateInstance call failed
Error [GENERAL | xrCreateInstance | OpenXR-Loader] : xrCreateInstance failed
```

随后日志里还出现一段 `XR session start`（54.4s）到 `XR session end`（72.9s，约 18.6 秒）——
Kit 的 hydra 渲染设置为 XR 会话切换过一次又切回来，但没有产出可用画面，83.6s 整个 App 关闭。

现场排查（PowerShell 注册表 + 进程检查）：

| 检查项 | 结果 |
|------|------|
| `HKLM:\SOFTWARE\Khronos\OpenXR\1\ActiveRuntime` | 已设置，指向 `...\SteamVR\steamxr_win64.json` |
| SteamVR 安装 | 存在（`Program Files (x86)\Steam\steamapps\common\SteamVR`） |
| `vrserver`/`vrmonitor` 进程 | **在跑**（当天 14:40 就启动，非本次新起） |
| PICO 客户端（PICO Connect / Streaming Assistant 等） | **未安装**——文件夹扫描 + 注册表已安装程序列表均为空 |

结论：OpenXR runtime 默认值和 SteamVR 服务本身都没问题，**缺的是 PICO 官方 Windows
客户端**——没有它，PICO 头显无法在 SteamVR 里注册成一个可用的 HMD 设备，
`xrCreateInstance`/session 建立自然失败，与 `-Xr`/锚点代码无关。

下一步需要用户决定：安装 PICO 对应的 Windows streaming 客户端（PICO 4 系列是
"Streaming Assistant"，PICO 企业/教育版可能是 "PICO Connect" 或 "PICO Business
Streaming"，需按头显具体型号确认），装好后头显应能出现在 SteamVR 设备列表里，
再重新测 `-Xr` 启动。

## 本机环境（2026-07-02 实测）

- GPU：NVIDIA RTX 3060 Laptop 6 GB（Kit 枚举 GPU 0 Active，5996 MB）+ Intel UHD；
  Oray/Todesk/MuMu 等虚拟显卡未进入 Kit 枚举，无影响
- RAM 64 GB；pagefile 29 GB → commit limit ≈ 92 GB
- 驱动 576.83，Graphics API D3D12
- conda env：`env_isaaclab`（miniconda3）

## XR 锚点：视角绑定 sonic_robot（2026-07-02）

目标确认为 OpenXR 路径，且视角要以 sonic_robot 的位置为参照（仿照
`locomanipulation_g1_env_cfg.py` 里 `Robot`/`RemoteRobot` 双机的 pelvis 锚定方式）。

**关键机制**：`env_cfg.xr = XrCfg(...)` 单独赋值不会生效——`XrCfg` 只在
`OpenXRDevice.__init__` 里被真正消费（创建锚点 prim、订阅 Kit `pre_sync_update`
持续同步）。全仓库唯一构造 `OpenXRDevice` 的地方是
`create_teleop_device()`，而它只有在 `env_cfg.teleop_devices.devices` 里存在
一个 key 等于 `args_cli.teleop_device` 的 `OpenXRDeviceCfg` 条目时才会触发
（`teleop_se3_agent.py` deploy_target_mode 分支的设备选择逻辑）。因此完整链路是：

```text
-Xr 开关
  -> --xr                        (XR kit 切换，见上方)
  -> --teleop_device handtracking (脚本联动追加)
  -> env_cfg.teleop_devices.devices["handtracking"] 命中
  -> create_teleop_device("handtracking", ...) -> OpenXRDevice(xr_cfg=self.xr)
  -> XRAnchor prim 挂在 anchor_prim_path 下，Dynamic Anchoring 跟随该 prim 移动
```

`SonicSoloLocomanipulationEnvCfg`/`SonicFullsceneLocomanipulationEnvCfg`
`__post_init__` 新增：

```python
self.xr = XrCfg(
    anchor_pos=(0.0, 0.0, -0.82),
    anchor_rot=(1.0, 0.0, 0.0, 0.0),
    anchor_prim_path="/World/envs/env_0/SONICRobot/pelvis",
    fixed_anchor_height=True,
)
self.teleop_devices = DevicesCfg(devices={"handtracking": OpenXRDeviceCfg(xr_cfg=self.xr)})
```

`sonic_robot` 在两个场景里都是同一个 `SONIC_G1_29DOF_CFG`，prim path 固定为
`{ENV_REGEX_NS}/SONICRobot`（源码定位：`locomanipulation_g1_env_cfg.py:491`），
带 `pelvis` link。`anchor_pos` Z 偏移 -0.82 与参考配置一致：把锚点从 pelvis
高度下沉到落地点附近，用户真实站立时的头部高度会自然落在机器人大致的视线
高度，而不是锁死一个刚性头部摄像机（旋转模式沿用参考的默认 FIXED，不随
机器人转身而转动房间朝向，避免眩晕）。`-0.82` 这个数值来自参考配置的经验值，
真机验证后可能需要微调。

`teleop_interface.add_callback("U"/"R"/...)` 传给 `OpenXRDevice` 时会被无校验地
存进 `self._additional_callbacks` 字典（`add_callback` 实现见
`openxr_device.py:241-249`），不会抛异常——这些键盘式回调名在 OpenXR 手势
消息总线上永远不会被触发，只是静默挂在那里，不影响锚点构造成功。

**代码改动**：

- `sonic_solo_locomanipulation_env_cfg.py`、`sonic_fullscene_locomanipulation_env_cfg.py`
  新增 import（`DevicesCfg`、`OpenXRDeviceCfg`、`XrCfg`）+ `__post_init__` 锚点配置
- `start_windows_isaaclab_sonic.ps1`：`-Xr` 分支追加 `--teleop_device handtracking`

## 待办

- [x] ~~清理提交内存后完整启动验证~~ 2026-07-02 已通过（日志证据见上方"当前状态"）
- [ ] **P0** Windows OpenXR runtime 配置与实测——2026-07-02 实测已定位缺口，见下方"问题 4"：PICO Connect 串流 → SteamVR，系统默认 OpenXR runtime
  设为 SteamVR；点击 Start AR 实测进会话（Ubuntu 侧已验证的 CloudXR pip runtime 是 Linux-only 路径体系，
  Windows 需要独立验证 SteamVR 路线或其他本地 runtime）——这是路径 A 能否真正跑通 PICO 会话的关键路径
- [ ] **P1** XR 会话激活状态下复测 `env_hz` 是否仍钉 50 Hz 实时（闭环七条件之一；XR 渲染开销更高，
  掉速会导致步态相位畸变）
- [ ] **P1** 真机验证后微调 `anchor_pos` Z 偏移（-0.82 是参考值，未在真实头显上标定过）
- [ ] **P2** PICO 手柄/手追踪作为遥操输入：当前只是挂了空 retargeter 列表的观察锚点，
  真正遥操需要给 `OpenXRDeviceCfg` 配 retargeters（deploy_target_mode 目前从不调用
  `teleop_interface.advance()`，接输入需要额外改造主循环）
- [ ] **P3** 收敛双份脚本拷贝（GR00T 为唯一源或反之），消除手动双写成本

知识库跟踪页（更详细，含踩坑记录）：机器人知识库
`NVIDIA/IsaacLab/SONIC-Windows-IsaacLab-XR模式与AR按钮集成跟踪.md`
