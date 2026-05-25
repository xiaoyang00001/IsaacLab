# ALVR + SteamVR 使用手册

## 端口冲突问题：27062 端口被占用

### 问题描述

启动 SteamVR 时报错提示 **27062 端口被占用**。

### 根本原因

ALVR（无线 VR 流媒体工具）默认使用 **27062** 端口。当 ALVR 服务进程异常退出后，TCP 连接可能卡在 `CLOSE_WAIT` 状态，导致端口未被及时释放，SteamVR 启动时无法绑定该端口。

### 诊断步骤

1. 查找占用端口的进程：
   ```bash
   netstat -ano | grep ":27062"
   ```
2. 根据输出的 PID 查看进程名：
   ```powershell
   Get-NetTCPConnection -LocalPort 27062 | ForEach-Object {
       $proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
       Write-Host "PID=$($_.OwningProcess) State=$($_.State) Process=$(if($proc){$proc.ProcessName}else{'zombie'})"
   }
   ```

### 解决方案

**方案 1：关闭 ALVR 并等待端口释放（推荐）**

1. 完全关闭 ALVR Dashboard
2. 等待约 1-2 分钟，让操作系统自动清理僵尸 TCP 连接
3. 重新启动 SteamVR

**方案 2：重启电脑**

最彻底的方法，100% 清理所有僵尸 TCP 连接。

**方案 3：修改 ALVR 使用的端口（长期方案）**

如果需要长期避免冲突，可在 ALVR 设置中将流媒体端口从默认的 `27062` 改为其他未占用端口（如 `27063`）。

### 注意事项

- ALVR 和 SteamVR 需要配合使用，不要在 ALVR 完全启动前启动 SteamVR
- 若频繁出现此问题，建议设置固定的启动顺序：先启动 ALVR，确认服务正常后再启动 SteamVR
