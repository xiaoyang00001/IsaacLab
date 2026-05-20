# 网络配置重构记录

## 概述

将最近提交 (`d7788fc`) 中分散在各模块的硬编码 IP 地址和端口，统一抽取到 `configs/network_cfg.py` 中集中管理。

## 变更动机

原始代码中，以下位置存在硬编码的 IP 和端口：

| 位置 | 硬编码值 |
|------|---------|
| `openxr_device.py:155` | `tcp://192.168.1.149:14026` |
| `locomanipulation_g1_env_cfg.py:160-161` | `tcp://192.168.1.149:15555` |
| `locomanipulation_g1_env_cfg.py:386` | `tcp://192.168.1.149:14025` |
| `locomanipulation_g1_env_cfg.py:388` | `zmq_player_id=1` |
| `locomanipulation_g1_env_cfg.py:392-393` | `local_player_id=1`, `target_remote_player_id=2` |
| `zmq_object_sync.py:47` | `tcp://192.168.10.46:15555` |

每次切换网络环境（如更换机器、切换局域网）都需要在多个文件中查找替换，容易遗漏和出错。

## 解决方案

### 新增文件

- **`configs/network_cfg.py`** — 网络配置中心，所有 IP 和端口集中定义

```python
from isaaclab_tasks.manager_based.locomanipulation.pick_place.configs.network_cfg import NETWORK_CFG

# 使用方式
NETWORK_CFG.zmq_game_server_endpoint   # tcp://192.168.1.149:14026
NETWORK_CFG.zmq_game_sub_endpoint      # tcp://192.168.1.149:14025
NETWORK_CFG.zmq_object_sync_endpoint   # tcp://192.168.1.149:15555

NETWORK_CFG.zmq_player_id              # 1
NETWORK_CFG.local_player_id            # 1
NETWORK_CFG.target_remote_player_id    # 2
```

### 修改文件

| 文件 | 改动说明 |
|------|---------|
| `openxr_device.py` | `OpenXRDeviceCfg` 新增 `zmq_game_server_endpoint`、`zmq_player_id` 字段，默认值从 `NETWORK_CFG` 读取 |
| `zeromq_game_sub_device.py` | `ZeroMqGameSubDeviceCfg` 的 `endpoint`、`local_player_id`、`target_remote_player_id` 默认值从 `NETWORK_CFG` 读取 |
| `zmq_object_sync.py` | `ZmqObjectSyncActionCfg` 的 `endpoint` 默认值从 `NETWORK_CFG` 读取 |
| `locomanipulation_g1_env_cfg.py` | 移除所有冗余的构造函数参数，不再需要直接导入 `NETWORK_CFG` |

### 设计原则

所有 Cfg 类的默认值直接从 `network_cfg.py` 读取，业务代码（如 `locomanipulation_g1_env_cfg.py`）**无需**再通过构造函数重复传入。改一处配置，全局生效。

## 配置参数说明

| 参数 | 默认值 | 用途 |
|------|--------|------|
| `zmq_game_server_ip` | `192.168.1.149` | ZeroMQ 游戏服务器 IP |
| `zmq_game_server_port` | `14026` | ZeroMqGameClient 通信端口 |
| `zmq_game_sub_port` | `14025` | ZeroMqGameSubDevice 订阅端口 |
| `zmq_object_sync_ip` | `192.168.1.149` | 物体同步服务器 IP |
| `zmq_object_sync_port` | `15555` | 物体同步 (ZmqObjectSync) 端口 |
| `zmq_player_id` | `1` | ZeroMQ 游戏客户端 Player ID |
| `local_player_id` | `1` | ZeroMQ 游戏订阅端本地 Player ID |
| `target_remote_player_id` | `2` | ZeroMQ 游戏订阅端远程 Player ID |

## 如何修改网络配置

只需编辑 `configs/network_cfg.py` 中的对应字段：

```python
@configclass
class NetworkCfg:
    zmq_game_server_ip: str = "192.168.1.149"   # 改为你的 IP
    zmq_game_server_port: int = 14026            # 改为你的端口
    # ...
```

## 架构图

```
configs/network_cfg.py  (唯一定义源)
       │
       ├──► OpenXRDeviceCfg 默认值
       │       └── zmq_game_server_endpoint, zmq_player_id
       │
       ├──► ZeroMqGameSubDeviceCfg 默认值
       │       └── endpoint, local_player_id, target_remote_player_id
       │
       └──► ZmqObjectSyncActionCfg 默认值
               └── endpoint

locomanipulation_g1_env_cfg.py  (无需传参，自动继承默认值)
       │
       ├──► OpenXRDeviceCfg(...)              ← 无需 endpoint/player_id 参数
       ├──► ZeroMqGameSubDeviceCfg(...)       ← 无需 endpoint/player_id 参数
       └──► ZmqObjectSyncActionCfg(...)       ← 无需 endpoint 参数
```
