# 咖啡厅/吧台场景 USD 资源调研笔记

## 背景

为咖啡厅双机递杯任务（`locomanipulation_g1_cafe_handover_env_cfg.py`）寻找
合适的吧台 USD 场景资产，最终同步生成了一个简单的本地 USDA 文件。

---

## 调研结论

### 1. LightwheelAI / Lightwheel-simready-asset（推荐）

| 属性 | 内容 |
|------|------|
| GitHub | https://github.com/LightwheelAI/Lightwheel-simready-asset |
| 许可 | CC BY-NC 4.0（非商业免费） |
| 资产数 | 259 个 SimReady USD 资产（含厨房场景与物件） |
| 格式 | USD 和 MJCF 双格式，已在 IsaacLab 4.5/5.0 验证 |
| 下载 | README 内附 Google Drive 链接（非 repo 内存储） |
| 适用场景 | 最直接：已验证可用于 IsaacLab Teleoperation + RL 工作流 |

**适用性**：明确包含 Kitchen Scene，与本任务框架最匹配，但没有专门的"吧台"
分隔结构，可能需要组合多件家具。

---

### 2. SideFX Bar Scene（最接近吧台场景）

| 属性 | 内容 |
|------|------|
| 下载页 | https://www.sidefx.com/contentlibrary/bar-scene/ |
| 许可 | Standard License（免费） |
| 大小 | 1.6 GB |
| 格式 | Houdini Solaris/LOP 项目文件，输出为 USD |
| 包含 | 吧台台面、酒瓶、吧台高脚椅、灯光、摄像机 |

**适用性**：是目前找到的唯一专门建模"吧台"的 USD 场景，结构完整。
需要 Houdini 18+ 打开并导出 USD，之后需在 Isaac Sim 内补充物理属性
（碰撞体、刚体设置）。

---

### 3. NVIDIA Omniverse Restaurant Demo Assets Pack

| 属性 | 内容 |
|------|------|
| 下载页 | https://docs.omniverse.nvidia.com/usd/latest/usd_content_samples/downloadable_packs.html |
| 大小 | 3.9 GB |
| 包含 | 含地面层餐厅完整场景（ground floor restaurant） |
| 适用性 | 官方品质最高，直接兼容 Isaac Sim，但体积大、无法直接从 GitHub 获取 |

---

### 4. Sketchfab — Bar Counter（RimaAkter）

| 属性 | 内容 |
|------|------|
| 模型页 | https://sketchfab.com/3d-models/bar-counter-37397956531640a58d99f54b08883315 |
| 许可 | CC BY 4.0（标注作者即可商用） |
| 多边形 | 1.4M 三角面（需减面） |
| 格式 | 可下载 USDZ |
| 适用性 | 单件道具级吧台，下载即用，但面数高，需用 Blender/IsaacSim 减面后再导入仿真 |

---

### 5. GitHub 中未找到的资源

经过搜索，GitHub 上目前没有：
- 专门为机器人仿真设计的咖啡厅/吧台 USD 场景
- Isaac Sim / IsaacLab 官方环境资产中也不包含 cafe / restaurant 类型

Isaac Sim 官方内置环境仅有：简单房间、仓库、医院、办公室、JetRacer 赛道。

---

## 本地生成的简单吧台 USDA

**文件**：`cafe_counter.usda`（与本文档同目录）

### 设计原则

- 纯 USD 基本图元（`UsdGeom.Cube`），无外部依赖
- 物理属性：`PhysicsRigidBodyAPI` (kinematic) + `PhysicsCollisionAPI`
- 所有锚点 Xform 的世界坐标与 `locomanipulation_g1_cafe_handover_env_cfg.py`
  中的 `FALLBACK_*` 常量完全对齐

### 吧台几何（五件式结构）

```
CounterTop  (台面·浅色石材)
  X: −0.32 ~ 1.52 m（含 2 cm 悬挑）
  Y:  0.18 ~ 0.64 m（含 2 cm 前悬）
  Z:  0.84 ~ 0.89 m（5 cm 厚台板，顶面 = 0.89 m）
  颜色: 浅灰石 (0.88, 0.85, 0.80)

FrontFascia (前挡板·深色木料)
  X: −0.30 ~ 1.50 m
  Y:  0.20 ~ 0.24 m（4 cm 厚正面板）
  Z:  0.00 ~ 0.84 m（地面到台面底）
  颜色: 深咖木 (0.30, 0.20, 0.12)

BackPanel   (背板·深色木料)   同前挡板尺寸，位于 Y = 0.60 ~ 0.64 m
LeftPanel   (左封板)          X = −0.30 ~ −0.26 m，Y = 0.24 ~ 0.60 m，Z = 0.00 ~ 0.84 m
RightPanel  (右封板)          X =  1.46 ~  1.50 m，Y = 0.24 ~ 0.60 m，Z = 0.00 ~ 0.84 m
```

### 锚点 Xform（对齐 env_cfg 常量）

| Prim 名称 | 世界坐标 (x, y, z) | 旋转 | 说明 |
|-----------|--------------------|------|------|
| `RobotSpawnA` | (0.00, 0.00, 0.75) | 无 | Giver，G1 默认朝 +X |
| `RobotSpawnB` | (1.15, 0.00, 0.75) | Z 轴 180° | Receiver，朝 -X 面向 Robot A |
| `CupSpawn` | (0.20, 0.42, 0.95) | 无 | 杯子初始位置，台面上方 |
| `HandoverZone` | (0.62, 0.42, 0.98) | 无 | 交接目标区域 |
| `ServeZone` | (1.00, 0.48, 0.95) | 无 | 出餐区（Robot B 放置目标）|
| `ViewerAnchor` | (0.62, 0.42, 0.98) | 无 | 默认观察视角参考点 |

### 在 env_cfg 中的两种使用方式

**方式 A：替换 packing_table.usd（作为道具 USD）**

```python
# 在 CafeHandoverG1SceneCfg 中修改 counter 定义：
counter = AssetBaseCfg(
    prim_path="/World/envs/env_.*/Counter",
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),  # 直接放世界原点
    spawn=UsdFileCfg(
        usd_path="<path_to>/cafe_counter.usda",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
)
```

**方式 B：作为场景级 USD 背景**

在 `scene` 中新增一个 AssetBaseCfg：

```python
cafe_scene = AssetBaseCfg(
    prim_path="/World/CafeScene",
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    spawn=UsdFileCfg(usd_path="<path_to>/cafe_counter.usda"),
)
```

此时 events / observations 代码可以通过 prim 名称（如 `/World/CafeScene/Root/RobotSpawnA`）
直接查找世界坐标，无需使用 fallback 常量。

### 后续升级方向

V1 稳定后可考虑：
1. 增加前置隔板（`FrontBarrier` Cube，Z=0.89~1.19，Y≈0.22，模拟吧台边缘护板）
2. 替换为 SideFX Bar Scene 的 USD 资产（需做物理属性补充）
3. 从 LightwheelAI 下载厨房场景并合并到此 USDA
4. 用 Blender 制作更精细的低面数吧台模型并导出 USD
