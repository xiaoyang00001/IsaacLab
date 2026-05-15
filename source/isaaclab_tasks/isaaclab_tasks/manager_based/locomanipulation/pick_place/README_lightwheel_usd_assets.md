# Lightwheel.ai USD 场景资产调研

调研日期：2026-05-15  
目标：为咖啡递送/拾取放置任务寻找合适的咖啡厅或吧台 USD 场景

---

## 结论

**lightwheel.ai 目前没有专门的咖啡厅/吧台 USD 场景**，但有厨房场景可作为替代方案。

---

## Lightwheel_OpenSource.zip 实际资产清单

下载路径：`D:\Downloads\Lightwheel_OpenSource.zip`

解压目录结构：
```
Lightwheel_OpenSource/
├── Locomotion/      # 完整场景（移动导航）
└── Manipulation/    # 单体可操作物件
```

---

### 完整场景（可直接用作任务背景）

| USD 路径 | 说明 |
|----------|------|
| `Locomotion/KitchenRoom/KitchenRoom.usd` | **完整厨房场景** ★ 推荐 |
| `Locomotion/KitchenRoom/KitchenRoom_RSS.usd` | 厨房场景（RSS变体，更大） |
| `Locomotion/Apartment/scene_04.usd` | 公寓场景（含吧台道具） |
| `Locomotion/2-StoryStaircase/2-StoryStaircase.usd` | 两层楼梯场景 |
| `Locomotion/Grass/Grass.usd` | 草地场景 |
| `Locomotion/GravelGround/GravelGround.usd` | 碎石地面 |
| `Locomotion/MudGround/MudGround.usd` | 泥地场景 |
| `Locomotion/SLATEGround/SLATEGround.usd` | 石板地场景 |
| `Locomotion/SnowGround/SnowGround.usd` | 雪地场景 |

---

### 咖啡厅/吧台相关资产 ★ 重点

#### 吧台道具（`Locomotion/Apartment/Props/`）
`SM_bar_01_0.usd` / `SM_bar_03_0.usd` ~ `SM_bar_16_0.usd`（共 15 个吧台道具）

#### 咖啡机（`Manipulation/`，共 25 个）
`CoffeeMachine039` / `041` / `044` / `045` / `048` / `057` / `059` / `060` / `064` / `067` / `070` / `074` / `075` / `079` / `082` / `083` / `086` / `087` / `090` / `092` / `093` / `094` / `099` / `102` / `103`

另有 `Locomotion/KitchenRoom/CoffeeMachine006/CoffeeMachine006.usd`（已集成在厨房场景内）

#### 电热水壶（`Manipulation/`，共 15 个）
`ElectricKettle001` ~ `ElectricKettle015`

#### 搅拌机（`Manipulation/`，共 15 个）
`Blender001` ~ `Blender015`

---

### 厨房场景内置道具（`Locomotion/KitchenRoom/`）

| 资产 | 说明 |
|------|------|
| `Kitchen_Cabinet001.usd` / `Cabinet002.usd` | 厨房橱柜（可开门） |
| `Kitchen_TopCabinet.usd` | 吊柜 |
| `Kitchen_InsularShelf.usd` | 岛台式置物架 |
| `Kitchen_Other/Kitchen_Bottle*.usd` | 各种瓶子（x7） |
| `Kitchen_Other/Kitchen_Box*.usd` | 各种盒子 |
| `Refrigerator001.usd` | 冰箱 |
| `Dishwasher054.usd` | 洗碗机 |
| `Microwave017.usd` | 微波炉 |
| `Sink054.usd` | 水槽 |
| `Table049.usd` | 厨房桌子 |
| `Toaster003.usd` | 烤面包机 |
| `Stovetop012.usd` | 灶台 |
| `RangeHood015.usd` | 抽油烟机 |
| `Pot057.usd` | 锅 |
| `WallStackOven004.usd` | 壁挂式烤箱 |
| `InteractiveAsset/SM_P_Choppingboard_01.usd` | 切菜板（可交互） |
| `InteractiveAsset/SM_P_Flavour_02.usd` 等 | 调味瓶罐（含托盘、瓶盖） |

---

### 其他单体操作物（`Manipulation/`）

| 类别 | 数量 |
|------|------|
| Microwave（微波炉） | 24 个 |
| Refrigerator（冰箱） | 27 个 |
| Sink（水槽） | 24 个 |
| Oven（烤箱） | 11 个 |
| Stove/Stovetop（灶台） | 14 个 |
| RangeHood（抽油烟机） | 12 个 |
| Toaster（烤面包机） | 22 个 |
| ToasterOven（小烤箱） | 24 个 |
| StandMixer（立式搅拌机） | 11 个 |
| Dishwasher（洗碗机） | 13 个 |

---

## 资源链接

- SimReady Library 资产库：https://www.lightwheel.ai/asset-library
- GitHub 开源资产（259个 USD，非商业免费）：https://github.com/LightwheelAI/Lightwheel-simready-asset
- NVIDIA USD Search 博客（2000+ 资产）：https://lightwheel.ai/media/lightwheel-usd-blog-CoRL
- LW-BenchHub（含 RoboCasa 268 个任务）：https://github.com/LightwheelAI/LW-BenchHub

---

---

## 如何获取资产

### 方案 A：直接下载 259 个单体 USD 资产（无需注册）【推荐先试】

**Google Drive 直链（免费，非商业使用）：**
https://drive.google.com/file/d/1S1W-vDNvsOOQU0qViEax_QS0zr0zGLfX/view?usp=sharing

- 登录 Google 账号即可下载，无需注册 lightwheel 账号
- 包含 251 个操纵资产（家居物品、厨房用品、工业组件等）+ 8 个移动导航资产
- USD 格式，兼容 Isaac Sim 4.5 / 5，即插即用
- 使用时需署名：`lightwheel_{asset_name}`

> **注意**：这 259 个是**单体物件资产**，不是完整的厨房/咖啡厅场景。

---

### 方案 B：获取 RoboCasa 100 个完整厨房场景（需注册 + SDK）

完整的厨房场景（10种布局 × 10种风格）需要通过 Lightwheel SDK 下载。

#### 步骤 1：注册账号
在 https://www.lightwheel.ai 注册账户。

#### 步骤 2：安装 SDK
```bash
pip install lightwheel-sdk   # 需要 Python 3.10+
```

#### 步骤 3：登录
```bash
lwsdk login
```

#### 步骤 4：下载厨房 USD 场景
```python
from lightwheel_sdk.loader import floorplan_loader

usd_path_future = floorplan_loader.acquire_usd(
    scene="robocasakitchen",
    layout_id=1,   # 布局编号 (1-10)
    style_id=1     # 风格编号 (1-10)
)
usd_path, metadata = usd_path_future.result()
print(f"USD 文件路径: {usd_path}")
```

---

### 方案 C：联系 Lightwheel 定制咖啡厅场景
- 邮件：haibo.yang@lightwheel.ai
- Discord：https://discord.com/invite/FrsNM5v9

### 方案 D：第三方资产平台
- **NVIDIA Omniverse Nucleus**：Isaac Sim 官方资产库，搜索 restaurant/kitchen
- **Sketchfab**：搜索 "cafe counter USD" 或 "bar counter USD"
- **Turbosquid / CGTrader**：商业授权咖啡厅场景

---

## 当前任务背景

本项目已有咖啡递送任务（cafe coffee handover task），参见：
- `README_cafe_coffee_handover_task_zh.md`
- 当前使用 warehouse_simple7 场景作为占位场景
- 目标替换为更真实的咖啡厅/厨房环境
