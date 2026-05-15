# Lightwheel.ai USD 场景资产调研

调研日期：2026-05-15  
目标：为咖啡递送/拾取放置任务寻找合适的咖啡厅或吧台 USD 场景

---

## 结论

**lightwheel.ai 目前没有专门的咖啡厅/吧台 USD 场景**，但有厨房场景可作为替代方案。

---

## Lightwheel SimReady Library 资产概览

- **总资产数**：259 个 USD 资产
- **资产分类**：
  - 操纵资产（251 个）：家居物品、厨房用品工具、工业组件、交互元素（门、抽屉、橱柜）
  - 移动导航资产（8 个）：地形变化和导航环境

### 已确认的具体场景类型

| 场景 | 说明 |
|------|------|
| Interactive Home Environment | 智能家居交互训练环境 |
| Burger Assembly Simulation | 汉堡装配（食品工业） |
| Strawberry Harvest Simulation | 草莓收获（农业） |
| Deformable Liver Cutting Simulation | 肝脏切割（生物医学） |
| Power/Cable Plug Insertion | 精密插入系列任务 |

### RoboCasa 厨房场景（最接近需求）

- **100 个厨房 USD 场景** = 10 种布局 × 10 种风格
- 物理精确仿真，完全交互式固件（橱柜、抽屉、家电）
- **可用于替代咖啡厅场景**

---

## 资源链接

- SimReady Library 资产库：https://www.lightwheel.ai/asset-library
- GitHub 开源资产（259个 USD，非商业免费）：https://github.com/LightwheelAI/Lightwheel-simready-asset
- NVIDIA USD Search 博客（2000+ 资产）：https://lightwheel.ai/media/lightwheel-usd-blog-CoRL
- LW-BenchHub（含 RoboCasa 268 个任务）：https://github.com/LightwheelAI/LW-BenchHub

---

## 如何获取 RoboCasa USD 厨房场景

### 步骤 1：安装 Lightwheel SDK

```bash
pip install lightwheel-sdk
```

系统要求：Python 3.10+

### 步骤 2：登录认证

**CLI 方式：**
```bash
lwsdk login --username your_username --password your_password
# 或交互式登录
lwsdk login
```

**代码方式：**
```python
from lightwheel_sdk.loader import login_manager
login_manager.login(username="your_username", password="your_password")
```

> 需要在 lightwheel.ai 注册账户获取用户名和密码。

### 步骤 3：下载厨房 USD 场景

```python
from lightwheel_sdk.loader import floorplan_loader

# 获取 RoboCasa 厨房 USD（layout_id: 1-10, style_id: 1-10，共100个场景）
usd_path_future = floorplan_loader.acquire_usd(
    scene="robocasakitchen",
    layout_id=1,   # 布局编号 (1-10)
    style_id=1     # 风格编号 (1-10)
)
usd_path, metadata = usd_path_future.result()
print(f"USD 文件路径: {usd_path}")
```

### 步骤 4：加载单个物体（可选）

```python
from lightwheel_sdk.loader import object_loader

file_path, object_name, metadata = object_loader.acquire_by_registry(
    registry_type="objects",
    registry_name=["chair"],   # 替换为所需物体名称
    file_type="USD"
)
```

### 可用场景参数

| 参数 | 说明 | 已知值 |
|------|------|--------|
| `scene` | 场景类型 | `"robocasakitchen"`（目前文档中唯一确认的场景） |
| `layout_id` | 厨房布局编号 | 1~10（10种布局） |
| `style_id` | 装修风格编号 | 1~10（10种风格） |

---

## 后续行动建议

### 方案 A：使用 Lightwheel RoboCasa 厨房场景（推荐先试）
1. 注册 lightwheel.ai 账号
2. `pip install lightwheel-sdk`
3. 用上述代码下载 `robocasakitchen` USD 场景
4. 在 Isaac Lab 中替换当前 warehouse_simple7 场景

### 方案 B：联系 Lightwheel 定制
- 邮件：haibo.yang@lightwheel.ai
- Discord：https://discord.com/invite/FrsNM5v9
- 询问是否有 cafe/bar 场景或定制方案

### 方案 C：第三方资产平台
- **NVIDIA Omniverse Nucleus**：Isaac Sim 官方资产库，搜索 restaurant/kitchen
- **Sketchfab**：搜索 "cafe counter USD" 或 "bar counter USD"
- **Turbosquid / CGTrader**：商业授权咖啡厅场景

---

## 当前任务背景

本项目已有咖啡递送任务（cafe coffee handover task），参见：
- `README_cafe_coffee_handover_task_zh.md`
- 当前使用 warehouse_simple7 场景作为占位场景
- 目标替换为更真实的咖啡厅/厨房环境
