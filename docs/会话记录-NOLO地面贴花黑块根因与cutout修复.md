# 会话记录：NOLO 地面贴花黑块——根因与 cutout 修复

- 日期：2026-07-18
- 分支：修复实施于工作区 `warehouse-simple6_v48.usd`，提交至 `pickplace-g1-collision-pd-merged-005`
- 涉及文件：
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/warehouse-simple6_v48.usd`（修改 + 首次纳入 git 管理）
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomanipulation/pick_place/nolo_label.png`（贴图依赖，目标分支已跟踪、内容一致，未改动）

## 现象

场景（跑 pick_place 任务时加载的 `warehouse-simple6_v48.usd`）里的 NOLO logo 贴图显示为一块**纯黑矩形**；但同一个 USD 单独打开时 logo 显示正常。

## 排查路径

1. 全仓库按文件名找 `*nolo*`：只有 `pick_place/nolo_label.png`（1024×256 RGBA PNG）。
2. `grep` 全仓库找不到任何文件引用 `nolo_label` —— **这是个坑**：二进制 `.usd`（crate 格式）的 token 字符串表是 LZ4 压缩的，`grep` 搜不到里面的资产路径。**必须用 `pxr.Sdf` 遍历 AttributeSpec 扫 `Sdf.AssetPath`**。
3. 用 Sdf 扫描后定位：`warehouse-simple6_v48.usd` 内 `/Root/NOLO_Mat/Tex.inputs:file = ./nolo_label.png`，绑定在 `/Root/NOLO_FloorDecal`（1.5m × 0.5m 地面贴花 Mesh，z=0.01，UV 正常，法线朝上）。
4. 材质网络（UsdPreviewSurface）：
   - `diffuseColor ← Tex.outputs:rgb`
   - `opacity ← Tex.outputs:a`
   - **未设置 `inputs:opacityThreshold`**（默认 0 = 半透明混合模式）
   - 未设置 `sourceColorSpace`
5. PIL 分析 PNG 通道：**79.8% 像素完全透明，且透明区 RGB = 纯黑 (0,0,0)**；logo 本体为黄色 (255,220,0)；最大 alpha 仅 235（没有完全不透明的像素）。

## 根因链（三件套）

1. PNG 大面积透明区的 RGB 是纯黑（logo 图常见做法，正常情况下被 alpha 挡住看不见）；
2. 材质把 alpha 连到 `opacity` 但 `opacityThreshold=0`，即依赖**半透明混合（blend）**渲染路径；
3. IsaacLab 跑任务用性能优先的 RTX 实时渲染预设，**不处理 UsdPreviewSurface 的分数透明度**（Fractional Cutout Opacity 关闭），alpha 被整体忽略 → 直接显示 RGB 通道 → 80% 纯黑的矩形。

单独打开 USD 时的渲染设置支持半透明混合，所以显示正常——症状"单独打开好、进场景黑"正是渲染预设差异所致。

## 修复（渲染器无关）

用 `pxr.Sdf` 直接写 layer（不经 Stage 合成，避免拉起全部网络引用）：

```python
from pxr import Sdf
layer = Sdf.Layer.FindOrOpen('warehouse-simple6_v48.usd')
shader = layer.GetPrimAtPath('/Root/NOLO_Mat/Shader')
tex = layer.GetPrimAtPath('/Root/NOLO_Mat/Tex')
a1 = Sdf.AttributeSpec(shader, 'inputs:opacityThreshold', Sdf.ValueTypeNames.Float)
a1.default = 0.4          # 切 cutout 镂空模式：alpha>=0.4 显示，否则镂空
a2 = Sdf.AttributeSpec(tex, 'inputs:sourceColorSpace', Sdf.ValueTypeNames.Token)
a2.default = 'sRGB'       # 显式声明色彩空间，排除自动探测差异
layer.Save()
```

- `opacityThreshold=0.4`：切换成 **cutout（镂空）模式**。该路径在所有 RTX 模式（含性能预设）下都不依赖半透明设置，单独打开与跑任务表现一致。阈值 0.4 对本图安全（logo 像素 alpha≈200~235，远高于 0.4×255=102）。
- `sourceColorSpace="sRGB"`：显式声明，防止不同加载路径下自动探测不一致。

修复后重开 layer 验证属性均已写入。

## 注意事项

- **若 Isaac Sim 正开着该场景，不要直接 Ctrl+S**（会用内存旧状态覆盖磁盘上的修复），重新打开文件即可。
- cutout 模式下 logo 边缘为硬切边（无半透明过渡）。地面贴花基本看不出差别；若要软边缘需开渲染设置 Fractional Cutout Opacity 或 quality 渲染模式，但那会重新依赖运行环境设置，不如 cutout 稳。
- 同类问题速查：透明 PNG 贴图在任务里发黑 → 先查 Shader 是否 authored `opacityThreshold`，再用 PIL 查 PNG 透明区 RGB 是不是黑。

## git 纳管说明

`warehouse-simple6_v48.usd` 此前命中 `.gitignore` 的 `**/*.usd` 规则一直未被跟踪，跨机器同步只能手动 copy。本次在 `pickplace-g1-collision-pd-merged-005` 分支 `git add -f` 强制纳管；**文件一旦被跟踪，后续修改会正常出现在 `git status`**，ignore 规则不再影响它。
