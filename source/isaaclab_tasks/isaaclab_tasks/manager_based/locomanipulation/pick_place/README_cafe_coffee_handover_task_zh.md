# 咖啡厅双机递咖啡任务设计文档

## 目标

设计一个新的双机协作任务：

- 两台机器人位于咖啡厅或吧台场景中
- 一台机器人负责取杯并递出
- 另一台机器人负责接杯并放到出餐区

当前场景还没有最终确定，因此这份文档不绑定具体 USD，
而是先定义任务结构、场景接口、命名锚点和代码改造方向。


## 第一版任务建议

第一版不要做液体仿真，只做“咖啡杯刚体交接”。

推荐的 V1 目标：

- 使用一个带盖咖啡杯或简化杯体刚体
- Robot A 从取杯点抓起杯子
- Robot A 将杯子移动到交接区
- Robot B 从交接区接住杯子
- Robot B 将杯子放到出餐区或顾客取餐区

这样做的原因：

- 双机协作关系清晰
- 不依赖复杂流体
- 容易定义阶段性成功条件
- 适合沿用当前 `robot / remote_robot` 双机框架


## 推荐场景拓扑

最推荐的空间关系是“隔着吧台交接”：

- `Robot A` 位于吧台内侧，扮演店员
- `Robot B` 位于吧台外侧，扮演接杯方
- 中间存在一个固定 `handover zone`
- 最终目标区域是 `serve zone`

这个拓扑的优点：

- 更像真实咖啡厅任务
- 两台机器人职责自然分开
- 交接区位置稳定，容易调
- 适合通过锚点固定场景接口

如果第一版调试困难，可以退一步做“同侧接力”版本：

- 两台机器人位于同侧
- 先验证抓杯与交接
- 之后再切回隔柜台版本


## 场景选择标准

即使场景暂时未定，也建议按下面标准筛选：

1. 场景里必须有清晰的吧台或桌面边界。
2. 两台机器人都能接近交接区，但初始站位不能互相穿插。
3. 交接区上方应有足够的手臂活动空间。
4. 取杯区和出餐区最好都在稳定平面上。
5. 场景应允许加入少量命名 prim 作为锚点。

不建议第一版就选择：

- 空间过于狭窄的咖啡厅
- 有大量动态行人或门的场景
- 必须同时完成导航、避障、交接、摆放的复杂场景


## 必须具备的场景锚点

无论最终用哪个 USD，建议场景里都提供以下 prim：

- `RobotSpawnA`
- `RobotSpawnB`
- `CupSpawn`
- `HandoverZone`
- `ServeZone`
- `ViewerAnchor`

这些锚点的职责如下：

- `RobotSpawnA`：机器人 A 初始位姿
- `RobotSpawnB`：机器人 B 初始位姿
- `CupSpawn`：咖啡杯出生点
- `HandoverZone`：交接动作的目标空间区域
- `ServeZone`：最终放置区域
- `ViewerAnchor`：默认观察视角或参考视点

推荐做法：

- 锚点直接做成空 `Xform`
- 名称固定，位姿由场景决定
- 代码只按名字找 prim，不再硬编码世界坐标


## 机器人角色建议

建议保留当前双机器人结构：

- `robot`
- `remote_robot`

但在任务语义层定义两个角色：

- `giver_robot`
- `receiver_robot`

第一版可以先直接约定：

- `robot` = giver
- `remote_robot` = receiver

之后如果需要支持“本机控制 receiver / 对端控制 giver”，
可以再增加基于 `NETWORK_CFG` 的角色映射层，而不是把任务逻辑写死在 `robot` 名称上。


## 任务阶段设计

建议把任务拆成四个阶段：

### 阶段 0：初始化

- 机器人按锚点放置
- 杯子按 `CupSpawn` 放置
- viewer 按 `ViewerAnchor` 初始化

### 阶段 1：取杯

- giver 接触并抓起杯子
- 杯子离开桌面或杯架

判定信号建议：

- giver 末端与杯子接近
- 杯子高度超过阈值
- 杯子在 giver 控制范围内稳定一段时间

### 阶段 2：递杯到交接区

- giver 将杯子移动到 `HandoverZone`
- 杯子姿态保持基本直立

判定信号建议：

- 杯子中心进入交接区 AABB 或球形区域
- 杯体倾斜角小于阈值

### 阶段 3：receiver 接杯

- receiver 与杯子建立稳定接触或抓持
- giver 释放杯子

判定信号建议：

- receiver 末端接近杯子
- giver 抓持状态解除
- 杯子未掉落

### 阶段 4：放到出餐区

- receiver 将杯子移动到 `ServeZone`
- 杯子放稳

判定信号建议：

- 杯子进入 `ServeZone`
- 杯子线速度和角速度足够小
- 杯体保持直立


## 成功条件建议

不要把 success 只写成最终一步，建议拆成阶段事件：

- `pickup_success`
- `handover_zone_reached`
- `handover_success`
- `serve_success`

最终 episode success：

- `serve_success == True`

这样做的好处：

- 更容易调试
- 更容易加入中间奖励
- 更容易知道失败卡在哪一段


## 失败条件建议

第一版建议至少包含这些失败条件：

- 杯子掉到地面
- 杯体倾倒超过阈值
- 超时
- 两台机器人底座或主要身体严重穿模

可以暂时不加太复杂的条件，例如：

- 手部微小碰撞惩罚
- 柜台边缘轻微接触惩罚


## 建议使用的交互物体

第一版推荐使用简化刚体杯子：

- 带盖咖啡杯
- 轻量、凸包碰撞
- 重心靠中下部

不建议第一版直接上：

- 开口杯
- 带真实液体
- 吸管、托盘、多杯同时操作

如果暂时没有咖啡杯模型，可以先用替代物：

- 小圆柱
- 带把手的简化杯
- 低多边形杯体


## 观测设计建议

如果继续沿用当前 teleop 结构，第一版至少需要这些观测：

- giver 机器人关节状态
- receiver 机器人关节状态
- giver / receiver 末端位姿
- 杯子位姿
- 杯子线速度与角速度
- `HandoverZone` 位姿
- `ServeZone` 位姿

如果后面要做 RL，再额外考虑：

- 当前阶段 one-hot
- giver 到杯子的相对位姿
- receiver 到杯子的相对位姿
- 杯子到目标区的相对位姿


## 奖励设计建议

当前旧任务没有奖励模块，新任务如果要训练，建议做分阶段奖励：

- giver 接近杯子
- giver 成功抓起杯子
- 杯子移动到交接区
- receiver 接近交接区内的杯子
- receiver 成功接杯
- receiver 将杯子移动到出餐区
- 最终稳定放置成功

同时加一些负项：

- 杯子掉落
- 杯体倾倒过大
- 长时间无进展


## 对现有代码的改造建议

### 建议新增文件

建议不要继续堆在老的传送带任务文件上，推荐新增：

- `locomanipulation_g1_cafe_handover_env_cfg.py`
- `mdp/cafe_handover_events.py`
- `mdp/cafe_handover_phases.py`
- `mdp/cafe_handover_terminations.py`
- `mdp/cafe_handover_observations.py`

### 不建议复用的旧逻辑

以下逻辑高度绑定传送带，不建议直接复用：

- 基于 `ConveyorBelt_A08_06` 的 bbox 对齐
- 传送带表面速度初始化
- 测试箱子出生点和同步逻辑

### 可以复用的旧结构

这些结构可以保留：

- 双机器人 scene 组织方式
- `robot / remote_robot` 双机实体
- G1 upper-body IK action
- XR / ZeroMQ teleop 设备配置
- viewer 跟随当前本机控制机器人这套思路


## 推荐代码接口

建议新任务里的事件函数按下面方向设计：

- `place_robots_from_named_prims`
- `place_object_from_named_prim`
- `align_viewer_to_named_prim`
- `task_phase_index`
- `task_phase_one_hot`

建议新任务里的终止/成功函数：

- `cup_dropped`
- `cup_tilt_exceeded`
- `handover_success`
- `serve_success`

建议避免新任务继续出现：

- 场景名字硬编码
- 传送带 prim 名硬编码
- 旧任务物体名直接复用为 `test_box`


## 第一版最小落地方案

如果希望尽快起一个可运行版本，建议第一版只做：

1. 保留 G1。
2. 保留双机器人。
3. 保留 teleop。
4. 新建一个简化咖啡厅或吧台场景。
5. 场景中只放一个杯子。
6. 用命名锚点替代所有硬编码坐标。
7. 先实现抓起、交接、放到出餐区三段。

第一版暂时不要做：

- 液体
- 多杯调度
- 顾客角色
- 移动导航
- 复杂障碍避让


## 第二版可扩展方向

等第一版稳定后，再考虑这些增强：

- 杯托或托盘
- 两杯连续交接
- 吧台上障碍物
- 单手机器人和双手机器人混合协作
- 顾客取餐区更远，需要更明显的传递过程
- 轻量导航或小范围底盘调整


## 当前未决策项

在开始写代码前，建议最终确认以下几点：

1. 场景是否隔着柜台。
2. 机器人是否继续使用 G1。
3. 是否保留双机 teleop。
4. 最终目标是“交接成功”还是“交接后放入出餐区”。
5. 杯子是简化刚体还是已有咖啡杯模型。


## 当前实现进度（2026-05-15）

当前仓库里已经落了一套可继续扩展的咖啡厅双机递杯骨架，重点不是美术完成度，
而是先把任务接口、场景锚点和双机协作链路收口。

- 已注册占位任务：`Isaac-CafeHandover-Locomanipulation-G1-Abs-v0`
- 已注册模板场景任务：`Isaac-CafeHandover-Locomanipulation-G1-Template-v0`
- 已新增主环境配置：`locomanipulation_g1_cafe_handover_env_cfg.py`
- 已新增模板场景接入层：`locomanipulation_g1_cafe_handover_template_env_cfg.py`
- 已新增模板场景文件：`cafe_handover_scene_template.usda`
- 已保留双 G1、双 teleop、ZeroMQ 杯子同步链路
- 已固定逻辑锚点名：`RobotSpawnA`、`RobotSpawnB`、`CupSpawn`、`HandoverZone`、`ServeZone`、`ViewerAnchor`
- 已在启动时打印锚点存在状态，便于后面接正式咖啡厅 USD
- 已把 fallback 可视化标记隔离到 `TaskDebug/*_FallbackMarker`，避免和正式场景锚点重名
- 已补上“任务阶段状态”观测：`task_phase_index`、`task_phase_one_hot`、`pickup_success`、`handover_zone_reached`、`handover_success`、`serve_success`

当前这套阶段状态还是“瞬时启发式”实现，不是带记忆的状态机。也就是说：

- 它根据当前杯子位置、姿态、交接区/出餐区位置、两台机器人末端与杯子的相对距离来推断阶段
- 它适合先做 teleop 调试、日志观察和后续奖励接口占位
- 它还没有接入真实抓持、接触或“giver 已释放 / receiver 已接管”的强语义信号

后续如果正式场景和抓持信号稳定下来，建议把阶段状态升级成“可锁存”的任务状态机，
避免阶段在边界条件下前后抖动。


## 结论

当前最值得推进的路线是：

- 新建一个咖啡厅双机递杯任务
- 保留 G1 与双机 teleop
- 使用命名锚点驱动场景接口
- 第一版先做“刚体咖啡杯交接”

这样既能保持现有系统复用率，又不会把新任务继续绑死在旧传送带场景上。
