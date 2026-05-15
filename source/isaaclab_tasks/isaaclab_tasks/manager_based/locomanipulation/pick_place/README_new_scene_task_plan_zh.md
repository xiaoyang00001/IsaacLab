# 新场景与新交互任务改造方案

## 目标

当前这套 `locomanipulation/pick_place` 任务，已经和以下要素强耦合：

- `warehouse.usd` / `warehouse-simple7-aligned.usda`
- `ConveyorBelt_A08_06` 等传送带 prim
- 双机器人协作结构：`robot` / `remote_robot`
- G1 上半身 IK 与 XR / ZeroMQ teleop
- 基于 pick-place 的成功判定

如果准备切换到一个新的场景，并改成新的机器人交互任务，最稳的做法不是直接覆盖现有配置，
而是新开一套 task 配置，保留旧任务可运行。


## 总体建议

建议按下面的优先级改：

1. 先保留机器人和 teleop 框架，只替换场景与任务逻辑。
2. 再决定是否还保留双机器人协作。
3. 最后再决定是否更换机器人资产本身。

原因是：

- 场景切换主要影响 `SceneCfg` 和 `EventsCfg`
- 任务切换主要影响观测、成功判定、奖励
- 换机器人会连带影响 IK、关节名、link 名、retargeter，改动面最大


## 推荐做法

### 1. 不要直接改老任务

不要直接在下面这个文件上持续堆修改：

- `locomanipulation_g1_env_cfg.py`

建议新建：

- `locomanipulation_g1_<new_task>_env_cfg.py`

然后在 `__init__.py` 里注册一个新的 task id。

这样做的好处：

- 老任务还能继续回归测试
- 新任务可以独立迭代
- 场景切换失败时更容易回退


### 2. 场景里放“命名锚点”，不要再靠 bbox 猜位置

当前实现里，机器人、测试箱、viewer 都依赖传送带 bbox 做相对摆位。
这适合“围绕传送带”的任务，但不适合通用新场景。

更推荐在新场景 USD 里直接放这些命名锚点：

- `RobotSpawnLocal`
- `RobotSpawnRemote`
- `ObjectSpawn0`
- `Goal0`
- `ViewerAnchor`

代码只按 prim 名读取这些锚点的世界位姿，然后：

- 本地机器人放到 `RobotSpawnLocal`
- 远端机器人放到 `RobotSpawnRemote`
- 交互物体放到 `ObjectSpawn0`
- 成功判定参考 `Goal0`
- 初始视角参考 `ViewerAnchor`

这样以后换场景，通常只需要换 USD，不需要再改一堆 magic number。


### 3. 优先保留 G1 与现有 teleop

如果你的目标只是“换场景、换任务”，建议先继续使用当前 G1 配置。

先不动这些模块：

- `G1_29DOF_CFG`
- `G1_UPPER_BODY_IK_ACTION_CFG`
- `OpenXRDeviceCfg`
- `ZeroMqGameSubDeviceCfg`

先把新任务跑通，再考虑换机器人。


## 当前代码里需要重点改的部分

### 场景层

主要入口：

- `locomanipulation_g1_env_cfg.py`

重点位置：

- `LocomanipulationG1SceneCfg`
- `background`
- `test_box` / `test_box1` / `object`
- `robot` / `remote_robot`

如果新任务不再使用传送带测试箱，那么：

- 删除或替换 `test_box`
- 删除或替换 `test_box1`
- 根据新任务定义新的交互物体


### 事件层

当前耦合最重的是 `EventsCfg` 和 `mdp/events.py`。

现在这些逻辑都偏向“传送带任务”：

- 对齐机器人到传送带
- 对齐测试箱到传送带
- 对齐 viewer 到传送带
- 给传送带施加表面速度

如果新任务没有传送带，这些逻辑应该整体换成：

- 按锚点放置机器人
- 按锚点放置物体
- 按锚点初始化 viewer
- 如有必要，再添加任务专属事件

建议新增而不是硬改的函数类型：

- `place_robots_from_named_prims`
- `place_objects_from_named_prims`
- `align_viewer_to_named_prim`


### 观测层

当前观测主要围绕：

- 本地机器人状态
- 远端机器人状态
- 双手末端位姿
- 手部关节状态
- 基于旧 object 的任务观测

如果新任务变化较大，需要重新审视：

- 是否还要观测远端机器人
- 是否需要目标物体状态
- 是否需要目标点或目标姿态
- 是否需要接触状态、阶段状态、门把手状态等


### 终止与成功判定

当前成功判定仍然是 pick-place 风格：

- `task_done_pick_place`

如果新任务改成以下类型之一，就应该单独实现成功函数：

- 搬运到目标区域
- 双机交接物体
- 开门 / 拉抽屉
- 按按钮 / 拨动开关
- 协同放置

建议每个新任务都有专属 success 条件，不复用旧的 pick-place 完成逻辑。


### 奖励层

当前配置里：

- `rewards = None`

这表示当前更偏 teleop / 任务执行配置，而不是完整 RL 训练任务。

如果新任务后面要用于训练，建议补：

- 接近目标奖励
- 抓取稳定奖励
- 物体朝目标移动奖励
- 最终成功奖励
- 失败惩罚


## 是否保留双机器人

### 方案 A：继续保留双机器人

适合：

- 双人 teleop
- 双臂或双机协同搬运
- 交接任务

保留以下结构：

- `robot`
- `remote_robot`
- `upper_body_ik`
- `remote_upper_body_ik`
- `xr`
- `xr2`


### 方案 B：先收缩成单机器人

适合：

- 单人 teleop
- 先验证新场景与新交互任务
- 想降低维护成本

可以先裁掉：

- `remote_robot`
- `remote_upper_body_ik`
- `xr2`
- 远端机器人相关观测

如果后面确认需要协作，再把双机结构加回来。


## 如果还要换机器人

只有在以下问题出现时，才建议同时换机器人：

- G1 的手型不适合新任务
- G1 的关节空间受限
- 现有 retargeter 无法满足交互需求

如果要换机器人，影响面会扩大到：

- 机器人资产 cfg
- IK controlled joints
- 目标末端 link 名
- 手部关节名
- XR retargeter
- 成功判定中的 task link
- 观测里的 link / joint 名

所以更推荐分两步：

1. 先用 G1 跑通新场景与新任务
2. 再替换机器人资产


## 建议的落地顺序

1. 新建一个新的 env cfg 文件，不覆盖老文件。
2. 新建一个新的 scene cfg，只保留新场景真正需要的资产。
3. 在 USD 里加入命名锚点，改成按锚点摆位。
4. 删除 conveyor 专属事件，换成新任务专属事件。
5. 改成功判定和终止条件。
6. 视需要补观测。
7. 如果要训练，再加奖励。
8. 最后才考虑是否更换机器人。


## 推荐的最小首版

如果你想最快得到一个可运行的新任务，建议第一版只做这些：

- 保留 G1
- 保留现有 XR / ZeroMQ teleop
- 保留双机器人结构或先裁成单机器人二选一
- 替换场景 USD
- 用命名锚点摆位
- 换一个新的交互物体
- 写一个新的 success 条件

不要在第一版就同时做：

- 换机器人
- 改 IK 架构
- 改 teleop 协议
- 改复杂奖励


## 下一步建议

如果已经明确以下四件事，就可以直接开始落代码：

- 新场景是什么
- 新任务是什么
- 是否保留双机器人
- 是否继续用 G1

一旦这四项确定，就可以直接把新 task 的骨架文件创建出来。
