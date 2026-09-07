# R2V2 灵巧手：第一阶段验证

后续已增加 [固定手腕圆柱抓握试验](r2v2_cylinder_grasp.md)，与本文空载验收分开运行。

分支：`r2v2-fsm`。本阶段完成新资产导入、手部驱动/联动、二值命令生成
平滑关节参考，以及固定手腕下的物理测试。**不加载策略网络、不连接硬件、
不验证站立和抓取成功**。原有 G1 仿真入口尚未完成 R2V2 接线，请使用下面的新入口。

## 环境与运行

在仓库根目录执行。独立小环境不依赖 Torch、Unitree SDK，也不改训练环境：

```bash
uv venv --python 3.11 .venv-r2v2
uv pip install --python .venv-r2v2/bin/python -r requirements-r2v2-sim.txt
.venv-r2v2/bin/python -m pytest -q tests/test_r2v2_hands.py
```

无窗口验证（自动跑 32 秒模拟时间，退出码 0 表示所有验收检查通过）：

```bash
.venv-r2v2/bin/python deploy_mujoco/r2v2_hand_test.py --headless
```

录制左右手特写视频（Linux 无 DISPLAY 时自动使用 EGL）：

```bash
MUJOCO_GL=egl .venv-r2v2/bin/python deploy_mujoco/r2v2_hand_test.py --headless --video
```

有图形桌面时，交互调试：

```bash
.venv-r2v2/bin/python deploy_mujoco/r2v2_hand_test.py --interactive
```

在 MuJoCo 窗口按 `L`/`R` 切换左/右手，`O` 双手张开，`C` 双手闭合。
关闭窗口退出。不带 `--interactive` 的窗口模式播放自动验证序列，不响应开合按键。
`--duration` 可以限制运行时间；自动模式少于完整 32 秒时，可能因检查点未完成而
返回失败，不能把短预览当成完整验收。交互模式只检查运行中的数值/轨迹指标，
不检查自动序列是否完成。

默认输出到 `artifacts/r2v2_hands/<UTC时间>/`，可用 `--output <新目录>` 指定。
程序拒绝覆盖非空目录。输出包括：

- `hands.mp4`：开启 `--video` 后生成，1280×720 / 30 fps。
- `open.png`、`closed.png`、`reversal.png`：视频模式的关键帧。
- `metrics.json`：验收结果、阈值、完整配置、事件、检查点、关节误差及速度峰值。
- `trajectory.npz`：100 Hz 的测量位置/速度、参考位置/速度，`columns` 给出列名。

## 控制接口与参数

配置：`deploy_mujoco/config/r2v2_hands.yaml`。角度为 rad，时间为 s。
每手六路顺序明确指定：拇指对掌、拇指弯曲、食指、中指、无名指、小指。
左右关节轴直接使用源模型，不能通过随意取负号实现镜像。
当前左右手拇指对掌初始角均为 75°（1.3089969389957472 rad），`open` 和
`closed` 的第一项一致；二值开合只改变其余五路参考，对掌目标始终保持 75°。

```python
from common.r2v2_hand_control import DualHandControl

hands = DualHandControl(model, data, config)  # 已初始化模型与手部位置
hands.command("left", 1)   # 1 = 闭合到预设手型
hands.command("right", 0)  # 0 = 张开
# 每个 control_dt (0.01 s) 调用一次：
hands.update()
# 每个 simulation_dt (0.002 s) 调用一次，随后 mj_step：
hands.apply(data)
```

二值指令是保持型命令，重复发送不重置轨迹。运动中反向时，Ruckig 从当前
参考位置、速度、加速度重新规划，而不是强行将速度归零。
这里只使用 Ruckig 的本地点到点生成，不调用云端 waypoint API。
参考位置通过带速度前馈的 PD 转换成力矩，并在每个物理步做力矩裁剪。

身体保持独立的 `StateAndCmd(28)` / `PolicyOutput(28)`；手部状态不混入 RL
观测或动作。`JointMap` 根据关节名称与实际执行器关联查找位置、速度和控制索引，
因此左手手指插在左右臂之间也不会导致右臂索引错位。

状态只包括 `OPEN / OPENING / CLOSED / CLOSING`。
**CLOSED 不是 HOLDING，不表示抓住物体。**本阶段也没有接触制动和失速超时状态；
不能拿这一空载控制器直接做真实抓取。

## 模型与轨迹设置

- 原始 XML、URDF、网格完整保留在 `r2v2_description/source/r2v2_with_hand/`。
- 新增每手六个主动执行器；十个从动关节由 URDF 的 mimic 自动生成等式约束。
- 拇指远端/近端比例 1；其余四指比例 1.155。配置检查同时验证从动关节的限位和速度。
- 保持原始关节 damping、armature、惯量和网格。仅增加测试所需的执行器、
  联动、显式相邻碰撞排除和求解设置，详见资产目录 README。
- 固定手腕只是测试夹具；完整机器人构建函数仍保留浮动基座、28 个身体驱动。
- `open/closed` 是带限位余量的预设手型，已进行单一圆柱样例测试，
  不是适用于任意瓶子的通用抓握标定。
- `trajectory.max_velocity/max_acceleration/max_jerk` 限制的是**参考轨迹**。
  物理关节存在 PD 跟踪误差，速度单独记录，并按当前 `0.10 rad/s` 跟踪容差检查；
  全部 22 个物理关节还必须满足 URDF 的速度上限。它不是接触情况下的硬限速保证。
- PD、力矩上限、开合手型均是仿真调试参数，不是实机标定值。

## 自动序列与验收

自动序列包含左手独立开合、右手独立开合、双手同步、19–21 秒连续中途反向，
最后双手张开。每个控制周期重复下发当前命令，验证幂等性。

18 项 pytest 用例覆盖模型维数、按名映射、URDF 联动、固定夹具/完整机器人
手部运动学一致性、身体 28 维接口不受影响、命令幂等性、反向连续性、非法
输入、75° 对掌参考保持，以及完整的 32 秒物理实验。实验另检查 18 项指标；阈值由 YAML 给出。

2026-09-07，75° 默认手型，MuJoCo 3.3.7 / Python 3.11 验证结果：

- 18 项空载自动测试全部通过，32 秒演示的 18 项检查全部通过；含圆柱测试共 25 项通过。
- 最大主动关节跟踪误差：左 0.029895 rad，右 0.026677 rad（约 1.71° / 1.53°）。
- 最大联动误差：0.00000839 rad（约 0.00048°）。
- 最大主动关节实际速度：左 0.84733 rad/s，右 0.84708 rad/s。
- 关节越界、命令切换瞬间参考状态跳变：均为 0。
- 最大自接触穿透约 0.171 mm，发生在初始掌部/拇指碰撞几何；没有关闭该碰撞。
- MuJoCo 数值警告：0；六个开/合检查点全部通过。

结果保存于 `artifacts/r2v2_hands/thumb75_regression/`。

这些结论只适用于当前模型、配置和已测试序列；不等同于任意手型无自碰撞，
也不等同于全身稳定、瓶子抓取成功或实机可用。

下一阶段：固定手腕加入瓶子，标定抓握手型，开发逐指接触减速/限力保持、
空握和超时判定，再接双臂 Reach。硬件接入前必须确认厂商型号及 SDK 的
位置/速度/力标定关系。
