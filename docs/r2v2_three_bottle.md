# 完整三瓶搬运流程

机器人从初始站立开始，搬箱到第二桌，依次取出三瓶、运输并放到货架，完成两次回程，最后撤离。完整示范为 **901.80 秒，50 Hz，共 45090 帧**。

搬箱段复用已有双腕搬箱策略。三瓶操作由 Loco 控制身体，腕轨迹经有界 IK 生成手臂关节 PD 目标，手指使用独立 Ruckig 控制。整个仿真在同一个世界里连续执行。

## 看结果

[四倍速三视角完整视频](media/three_bottle_4x.mp4)约 3 分 46 秒。视频、示范数据、运行模型和场景均随本分支提供，直接 `git clone` / `git pull` 即可获取。

在 MuJoCo 中查看示范：

```bash
.venv-three-bottle/bin/python deploy_mujoco/r2v2_three_bottle.py --replay --speed 4
```

## 安装与运行

在仓库根目录建立本流程的 Python 3.12 环境：

```bash
uv venv --no-project --python 3.12 .venv-three-bottle
uv pip install --python .venv-three-bottle/bin/python -r requirements-three-bottle.txt
```

完整运行，带 MuJoCo 窗口：

```bash
.venv-three-bottle/bin/python deploy_mujoco/r2v2_three_bottle.py --output output/three_bottle
```

完整运行，无窗口：

```bash
.venv-three-bottle/bin/python deploy_mujoco/r2v2_three_bottle.py --headless --output output/three_bottle
```

首次检查可追加 `--duration 1`，并使用一个新的输出目录。完整运行的进度以终端打印的物理时间和阶段为准。

## 文件结构

| 文件或目录 | 用途 |
| --- | --- |
| `deploy_mujoco/r2v2_three_bottle.py` | 运行和回放入口 |
| `src/r2v2_deploy/tasks/three_bottle/` | 流程、身体控制、IK、抓取、交接、上架和回程 |
| `configs/three_bottle/` | 工位、手型、回程和第三瓶参考点 |
| `assets/three_bottle/scene.xml` | 完整三瓶场景 |
| `assets/three_bottle/models/wrist_reference.onnx` | 腕参考与历史接口模型 |
| `policy/r2v2_skills/20261026/` | 复用的 Loco 和搬箱模型 |
| `assets/three_bottle/*prior.npz` | IK 姿态与后续瓶子参考 |
| `assets/three_bottle/demonstration_50hz.npz` | 完整 45090 帧示范 |
| `assets/three_bottle/DATA_SCHEMA.json` | 数组、关节顺序和数据说明 |
| `assets/three_bottle/manifest.json` | 资源大小与 SHA-256 |

代码阅读顺序：`workflow.py` 的接口 → `sequence.py` 的主控制循环 → `crate_control.py` 的搬箱段 → `grasp.py`、`handoff.py`、`placement.py`、`return_path.py` 的各段目标。`arm_control.py` 负责有界 IK，`body_control.py` 负责策略与手臂目标的组合，`measurements.py` 负责接触、支撑和安全测量。

控制频率为：物理积分与 PD 1000 Hz，手指轨迹 100 Hz，身体策略与手臂目标 50 Hz。参考 FK 使用独立 MuJoCo 数据；实际机器人只通过控制总线推进。

## 输出与检查

每次运行在所选目录生成：

- `report.json`：完成结果、物理时间、三瓶完成列表及运行版本。
- `events.json`：流程切换和各瓶事件。
- `trajectory_50hz.npz`：本次实际位置、速度、身体动作、手指指令和阶段。
- `terminal_state.npz`：终点完整 MuJoCo 积分状态。

成功时 `report.json` 中 `success` 为 `true`，`completed_bottles` 为 `[1, 2, 3]`。

运行资源和数值回归检查：

```bash
.venv-three-bottle/bin/python -m unittest discover -s tests -p test_three_bottle.py -v
```

完整示范数据包含物理状态、1461 维观测、28 维已执行动作、十帧动作历史、手指命令、双腕与三瓶位姿。位置单位为米，关节角为弧度，四元数顺序为 `wxyz`。数组形状和关节地址以 `DATA_SCHEMA.json` 为准。

整理后的入口已重新执行完整 901.80 秒流程，三瓶全部上架并完成撤离。与所附示范的 45090 个 50 Hz 采样点逐项对照，位置和速度最大差均为 0；新环境的三项检查与仅含 Git 文件的启动检查均通过。结果见 [validation.json](../assets/three_bottle/validation.json)。
