# R2V2 75° 手型抓取基线与 FSM 接入记录

记录日期：2026-09-07。分支：`r2v2-fsm`。
用户已选定 75° 对掌初始化，并确认固定手腕圆柱抓取演示。
本次只固化资产、控制器、相对位姿和轨迹记录；**没有实现全身 Reach + 手部抓取 FSM**。

## 已保存的基线

版本控制中的记录目录：
`reference_motion_bank/r2v2_grasp/thumb75_cylinder40mm_100g/`。
左右手分别保存在 `left/`、`right/`，是两次独立的固定手腕试验，不是同时双手抓同一物体。

- `grasp_trajectory.npz`：0–15 s，每 10 ms 一帧，包含初始帧，共 1501 帧；无需 pickle。
- `grasp_record.json`：字段约定、完整手部配置、实验参数、初始/握持/释放关键帧、
  验收结果、轨迹及模型/生成代码 SHA-256。以后修改默认配置，不会改写这份基线快照。
- `metrics.json`：完整抓取验收报告。
- `trace.json`：原有物体位置与接触力日志，保留用于与已确认的视频核对。

这是用同一配置重新运行并补全关节记录的基线；逐项核对的物理验收指标与
此前左手视频、右手无渲染试验一致。视频仍在本地
`artifacts/r2v2_cylinder/left_thumb75_40mm_100g_final/cylinder.mp4`，
未放进 Git；新机器可通过下方命令重现。

## 手型及时间轨迹

运行配置：`deploy_mujoco/config/r2v2_hands.yaml`。
每手六路参考按下表排序，左右使用各自模型关节轴，同样的正角度，不额外取负号。

| 通道后缀 | 作用 | open / rad | closed / rad |
| --- | --- | --- | --- |
| `thumb_metacarpal_joint` | 拇指对掌 | 1.3089969389957472（75°） | 同左 |
| `thumb_proximal_joint` | 拇指弯曲 | 0.04 | 0.60 |
| `index_proximal_joint` | 食指 | 0.03 | 0.85 |
| `middle_proximal_joint` | 中指 | 0.03 | 0.90 |
| `ring_proximal_joint` | 无名指 | 0.03 | 0.90 |
| `pinky_proximal_joint` | 小指 | 0.03 | 0.85 |

`0` = 张开，`1` = 闭合。Ruckig 在 100 Hz 生成位置、速度、加速度连续参考，
重复二值指令不重启轨迹。拇指对掌参考始终是 75°，其余五路收拢。
主动关节力矩上限依次 `[0.15, 0.30, 0.35, 0.35, 0.35, 0.35] Nm`。
拇指远端按近端 1 倍联动，其余四指远端按近端 1.155 倍联动。

本次实验：0.5 s 下发闭合，参考约在 2.19 s 到达闭合手型；5–7 s 托台下降，
7–12 s 无支撑握持，12 s 下发张开，15 s 结束。
**2.19 s 只是参考轨迹完成时间，不是“抓住”的判定时间。**
圆柱阻挡使实际关节角不等于空载 closed 目标，这是带限力矩 PD 的正常接触结果。
不要将实测关节轨迹直接当成新目标，也不要开环回放力矩。

## 圆柱与手腕相对位姿

定义 `W` 为世界坐标系、`H` 为 `left_hand_roll_link` 或 `right_hand_roll_link`、
`C` 为 `test_cylinder`。H 是模型的**手腕刚体坐标系，不是掌心、指尖或 Reach 的 TCP**。
C 原点位于圆柱中心，局部 Z 为轴线；直径 0.04 m、高 0.12 m、质量 0.10 kg。

采用列向量，`T_A_B` 把 B 坐标变换到 A；平移以 A 为基准，四元数顺序为 `wxyz`。
初始目标关系如下：

| 手 | 圆柱中心在手腕坐标系的位置 / m | 圆柱相对手腕四元数 wxyz |
| --- | --- | --- |
| 左 | `[0.145, -0.035, 0]` | `[1, 0, 0, 0]` |
| 右 | `[0.145, +0.035, 0]` | `[1, 0, 0, 0]` |

两者初始相对姿态的 XYZ roll/pitch/yaw 都为 0°。
固定夹具中两手腕世界位置是 `[-0.13, ±0.15, 0.45] m`、姿态均为单位四元数；
圆柱世界位置分别为 `[0.015, +0.115, 0.45]` 和 `[0.015, -0.115, 0.45] m`。
这些世界位置只是夹具布局，后续全身任务应使用相对变换，不要照抄世界坐标。

闭合过程中圆柱会移动并倾斜，并非一直保持初始相对位姿。
9.5 s 握持中点测得：左手相对位置约 `[0.135631, -0.030099, 0.000998] m`，
右手约 `[0.135978, +0.030026, 0.001471] m`。
对应四元数及全部时序均保存在 JSON 关键帧和 NPZ 中。
**接近目标应先使用 initial 关系，不应把握持后的偏移当成初始摆放标定。**

给定物体世界位姿，求期望手腕世界位姿：

```python
T_world_wrist_target = T_world_cylinder @ np.linalg.inv(T_wrist_cylinder_initial)
```

如果 Reach 控制的是另一个末端 E，还必须标定模型中的 `T_wrist_E`，再转换：

```python
T_world_E_target = T_world_wrist_target @ T_wrist_E
```

送给策略前还要按实际命令接口转换世界/本体坐标，不能只因都叫“末端目标”就
混用坐标系。均匀圆柱绕自身轴的方向不可仅靠形状唯一确定，视觉侧应约定一致的 C 轴系。

## NPZ 字段与读取

| 字段 | 内容 |
| --- | --- |
| `reference_joint_names` | 六路主动关节完整名称，参考和力矩的列顺序 |
| `measured_joint_names` | 11 个主动及从动关节名称，实测值的列顺序 |
| `time_s`, `command`, `phase` | 时间、该步执行的二值指令、实验时间段标签 |
| `reference_q_rad`, `reference_qd_rad_s`, `reference_qdd_rad_s2` | 六路参考 q / dq / ddq |
| `measured_q_rad`, `measured_qd_rad_s` | 11 个物理关节实测 q / dq |
| `commanded_torque_Nm`, `actuator_torque_Nm` | 六路裁剪后的控制输入、执行器输出力矩 |
| `T_world_wrist`, `T_world_cylinder`, `T_wrist_cylinder` | 三组完整 4×4 位姿矩阵 |
| `contact_part_names`, `contact_normal_force_N` | 掌部、拇指、四指的物体接触法向力 |
| `support_contact`, `floor_contact` | 物体与托台/地面是否接触 |

```python
from pathlib import Path
import json
import numpy as np

root = Path("reference_motion_bank/r2v2_grasp/thumb75_cylinder40mm_100g/left")
meta = json.loads((root / "grasp_record.json").read_text())
with np.load(root / "grasp_trajectory.npz", allow_pickle=False) as data:
    names = data["reference_joint_names"].tolist()
    initial_relation = data["T_wrist_cylinder"][0].copy()
    q_reference = data["reference_q_rad"].copy()
```

记录器不改变在线物理数据：在独立 `MjData` 上做 FK，得到与该行积分后 qpos
一致的位姿。原 `trace.json` 沿用旧版步内接触/物体位置缓存，可能相差一个物理步。
力矩、接触力及指令代表刚结束的物理步，不是整个 10 ms 窗口平均值。
例如 0.50 s 行尚为闭合前状态，0.51 s 行才记录到新闭合指令；12 s 同理。
`phase` 是实验时间段标签，不是接触反馈识别出的 FSM 状态。

## 验证与复现

左/右手均保持 5 s，最大位移分别 2.61 / 1.96 mm，拇指与对侧手指有效接触比例
均为 100%，握持期间无托台或地面接触；松手后下落约 177.5 / 179.5 mm 到托台。
物体自由运动，无焊接、吸附、外力或物体位姿重置。初始自接触约 0.171 mm，
无数值警告。详细局限见 [圆柱试验说明](r2v2_cylinder_grasp.md)。

归档时 32 项自动测试通过，覆盖原有模型/控制/抓取验证、记录不干扰物理、
坐标变换方向、时间与关节字段一致性、NPZ 无 pickle 读取和归档校验值。

在仓库根目录：

```bash
.venv-r2v2/bin/python -m pytest -q tests
.venv-r2v2/bin/python deploy_mujoco/r2v2_cylinder_test.py --video
.venv-r2v2/bin/python deploy_mujoco/r2v2_cylinder_test.py --side right --video
```

每次新运行都会生成完整轨迹，默认写入忽略的 `artifacts/`；不要覆盖这份已确认基线。
视频、环境、训练 checkpoint 不随本次 Git 提交，模型原始资产及小型基线数据随提交保存。

## 下一步 FSM 接入约定（待开发）

建议流程：准备张手 → 全身 Reach 到预抓取位姿 → 手部闭合 → 验证握持 →
抬起/搬运 → 放置并张手 → 撤离；异常进入单独失败处理。

1. 身体保持 28 维策略接口；每只手单独接收 0/1 指令，不增加 RL 手指动作或观测。
   用按名映射分开身体与手部通道，禁止直接把前 28 个 qpos 当作身体状态。
2. 固定 `T_wrist_E`，先检查 Reach 到达与速度收敛，再闭合；手型/关节参考延续当前状态，
   不在 FSM 切换时重置物理关节或把物体瞬移到掌心。
3. 使用 `DualHandControl.command/update/apply` 保持现有轨迹与限力矩接口。
   当前 `CLOSED` 只代表达到预设手型，不等于 HOLDING；有物体阻挡时可能一直为 CLOSING。
   抓住、空握、失稳、超时须由新增的任务层反馈判定，不能只等 CLOSED。
4. 固定手腕演示中的托台下降只是验证方法；全身任务应靠手腕抬升形成离台，
   同时验证身体平衡、手–环境碰撞、握持稳定、接近路径和重力方向。
5. 当前样例不保证其他尺寸、质量、目标方向或位置偏差下可抓住；先做最小仿真闭环，
   再扩展接触减速、逐指限力和鲁棒性测试，不直接上实机。
