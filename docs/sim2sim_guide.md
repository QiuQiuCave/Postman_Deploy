# FSMDeploy_G1 Sim2Sim 使用指南

本文档面向**策略消费端**：你已经训练好了一个 Unitree G1 的控制策略（`.pt` TorchScript 或 `.onnx`），想把它接到这个仓库里，在 MuJoCo 中跑 sim2sim 验证，最后再上真机。本文覆盖：

1. 项目是做什么的、怎么跑起来
2. 一个策略被加载、喂观测、跑出动作、落到仿真上的**完整数据通路**
3. 如何把你自己的策略文件接入（最小改动清单）
4. 观测/动作空间约定、关节顺序踩坑点
5. 常见问题排查

---

## 1. 项目定位

本仓库是 Unitree G1（29-DoF）的**多策略部署框架**。核心是一个有限状态机（FSM），每个"技能"（走路、跳舞、功夫、模仿…）是一个 `FSMState` 子类；用户通过手柄或键盘发出切换指令，FSM 在不同策略之间切换，每个策略负责自己的观测构造、模型推理和动作输出。

部署目标有两个：

- `deploy_mujoco/`：MuJoCo 仿真（sim2sim 验证用）
- `deploy_real/`：真机，通过 `unitree_sdk2_python`

两者共享同一套 `FSM` + 策略代码，只是"外壳"不同（一个读 MuJoCo 状态、一个读 DDS 话题）。**这正是 sim2sim 的意义**：策略代码不变，换外壳运行。

---

## 2. 环境安装

```bash
conda create -n robomimic python=3.8
conda activate robomimic

# PyTorch（sim2sim 只需要 CPU 也够，但 GPU 版本无妨）
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install numpy==1.20.0
pip install onnx onnxruntime mujoco pyyaml

# 手柄输入（可选，没有就用 keyboard_input 版本）
pip install pygame

# 真机才需要，sim2sim 可以跳过
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e .
```

仓库根目录结构：

```
FSMDeploy_G1/
├── FSM/                    # 状态机核心
│   ├── FSM.py              # FSM 主类，持有所有策略实例
│   └── FSMState.py         # 策略基类
├── common/
│   ├── ctrlcomp.py         # StateAndCmd / PolicyOutput 状态总线
│   ├── utils.py            # FSMStateName / FSMCommand 枚举、工具函数
│   ├── path_config.py      # 注入 PROJECT_ROOT 到 sys.path
│   ├── joystick.py         # Xbox 手柄封装
│   └── ...
├── policy/
│   ├── loco_mode/          # 走路（.pt，最稳定的参考实现）
│   ├── fixedpose/          # 回零位
│   ├── passive/            # 阻尼保护
│   ├── dance/              # Charleston dance（.onnx）
│   ├── kungfu/ kungfu2/ kick/
│   ├── beyond_mimic/
│   └── gae_mimic/          # GAE_Mimic 动作跟踪
├── deploy_mujoco/
│   ├── deploy_mujoco.py                  # 手柄入口
│   ├── deploy_mujoco_keyboard_input.py   # 键盘入口
│   └── config/mujoco.yaml
├── deploy_real/
└── g1_description/         # G1 的 MJCF 模型与 mesh
```

---

## 3. 快速跑通 sim2sim

**有手柄：**

```bash
python deploy_mujoco/deploy_mujoco.py
```

按键映射（手柄需要先连好）：

| 按键            | 功能                |
|-----------------|---------------------|
| `L3`            | Passive（阻尼保护） |
| `Start`         | FixedPose（回零位） |
| `A + R1`        | Locomotion          |
| `X + R1`        | Dance               |
| `Y + R1`        | KungFu              |
| `B + R1`        | Kick                |
| `Y + L1`        | BeyondMimic         |
| `B + L1`        | GAE_Mimic           |
| `Select` / `F1` | 退出                |

**没有手柄：**

```bash
python deploy_mujoco/deploy_mujoco_keyboard_input.py
```

输入字符串命令（例如 `a+r1` 进走路、`vel 0.5 0 0` 设线速度、`exit` 退出）。

`deploy_mujoco/config/mujoco.yaml` 里两个关键参数：

- `simulation_dt: 0.003` — MuJoCo 物理步长（333 Hz）
- `control_decimation: 7` — 每 7 个物理步调用一次策略 → **控制频率 ≈ 47.6 Hz**（约 0.021 s/步）

> 注意：策略的 `control_dt` 必须和训练时一致，否则行为会漂。如果你的策略是 50 Hz 训练的，保持默认即可；如果是 100 Hz，就把 `control_decimation` 改为 3。

---

## 4. 数据通路全景

理解一次 `mujoco.mj_step` → 一次策略推理的完整路径，是接入新策略的前提。

```
┌─────────────────────────────────────────────────────────────────────┐
│ deploy_mujoco.py 主循环（每 sim_dt 一次）                           │
│                                                                     │
│  1. 读手柄/键盘 → 写入 state_cmd.skill_cmd / state_cmd.vel_cmd     │
│  2. 用上一轮的 (actions, kps, kds) 算 PD 力矩 → d.ctrl             │
│  3. mujoco.mj_step(m, d)                                            │
│  4. 每 control_decimation 步：                                      │
│     ├── 从 d.qpos / d.qvel 抽取 q, dq, quat, ang_vel, gravity      │
│     ├── 写入 state_cmd                                              │
│     ├── FSM_controller.run()   ◄── 见下                             │
│     └── 读 policy_output.actions/kps/kds                            │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FSM.run()                                                           │
│                                                                     │
│  NORMAL 模式：                                                      │
│    cur_policy.run()           ← 真正的观测构造 + 推理 + 写 action  │
│    next = cur_policy.checkChange()                                  │
│    if next != cur: cur_policy.exit(); cur = get_next_policy(next); │
│                    FSMmode = CHANGE                                 │
│                                                                     │
│  CHANGE 模式（下一拍）：                                            │
│    cur_policy.enter()         ← 清 buffer、复位 counter             │
│    cur_policy.run()                                                 │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│ <YourPolicy>.run()                                                  │
│                                                                     │
│  1. 从 state_cmd 读 q, dq, ang_vel, gravity_ori（都是仿真里原始值）│
│  2. 按 joint2motor_idx 把电机顺序重排成策略训练时的顺序            │
│  3. 减 default_angles、乘各 scale，拼成 obs                         │
│  4. 模型推理：self.policy(obs)  或  ort_session.run(...)           │
│  5. action * action_scale + default_angles → target_dof_pos         │
│  6. 把 target_dof_pos 按电机顺序重排回去                            │
│  7. policy_output.actions / kps / kds ← 写入                        │
└─────────────────────────────────────────────────────────────────────┘
```

两个关键数据结构（`common/ctrlcomp.py`）：

```python
class StateAndCmd:     # 仿真 → 策略
    q          # (num_joints,)     当前关节位置（MuJoCo 电机顺序）
    dq         # (num_joints,)     当前关节速度
    gravity_ori# (3,)              重力方向（base frame）
    ang_vel    # (3,)              base 角速度
    vel_cmd    # (3,)              手柄线速/角速指令（-1..1）
    skill_cmd  # FSMCommand        切策略指令

class PolicyOutput:    # 策略 → 仿真
    actions    # (num_joints,)     目标关节位置（电机顺序）
    kps, kds   # (num_joints,)     PD 增益（电机顺序）
```

**电机顺序 = MuJoCo MJCF 中 actuator 的顺序**（见 `g1_description/g1_29dof_rev_1_0.xml`）。所有进出策略的向量都要统一到这个顺序。

---

## 5. ⚠️ 关节顺序陷阱（最常见的 bug 源）

训练环境（Isaac Gym / Isaac Lab / Genesis…）里定义的关节顺序**几乎不可能**和 MuJoCo MJCF 的 actuator 顺序一致。解决办法是在 yaml 里声明一个重映射：

```yaml
joint2motor_idx: [0, 6, 12, 1, 7, 13, 2, 8, 14, ...]
# 含义：policy_order[i] 对应 motor_order[joint2motor_idx[i]]
```

以 `LocoMode` 为例（`policy/loco_mode/LocoMode.py`）：

```python
# 观测：把电机顺序的 q/dq 按 joint2motor_idx 重排成策略顺序
for i in range(len(self.joint2motor_idx)):
    self.qj_obs[i]  = self.qj [self.joint2motor_idx[i]]
    self.dqj_obs[i] = self.dqj[self.joint2motor_idx[i]]

# 动作：把策略顺序的 action 重排回电机顺序
action_reorder = loco_action.copy()
for i in range(len(self.joint2motor_idx)):
    motor_idx = self.joint2motor_idx[i]
    action_reorder[motor_idx] = loco_action[i]   # 写回
```

其他策略（如 `Dance`）用的是 `dof23_index`——只用 29 维里的 23 个关节（手腕和踝关节被锁死）——形式不同但思路一致：**索引数组 = 策略看到的关节在电机向量中的位置**。

**踩坑自检**：策略一加载就发飘、重心 NaN、或者动作在错误的关节上响应——先打印 `self.joint2motor_idx` 对比训练时的关节名顺序。

---

## 6. 接入你自己的策略（最小改动清单）

假设你要加一个叫 `MyLoco` 的策略（输出 29 维关节目标）。

### 6.1 注册枚举

`common/utils.py`：

```python
class FSMStateName(Enum):
    ...
    SKILL_MY_LOCO = 12       # 新增

class FSMCommand(Enum):
    ...
    SKILL_MY_LOCO = 12       # 新增（如果你想用新按键触发）
```

### 6.2 新建策略目录

```
policy/my_loco/
├── __init__.py
├── MyLoco.py
├── config/
│   └── MyLoco.yaml
└── model/
    └── my_policy.pt   ← 你训练出来的文件放这
```

### 6.3 写 config

参考 `policy/loco_mode/config/LocoMode.yaml`。必填字段：

```yaml
policy_path: "my_policy.pt"        # 或 onnx_path: "my_policy.onnx"
kps: [...29 个 float...]            # 电机顺序
kds: [...29 个 float...]            # 电机顺序
default_angles: [...29 个 float...] # 电机顺序，策略训练时的参考站姿
joint2motor_idx: [...29 个 int...]  # 见第 5 节

num_actions: 29
num_obs: 96                         # 按你的 obs 拼接规则算
action_scale: 0.25                  # 训练时的 action 缩放
ang_vel_scale: 1.0
dof_pos_scale: 1.0
dof_vel_scale: 1.0
cmd_scale: [1.0, 1.0, 1.0]
cmd_init: [0, 0, 0]
cmd_range:
  lin_vel_x: [-0.4, 0.7]
  lin_vel_y: [-0.4, 0.4]
  ang_vel_z: [-1.57, 1.57]
```

### 6.4 写策略类

直接复制 `policy/loco_mode/LocoMode.py` 改名，关键点只有四处：

1. 构造函数里把 `FSMStateName.LOCOMODE` 改成 `FSMStateName.SKILL_MY_LOCO`，`name_str` 改成自己的名字；yaml 路径改成 `MyLoco.yaml`。
2. `run()` 里的 obs 拼接顺序必须和训练时**完全一致**。`LocoMode` 的是：`[ang_vel(3), gravity(3), cmd(3), qj(29), dqj(29), prev_action(29)] = 96`。你的训练代码怎么拼，这里就怎么拼。
3. 如果是 ONNX 模型，把 `torch.jit.load` 换成 `onnxruntime.InferenceSession`（见 `policy/dance/Dance.py` 第 58-64 行）。
4. `checkChange()` 里定义从本策略能切到哪些状态。典型做法：按 `FSMCommand.PASSIVE` 切回 Passive，按 `FSMCommand.LOCO` 切回 LocoMode。

### 6.5 在 FSM 里注册

`FSM/FSM.py`：

```python
from policy.my_loco.MyLoco import MyLoco

class FSM:
    def __init__(self, ...):
        ...
        self.my_loco_policy = MyLoco(state_cmd, policy_output)   # 实例化

    def get_next_policy(self, policy_name):
        ...
        elif policy_name == FSMStateName.SKILL_MY_LOCO:           # 路由
            self.cur_policy = self.my_loco_policy
```

### 6.6 允许从某个状态切进来

最常用的做法：在 `LocoMode.checkChange()` 里加一个分支，让走路中按某个指令切到你的策略：

```python
elif self.state_cmd.skill_cmd == FSMCommand.SKILL_MY_LOCO:
    return FSMStateName.SKILL_MY_LOCO
```

### 6.7 绑按键

`deploy_mujoco/deploy_mujoco.py` 主循环中新增一行：

```python
if joystick.is_button_released(JoystickButton.X) and \
   joystick.is_button_pressed(JoystickButton.L1):
    state_cmd.skill_cmd = FSMCommand.SKILL_MY_LOCO
```

键盘入口 `deploy_mujoco_keyboard_input.py` 类似，在命令解析块里加一条分支即可。

### 6.8 跑

```bash
python deploy_mujoco/deploy_mujoco.py
# 手柄：先按 A+R1 进 LocoMode，再按 X+L1 切到 MyLoco
```

---

## 7. 观测/动作空间约定

### 观测拼接（LocoMode 范式，num_obs=96）

```
obs[0:3]    = ang_vel * ang_vel_scale                       # base 角速度
obs[3:6]    = gravity_orientation                           # 投影到 base 的重力
obs[6:9]    = cmd * cmd_scale                               # 归一化后的速度指令
obs[9:38]   = (qj - default_angles) * dof_pos_scale         # 29 维关节位置偏差
obs[38:67]  = dqj * dof_vel_scale                           # 29 维关节速度
obs[67:96]  = last_action                                   # 上一步的 raw action
```

> `cmd` 来自手柄的 `vel_cmd`（-1..1），经 `scale_values` 映射到 `cmd_range` 里的物理单位。

### 动作到力矩

`deploy_mujoco.py:80` 有一个统一的 PD 层：

```python
tau = (target_q - q) * kp + (0 - dq) * kd
d.ctrl[:] = tau
```

所以**策略输出的是目标关节位置**，不是力矩。`action_scale` 乘完、加上 `default_angles`，得到 `target_q`；`kp/kd` 从 yaml 读。

### 历史窗口策略（Dance 范式）

如果你的策略吃 obs 历史（例如 mimic 类），参考 `policy/dance/Dance.py`：在构造函数里开 `ang_vel_buf`、`dof_pos_buf` 等 `(history_length * dim,)` 的滚动缓存，每步 `np.concatenate((new, buf[:-dim]))` 前插；`enter()` 里记得清零。

### 相位驱动策略（mimic 类）

Dance / KungFu / BeyondMimic / GAE_Mimic 都有一个 `ref_motion_phase`，每步按 `counter_step * control_dt / motion_length` 递增。`enter()` 里复位 `counter_step=0`，`exit()` 里也要清。

---

## 8. 调试清单

策略一接上就行为不对？按这个顺序排查：

1. **关节顺序** — 打印 `joint2motor_idx`，对照训练代码里的关节名列表，一一确认。
2. **control_dt 对不对** — `simulation_dt * control_decimation` 应等于训练时的 `dt`。
3. **obs 维度** — `self.obs` 的 shape 和训练时的 `num_obs` 一致吗？拼接顺序呢？
4. **default_angles** — 是不是训练时的参考站姿？如果是 MJCF 里 `<key>` 的 qpos，也要和训练对齐。
5. **action_scale** — 和训练代码里 `action_scale` / `action_clip` 完全一致吗？
6. **kps/kds** — 训练时用的是什么 PD 增益？写到 yaml 里。
7. **base 坐标系** — `ang_vel` 和 `gravity_orientation` 都是 base frame 下的量，不是世界系。
8. **单位** — Isaac Lab 用弧度，MuJoCo 里也是弧度，但别忘了检查 `dof_pos_scale / dof_vel_scale`。
9. **FixedPose 过渡** — 切策略前先过一次 FixedPose 能避免初值发散：`checkChange` 返回 `FSMStateName.FIXEDPOSE`，再从 FixedPose 切到目标策略。
10. **Passive 作保险丝** — 调试时永远要保证 `L3` 键能立即切回 Passive。

---

## 9. 策略模型文件存放约定

模型文件**不入 git**，放到各策略目录的 `model/` 下：

```
policy/loco_mode/model/policy_29dof.pt
policy/dance/model/dance.onnx
policy/gae_mimic/model/policy.onnx
policy/gae_mimic/motion/lafan1/walk1_subject2.npz   ← mimic 类还要参考轨迹
```

yaml 里的 `policy_path` / `onnx_path` 是**相对于策略自己目录下 `model/` 的相对路径**（`LocoMode.py:23` 里拼的是 `os.path.join(current_dir, "model", config["policy_path"])`）。

---

## 10. 从 sim2sim 到真机

sim2sim 跑稳之后再上真机，流程上只换入口：

```bash
python deploy_real/deploy_real.py
```

策略代码、config、model 文件全部不变。`deploy_real/deploy_real.py` 内部会起 DDS 订阅，把 G1 的实际关节状态填进同样的 `StateAndCmd`，同样通过 `FSM` 驱动，把 `PolicyOutput` 下发到 `unitree_sdk2_python` 的 `LowCmd` 话题。

**上真机前必做：**

- 仿真里至少跑 5 分钟不发散
- `L3`（Passive）必须可以随时打断
- 装好安全吊架、腰部固定架按官方说明解锁
- 复杂动作先拆手腕，避免自碰
- `kps/kds` 在仿真稳定后可以适度缩小（典型真机比仿真小 10-30%）

---

## 11. 参考文件一览

| 你要改的东西           | 看这个文件                                  |
|------------------------|---------------------------------------------|
| 仿真步长 / 控制频率    | `deploy_mujoco/config/mujoco.yaml`          |
| 主循环（手柄）         | `deploy_mujoco/deploy_mujoco.py`            |
| 主循环（键盘）         | `deploy_mujoco/deploy_mujoco_keyboard_input.py` |
| 状态机主类             | `FSM/FSM.py`                                |
| 状态基类 / 接口        | `FSM/FSMState.py`                           |
| 枚举 `FSMStateName/Cmd`| `common/utils.py`                           |
| 观测 / 动作总线        | `common/ctrlcomp.py`                        |
| 参考策略（TorchScript）| `policy/loco_mode/LocoMode.py`              |
| 参考策略（ONNX + 历史）| `policy/dance/Dance.py`                     |
| 参考策略（ONNX + mimic）| `policy/gae_mimic/gae_mimic.py`            |
| MJCF 模型（关节顺序）  | `g1_description/g1_29dof_rev_1_0.xml`       |

