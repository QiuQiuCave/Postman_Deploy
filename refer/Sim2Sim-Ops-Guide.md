# Sim2Sim 运维指南

这份文档汇总部署侧两类常见运维操作:

1. **重训/新训了 DualAgentTracking 策略,如何搬到部署这边**(§1、§2)
2. **调 MuJoCo 里的运输箱**(几何、位置、质量、在手心的生成偏移、悬吊时长、停放位)(§3)

设计说明和代码路径另见 `refer/DualAgentTracking-Sim2Sim-Integration.md`
(tracking)和 `refer/DualAgentVelocity-Sim2Sim-Integration.md`(box-trans-vel)。

---

## 1. 重训同一条 demo(例如 walk),替换现有 ONNX + motion

只需要覆盖 `model/dual_agent_combined.onnx` 和(可选)
`motion/walk_tracking_ref.npz`。yaml 不动,state 类不动,enum 不动,键位不动。

```bash
# 1) 切到训练仓库,激活 isaaclab 环境
cd /home/qiuziyu/code/postman/upper_lower
conda activate unitree_isaaclab
# 或直接用绝对路径: /home/qiuziyu/miniforge3/envs/unitree_isaaclab/bin/python

# 2) 导出 ONNX(--task tracking 走 upper_obs_dim = 96 / lower_obs_dim = 109,
#    2026-04-21 retrain 之后的 slim 版本;旧 480/121 ONNX 已不兼容)
#    --output_dir 直接指到部署仓库的 model/ 目录,省一次 cp
TS=2026-04-21_20-01-44_joint_train      # 你这次训练的目录名
CKPT=15000                              # 你选的 checkpoint step
python scripts/factoryIsaac/dual_agent_export_onnx.py \
  --upper_policy logs/rsl_rl/g1_dual_agent/${TS}/upper_model_${CKPT}.pt \
  --lower_policy logs/rsl_rl/g1_dual_agent/${TS}/lower_model_${CKPT}.pt \
  --task tracking \
  --output_dir /home/qiuziyu/code/postman/FSMDeploy_G1/policy/dual_agent_tracking/model

# 导出脚本会在 output_dir 里写 dual_agent_combined.onnx,
# 和 yaml 里 policy_path 一致,无需重命名。

# 3) (可选) 换了 motion 源文件时,重跑 preprocess
python scripts/factoryIsaac/dual_agent_tracking_preprocess_motion.py \
  --motion_file data/demo/lafan/walk4_subject1.npz \
  --output /home/qiuziyu/code/postman/FSMDeploy_G1/policy/dual_agent_tracking/motion/walk_tracking_ref.npz

# 4) 切回部署仓库,跑 MuJoCo
cd /home/qiuziyu/code/postman/FSMDeploy_G1
python deploy_mujoco/deploy_mujoco_keyboard_input.py
# 命令序列: start → a+r1 → a+l1(→ 箱子 spawn → 跟踪 walk)
```

**注意**:`--output_dir` 会创建(若不存在)一个叫 `exported_tracking/` 的
子目录放额外产物;核心的 `dual_agent_combined.onnx` 直接写在
`--output_dir` 根目录下。需要哪个就拷哪个。

---

## 2. 训了一个**新的 tracking 任务**(不同 motion,不同动作)

每个新 tracking 任务 = 一份独立 ONNX + 一份独立 motion npz + 一个独立的
state 类 + 一个键位(目前 `b+l1` / `x+l1` / `y+l1` 预留)。最省心的做法是
直接克隆 `DualAgentTracking` 再改几处符号。

### Step 1: 训练侧导出 artifact

以一版假设的 jumping demo 为例,源 motion 文件
`data/demo/lafan/jump3_subject1.npz`:

```bash
cd /home/qiuziyu/code/postman/upper_lower
conda activate unitree_isaaclab

# 2-1 先 preprocess motion 到部署仓库的 motion/ 目录(新文件名)
python scripts/factoryIsaac/dual_agent_tracking_preprocess_motion.py \
  --motion_file data/demo/lafan/jump3_subject1.npz \
  --output /home/qiuziyu/code/postman/FSMDeploy_G1/policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz

# 2-2 再导出 ONNX(同名 dual_agent_combined.onnx,放到新 policy 的 model/ 目录)
TS=<new-training-ts>
CKPT=<step>
python scripts/factoryIsaac/dual_agent_export_onnx.py \
  --upper_policy logs/rsl_rl/g1_dual_agent/${TS}/upper_model_${CKPT}.pt \
  --lower_policy logs/rsl_rl/g1_dual_agent/${TS}/lower_model_${CKPT}.pt \
  --task tracking \
  --output_dir /home/qiuziyu/code/postman/FSMDeploy_G1/policy/dual_agent_jump_tracking/model
```

关键约束(训练侧改动时同步检查):

- `upper_obs` 维度必须仍是 96(单帧),`lower_obs` 维度必须仍是 109。新增
  obs term → 同时改 deploy 侧 `run()` 和 yaml 的 `num_obs_upper /
  num_obs_lower`,导出脚本本身会 assert 维度。如果训练把 anchor / lin_vel
  重新放回 actor,部署侧要恢复 deploy_loop 的 `anchor_pos_w/quat_w` 注入
  路径(代码还在,只是 actor 当前不读),并把这条 obs 加进 `lower_obs_flat`。
- 关节顺序不能动。动了就要改 yaml 的 `joint2motor_idx` 和 `default_angles`,
  还要 re-derive preprocess 脚本里的 `TORSO_BODY_IDX_IN_NPZ`。
- motion fps 必须等于 policy fps(50 Hz,即 `control_dt: 0.02`)。训练若
  换频率,改 yaml `control_dt` 并在 `MotionBuffer` 里加插值。

### Step 2: 部署侧挂一个新 state 类

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1

# 2-1 克隆 policy 目录
cp -r policy/dual_agent_tracking policy/dual_agent_jump_tracking
mv policy/dual_agent_jump_tracking/DualAgentTracking.py \
   policy/dual_agent_jump_tracking/DualAgentJumpTracking.py
mv policy/dual_agent_jump_tracking/config/DualAgentTracking.yaml \
   policy/dual_agent_jump_tracking/config/DualAgentJumpTracking.yaml

# 2-2 确认 artifact 已就位(Step 1 的输出)
ls policy/dual_agent_jump_tracking/model/dual_agent_combined.onnx
ls policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz
```

然后手动改这 4 个文件,`git diff` 时自查:

- **`policy/dual_agent_jump_tracking/DualAgentJumpTracking.py`**
  - `class DualAgentJumpTracking(FSMState)`
  - `self.name = FSMStateName.DUAL_AGENT_JUMP_TRACK`(下面新加的 enum)
  - `self.name_str = "dual_agent_jump_tracking_mode"`
  - 构造函数里 yaml 名改成 `DualAgentJumpTracking.yaml`
  - `checkChange` 的 `else` 分支返回新 FSMStateName

- **`policy/dual_agent_jump_tracking/config/DualAgentJumpTracking.yaml`**
  - `motion_file: "jump_tracking_ref.npz"`

- **`common/utils.py`**
  - `FSMStateName.DUAL_AGENT_JUMP_TRACK = 18`
  - `FSMCommand.DUAL_AGENT_JUMP_TRACK = 18`

- **`FSM/FSM.py`**
  - `from policy.dual_agent_jump_tracking.DualAgentJumpTracking import DualAgentJumpTracking`
  - `self.dual_agent_jump_tracking_policy = DualAgentJumpTracking(...)`
  - `get_next_policy` 加分支

- **`policy/loco_mode/LocoMode.py`** 和(按需)其它 tracking 同族 state 的
  `checkChange`:加 `FSMCommand.DUAL_AGENT_JUMP_TRACK → FSMStateName.DUAL_AGENT_JUMP_TRACK`。
  一般只有从 `LocoMode` 切进来是真实用得到的路径。

- **`deploy_mujoco/deploy_mujoco_keyboard_input.py`** 和
  **`deploy_mujoco/deploy_mujoco.py`**:在 L1 组里占一个未用键(`b+l1`、
  `x+l1`、`y+l1` 任选):
  ```python
  # keyboard
  elif cmd == "b+l1":
      state_cmd.skill_cmd = FSMCommand.DUAL_AGENT_JUMP_TRACK

  # joystick(和 a+l1 DualAgentTracking 对称,换键位)
  if joystick.is_button_released(JoystickButton.B) and \
     joystick.is_button_pressed(JoystickButton.L1):
      state_cmd.skill_cmd = FSMCommand.DUAL_AGENT_JUMP_TRACK
  ```
  并同步更新键盘入口顶部的帮助列表(`print("  -- L1 group ...")`)。

- **`CLAUDE.md` 的 Safety 段**:把新键位加进 L1 列表,标注 sim2sim-only。

### Step 3: smoke test

```bash
# FSM 构造会加载每个策略的 ONNX + motion npz,一次性验证所有路径。
python -c "
from common.ctrlcomp import StateAndCmd, PolicyOutput
from FSM.FSM import FSM
fsm = FSM(StateAndCmd(29), PolicyOutput(29))
print('jump_tracking.needs_transport_box =',
      fsm.dual_agent_jump_tracking_policy.needs_transport_box)
"

# 然后在 MuJoCo 里真跑一下。
python deploy_mujoco/deploy_mujoco_keyboard_input.py
# start → a+r1 → <你的新键位>
```

---

## 3. MuJoCo 运输箱的配置在哪里改

`transport_box` 是场景里唯一的、由所有 `needs_transport_box = True` 策略
共用的刚体。它的行为被**两处**分开控制:

| 想改什么 | 改哪个文件 | 改哪段 |
|---|---|---|
| 箱子几何/质量/颜色 | `g1_description/scene.xml` | `<body name="transport_box">` 里的 `<geom ...>` |
| 场景中停放位(未激活时) | `g1_description/scene.xml` **和** `deploy_mujoco/*.py` | body 的 `pos="..."` **和** `box_park_pos` |
| 抱到手心时的生成偏移(高/前后/左右) | `deploy_mujoco/deploy_mujoco.py` **和** `deploy_mujoco/deploy_mujoco_keyboard_input.py` | `box_offset_base` |
| 生成后硬 pin"绳吊"时长 | 同上两个 deploy 文件 | `box_hold_dur` |

### 3.1 箱子几何、质量、颜色

`FSMDeploy_G1/g1_description/scene.xml`:

```xml
<body name="transport_box" pos="100 100 0.15">
  <freejoint/>
  <geom type="box" size="0.15 0.15 0.15" mass="1.5" rgba="0.82 0.45 0.22 1"/>
</body>
```

- **`size="0.15 0.15 0.15"`** 是 MuJoCo box 的**半长**,对应 0.3 m 立方体
  (和训练 cfg `BOX_SIZE=0.3` 对齐)。训练里换大小,这里也要改。
- **`mass="1.5"`**(kg)。训练侧 `mass=1.5`。
- **`rgba`** 纯视觉,随意。
- **顶层 `pos="100 100 0.15"`** 是场景里的**初始停放位置**。这个值在
  `d.qpos` 第一次初始化时被写入。运行时的重置停放由下一节的
  `box_park_pos` 覆盖,所以这两个值应该保持一致。

### 3.2 停放位(未激活时)

`scene.xml` 里的 `pos="100 100 0.15"` 和两个 deploy 入口里的
`box_park_pos` **必须保持一致**。后者在策略退出抱箱状态时被写回
`d.qpos`:

```python
# deploy_mujoco/deploy_mujoco.py:55
# deploy_mujoco/deploy_mujoco_keyboard_input.py:109
box_park_pos    = np.array([100.0, 100.0, 0.15], dtype=np.float64)
```

想换停放位(比如让箱子落在场景另一角而不是 `(100,100)`),**两个地方都要
改**,否则首次退出会把箱子"传送"到不一致的位置。

### 3.3 抱到手心时的生成偏移

这是用户最常调的参数——决定箱子落在机器人手心什么位置。

```python
# deploy_mujoco/deploy_mujoco.py:56
# deploy_mujoco/deploy_mujoco_keyboard_input.py:110
box_offset_base = np.array([0.32, 0.0, 0.14], dtype=np.float64)
#                          [ x,    y,    z  ]
#                           向前   左右  往上
```

坐标系是**机器人 pelvis frame**,`+x` 向前、`+y` 向左、`+z` 向上。
spawn 时 deploy loop 用 pelvis 当前 quat 把这个 body-frame 偏移旋到世界,
再叠加 pelvis world 位置,所以不管机器人朝哪,箱子都会在手心。

训练派生值是 `0.26`(IsaacLab 世界坐标 `(0.32, 0, 1.05)` 减 pelvis 休止
高度 ~0.79)。**实测下调 12 cm 到 `0.14`**,让箱子坐进抓握区深一点,手合拢
时冲击更小。如果换了抱箱姿态或换了箱子尺寸,可能需要重新微调。

调的时候注意:

- 调高低 → 改 `box_offset_base[2]`。每次步进 ±1~2 cm,目视箱子和手心的
  相对位置。太高箱子掉进手里会有冲击,太低会穿模。
- 调前后 → 改 `box_offset_base[0]`。默认 0.32 对应训练时箱子中心到 pelvis
  中心的水平距离。
- 调左右 → 改 `box_offset_base[1]`。训练是对称抱箱,默认 0.0,一般不用动。
- **两个 deploy 入口都要改**,值要保持一致,否则手柄/键盘两种启动方式行为
  会不一致。

### 3.4 "绳吊"硬 pin 时长

生成箱子后的 1 秒内,deploy loop 会在每个 `mj_step` 前把箱子的 qpos/qvel
硬覆盖回生成位,等价于一根理想刚性绳吊着,给双臂合拢的时间,避免 contact
冲击让箱子飞出去。

```python
# deploy_mujoco/deploy_mujoco.py:57
# deploy_mujoco/deploy_mujoco_keyboard_input.py:111
box_hold_dur    = 1.0
```

- 如果策略 ramp-in 更慢(比如自训策略不稳,给了 `ramp_time: 2.0` 而非
  `0.5`),悬吊时间要跟着调大,不然绳松开时手还没合拢好,箱子会掉。
- 调短(<0.5)一般只在调试抓取动作、希望快速看到"松绳"效果时用,正常
  部署别动。

### 3.5 哪里**不用**改

- **哪个策略该生成箱子**:不用改 deploy loop,在策略类里改
  `needs_transport_box = True / False`。deploy loop 通过这个类属性决定,对
  新增策略零侵入。
- **生成偏移要随箱子几何变**?几何改了(比如换成 0.4 m 立方体),`size`
  变 0.2,`box_offset_base` 里前后距离也得相应加大(避免穿胸)。**这里没有
  自动联动**,两处都要手动同步。
