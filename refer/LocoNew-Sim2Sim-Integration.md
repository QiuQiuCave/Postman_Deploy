# LocoNew Sim2Sim 集成记录

## 背景

`upper_lower` 项目新训练了一版 G1 locomotion 策略(29-DoF),训练配置为
`upper_lower/source/amp_tasks/amp_tasks/velocity/robots/g1/velocity_env_cfg.py`,
导出的 TorchScript 是
`upper_lower/logs/rsl_rl/g1_loco/2026-04-19_16-24-28_cliped_with_lin/exported/policy.pt`。

目标:在 `FSMDeploy_G1` 里新增一个独立的 `loco_new` FSM 状态,把这个新策略挂进去,
用 MuJoCo 做 sim2sim 验证,不影响原有的 `LocoMode`,方便 A/B 对比。

与旧 `LocoMode` 的关键差异:

| | 旧 `LocoMode` | 新 `loco_new` |
|---|---|---|
| obs 维度 | 96 | **99**(最前面多了 3 维 `base_lin_vel`) |
| `ang_vel_scale` | 1.0 | **0.2** |
| `dof_vel_scale` | 1.0 | **0.05** |
| `last_action` 剪裁 | 无 | **±12** |
| `action_scale` | 0.25 | 0.25 |
| 腿部 Kp/Kd | 200 / 5(硬件调过) | 40.18 / 2.56(训练原值) |
| 默认角度 | 髋 −0.2,膝 0.42 | 髋 −0.312,膝 0.669,肩前 0.2,肘 0.6 |
| 命令范围 | lin_x [−0.4, 0.7],lin_y ±0.4,ω_z ±1.57 | lin_x [−0.5, 1.0],lin_y ±0.3,ω_z ±0.2 |

---

## 新增文件

### 1. `policy/loco_new/LocoNew.py`
克隆 `LocoMode.py` 的结构,`FSMState` 子类。主要区别:

- **`__init__`**:
  - 多读几个 yaml 字段:`base_lin_vel_scale`、`gravity_scale`、`obs_clip_default`、
    `last_action_clip`
  - 加载后做两条自检:
    - `self.policy(torch.zeros(1, 99)).shape[-1] == 29` — 模型输入/输出维度对得上
    - `joint2motor_idx[0, 11, 2] == (0, 15, 12)` — Isaac 29-DoF 顺序与电机顺序的对齐点检
- **`enter`**:跟 `LocoMode` 一样,按 `joint2motor_idx` 把 `kps/kds/default_angles`
  从 policy 顺序一次性 reorder 成 motor 顺序。
- **`run`**:obs 按训练里的 7 项顺序逐项拼 99 维:
  ```
  [0:3]     base_lin_vel · 1.0           clip ±100
  [3:6]     base_ang_vel · 0.2           clip ±100
  [6:9]     projected_gravity            不 clip
  [9:12]    velocity_command             不 clip(cmd_scale=[1,1,1])
  [12:41]   joint_pos - default          clip ±100
  [41:70]   joint_vel · 0.05             clip ±100
  [70:99]   last_action                  clip ±12
  ```
  注意 `last_action` 存的是**原始策略输出**(pre-scale、pre-offset),跟训练端
  `mdp.last_action` 的语义一致,也跟 `LocoMode` 的 `self.action` 用法一致。
- **`checkChange`**:
  - `FSMCommand.PASSIVE` → `PASSIVE`
  - `FSMCommand.LOCO` → `LOCOMODE`(直接切回旧策略做 A/B)
  - 其余 → 保持 `LOCO_NEW`

### 2. `policy/loco_new/config/LocoNew.yaml`
所有向量都是 **policy 顺序**(Isaac 29-DoF),`enter()` 再 reorder 到 motor 顺序。

- `kps/kds`:**直接用训练 env.yaml 里的 actuator 组值**(hip_pitch 40.18、knee 99.10、
  shoulder 14.25 等)。**不要**复用 `LocoMode.yaml` 的 200/5,那组是针对旧策略动作风格
  调的硬件 PD,跟新策略的 `action_scale · 0.25` 输出幅度对不上,直接用会炸。
- `default_angles`:训练 init pose(hip −0.312、knee 0.669、shoulder_pitch 0.2、
  shoulder_roll ±0.2、ankle_pitch −0.363、elbow 0.6,其余 0)。
- `joint2motor_idx`:**从 `LocoMode.yaml` 原样复制**。Isaac 29-DoF 顺序一致,
  spot-check 过 `idx 0=left_hip_pitch→motor 0`、`idx 2=waist_yaw→motor 12`、
  `idx 11=left_shoulder_pitch→motor 15`。
- `tau_limit`:从 `LocoMode.yaml` 复制,当前运行时不用,保留作对齐参考。
- `cmd_range`:训练端 `UniformLevelVelocityCommandCfg.limit_ranges`,
  lin_x [−0.5, 1.0]、lin_y ±0.3、ω_z ±0.2。
- `obs_clip_default: 100.0`、`last_action_clip: 12.0` — 跟训练 obs 剪裁一致。
- `num_obs: 99`、`num_actions: 29`。

### 3. `policy/loco_new/model/policy_loco_new.pt`
原来放在 `policy/loco_mode/model/` 下,已经移过来。遵循"一个 policy 一个 model
目录"的约定,避免跨策略耦合。

---

## 修改的既有文件

### `common/ctrlcomp.py`:`StateAndCmd` 加 `base_lin_vel`
```python
self.base_lin_vel = np.zeros(3, dtype=np.float32)
```
纯追加,老策略不读这个字段,不影响它们的行为。新策略通过这个共享总线读。

> 注:`base_quat` 早就已经在 `deploy_mujoco` / `deploy_real` 里用 duck-typing 赋值,
> 虽然之前没在 `ctrlcomp.py` 声明。这次为了清晰,`base_lin_vel` 显式声明。

### `common/utils.py`:加枚举
```python
class FSMStateName(Enum):
    ...
    LOCO_NEW = 13

class FSMCommand(Enum):
    ...
    LOCO_NEW = 13
```

### `FSM/FSM.py`:注册新策略
- `import LocoNew`
- `__init__` 里实例化 `self.loco_new_policy = LocoNew(...)`
- `get_next_policy` 加一个分支,`LOCO_NEW` → `self.loco_new_policy`

### `policy/loco_mode/LocoMode.py`:`checkChange` 加一条
新增 `FSMCommand.LOCO_NEW → FSMStateName.LOCO_NEW`。意义:可以从旧 `LocoMode`
直接热切到新策略,不用经过 `FixedPose` 或 `Passive`,方便实时对比。

### `policy/fixedpose/FixedPose.py`:`checkChange` 加一条
同上,可以从 `start`(POS_RESET)进 `FixedPose` 之后直接 `x+l1` 进新策略。

### `deploy_mujoco/deploy_mujoco_keyboard_input.py`
两处改动:
1. 菜单打印加一行 `x+l1 - Locomotion NEW (sim2sim)`。
2. 命令分支加
   ```python
   elif cmd == "x+l1":
       state_cmd.skill_cmd = FSMCommand.LOCO_NEW
   ```
3. 状态采集块里,加**本体坐标系的线速度读取**:
   ```python
   cvel = np.zeros(6)
   mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, 1, cvel, 1)
   state_cmd.base_lin_vel = cvel[3:6].astype(np.float32)
   ```
   用 MuJoCo 自带的 `mj_objectVelocity`(`flg_local=1`)拿到 body 1(base)的 6-dof
   局部坐标速度,`cvel[3:6]` 是线速度部分。直接这么做的好处是不引入 scipy,
   也不用手写四元数共轭旋转。

### `deploy_mujoco/deploy_mujoco.py`
同上,joystick 路径加:
```python
if joystick.is_button_released(X) and joystick.is_button_pressed(L1):
    state_cmd.skill_cmd = FSMCommand.LOCO_NEW
```
以及同样的 `mj_objectVelocity` 调用。

### `deploy_real/deploy_real.py`
只加占位:
```python
# TODO: body-frame velocity estimator; LOCO_NEW is sim-only on hardware.
self.state_cmd.base_lin_vel = np.zeros(3, dtype=np.float32)
```
真机的 IMU 拿不到可信的 body 系线速度,所以不在真机上绑定 `LOCO_NEW`,
策略明确标为 sim-only。

### `CLAUDE.md`(根)和 `FSMDeploy_G1/CLAUDE.md`
在"安全"/"稳定策略"列表里追加:`LocoNew` 仅 sim2sim,真机不稳定(base_lin_vel 归零)。

---

## 关键约定(以后改东西容易踩的坑)

1. **Kp/Kd 必须用训练端的值**,不要拿 `LocoMode.yaml` 的硬件 PD 复制。新策略输出
   是 `action · 0.25 + default`,如果 Kp 过高会把训练外的分布直接放大,很快炸。
2. **obs 顺序不能错**。按训练 `ObservationsCfg` 的声明顺序拼 `[base_lin_vel,
   base_ang_vel, projected_gravity, velocity_commands, joint_pos_rel,
   joint_vel, last_action]`。每一项的 scale 和 clip 也要严格一致,特别是
   `ang_vel_scale=0.2`、`dof_vel_scale=0.05`、`last_action_clip=12`——这三个很容易
   按 LocoMode 的 1.0/1.0/无剪裁 填成默认值,填错表现是"头晃"/"手臂慢慢飘起来"。
3. **`last_action` 存原始策略输出**,不是 scale+offset 之后的 q_cmd。这是训练端
   `mdp.last_action` 的定义,LocoMode 和 LocoNew 都是这样做的。
4. **`joint2motor_idx` 用 policy→motor 的约定**:obs 在输入时把电机顺序的
   `q/dq` 按 `joint2motor_idx[i]` 取到 policy 顺序;action 在输出时把 policy
   顺序的 `q_cmd` 按 `joint2motor_idx[i]` 写回到电机顺序。改训练端的 joint 名列表
   必须同步改这个表。
5. **`base_lin_vel` 是 body 系**,不是世界系。MuJoCo 里用
   `mj_objectVelocity(..., flg_local=1)` 或者自己手写 `R.T · v_world`。取错表现是
   站着不动 `base_lin_vel` 非零、行走时机身持续下压。
6. **切换回去**:在 LocoNew 里发 `FSMCommand.LOCO`(也就是 `a+r1`)就切回旧
   `LocoMode`,不经过 Passive;方便 A/B。

---

## 触发方式

| 输入 | 命令 | 目标状态 |
|---|---|---|
| 键盘 | `x+l1` | `LOCO_NEW` |
| 键盘 | `a+r1` | `LOCOMODE`(旧) |
| 手柄 | `X + L1`(按住 L1 再按 X) | `LOCO_NEW` |
| 键盘/手柄 | `l3` | `PASSIVE`(随时 E-stop) |

从 `FixedPose` 和 `LocoMode` 都可以直接进 `LOCO_NEW`;从 `LOCO_NEW` 可以直接
进 `PASSIVE` 或回 `LOCOMODE`。

---

## 验证流程(键盘模式)

```bash
python deploy_mujoco/deploy_mujoco_keyboard_input.py
```

依次输入:
1. `start` — 进 `FixedPose`,机器人过渡到旧默认位。
2. `a+r1` — 进旧 `LocoMode`,稳住。
3. `x+l1` — 切到 `LocoNew`。期望观察:
   - 姿态可见变化(膝 0.669 比 0.42 明显更蹲,髋前俯,肩微前伸 0.2)。
   - 腿明显更软(Kp 从 200 降到 40 左右)。
4. `vel 0.3 0 0` — 慢走前进,在 `lin_vel_x` 训练上限内。
5. `vel 0.5 0 0` — 训练速度上限。
6. `vel 0 0 0.2` — 原地 yaw。
7. `vel 0.3 0.2 0.1` — 组合。
8. `a+r1` — 热切回旧 LocoMode,验证 `LocoNew.checkChange` 的 `LOCO→LOCOMODE` 分支。
9. `x+l1` — 再切回新策略,验证幂等。
10. `l3` — Passive 收尾。

失败模式诊断:

| 现象 | 很可能原因 |
|---|---|
| 一进 LocoNew 立刻崩腿 | `default_angles` reorder 错了,或 Kp 填成旧值 |
| 头/腰抖 | `ang_vel_scale` 没改成 0.2 |
| 手臂几秒内慢慢飘起来 | `last_action_clip=12` 没生效 |
| 机身持续前倾下沉 | `base_lin_vel` 拿到了世界系 / 没调用 `mj_objectVelocity(..., 1)` |
| 一站就飞 NaN | 对照表:`joint2motor_idx` 和训练端 joint 列表不一致 |

---

## 文件清单

新增:
- `FSMDeploy_G1/policy/loco_new/LocoNew.py`
- `FSMDeploy_G1/policy/loco_new/config/LocoNew.yaml`
- `FSMDeploy_G1/policy/loco_new/model/policy_loco_new.pt`(从 loco_mode/model 移过来)

修改:
- `FSMDeploy_G1/common/ctrlcomp.py` — 加 `base_lin_vel` 字段
- `FSMDeploy_G1/common/utils.py` — 加 `LOCO_NEW` 枚举
- `FSMDeploy_G1/FSM/FSM.py` — 注册实例 + 路由分支
- `FSMDeploy_G1/policy/loco_mode/LocoMode.py` — `checkChange` 加 `LOCO_NEW` 出口
- `FSMDeploy_G1/policy/fixedpose/FixedPose.py` — 同上
- `FSMDeploy_G1/deploy_mujoco/deploy_mujoco_keyboard_input.py` — `base_lin_vel`
  采集 + `x+l1` 键盘分支 + 菜单
- `FSMDeploy_G1/deploy_mujoco/deploy_mujoco.py` — `base_lin_vel` 采集 + `X+L1`
  手柄分支
- `FSMDeploy_G1/deploy_real/deploy_real.py` — `base_lin_vel` 占位(真机 sim-only)
- `CLAUDE.md`、`FSMDeploy_G1/CLAUDE.md` — 安全说明加 LocoNew 仅 sim2sim
