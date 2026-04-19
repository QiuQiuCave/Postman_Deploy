# BoxTransportVelocity Sim2Sim 集成记录

## 背景

`upper_lower` 项目训练了一版 G1 单网络抱箱搬运策略,29-DoF 全身体控制,
ONNX 导出在
`upper_lower/logs/rsl_rl/g1_box_transport_velocity/2026-02-01_16-03-30_base/exported/policy.onnx`,
环境配置是
`upper_lower/source/amp_tasks/amp_tasks/box_transport/robots/g1/box_transport_velocity_env_cfg.py`。

目标:在 `FSMDeploy_G1` 里新增独立 FSM 状态 `box_transport_velocity`,
ONNX 推理,绑定 `b+r1` 触发,做 sim2sim 验证。先打通无箱子的纯站立/行走链路。

需求 spec 见 `refer/BoxTransport-Sim2Sim-Prompt.md`。

与已有 `LocoMode` / `LocoNew` 的关键差异:

| | `LocoNew` (99-dim) | `BoxTransportVelocity` (480-dim) |
|---|---|---|
| 后端 | TorchScript / ONNX 双支持 | 仅 ONNX |
| obs 维度 | 99 单帧 | **480 = 96 × 5 历史** |
| 历史结构 | 无 | **per-term ring buffer**, oldest→newest |
| `last_action` clip | ±12 | ±12 |
| `joint_vel_rel` scale | 0.05 | 0.05 |
| `cmd_scale` | [1,1,1] | [1,1,1] |
| `action_scale` | 全身一致 0.25 | **每关节不同**(0.07~0.55) |
| 默认角(腿) | 髋 −0.312,膝 0.669 | 同 |
| 默认角(臂) | 中性张开 | **抱箱姿态**(肩 ±1,腕 ±1.2,肘 −0.18) |
| 命令范围 | lin_x [−0.5,1.0], ω_z ±0.2 | 同 |

---

## 新增文件

### 1. `policy/box_transport_velocity/BoxTransportVelocity.py`

`FSMState` 子类,onnxruntime 后端,关键点:

- **`__init__`**:加载 yaml,建 6 个 per-term ring buffer (`hist_*`,
  shape `[H=5, term_dim]`),warmup ONNX 5 次,断言输入/输出维度。
- **`enter`**:
  - 按 `joint2motor_idx` reorder kps/kds/default_angles 到 motor 顺序
  - 清零所有 history,置 `_history_primed = False`
  - 快照当前 `state_cmd.q`,启动 2s pose ramp-in(见下文)
- **`run`** 流程:
  1. ramp-in 阶段:线性插值到 default,**不喂 ONNX,不写 history**
  2. 读 robot state,motor→policy reorder
  3. 逐项 scale + clip(顺序匹配 IsaacLab `ObservationManager`)
  4. 第一帧用 `_prime` 把所有 5 个 slot 填同值,之后用 `_push` 滚动
  5. concat 6 个 history 块得到 480 维 obs(每块 `(H, dim).reshape(-1)`,
     即 `[oldest, ..., newest]` 行优先)
  6. ONNX inference,外层再 clip ±100(safety belt)
  7. action × per-joint scale + default → policy 序 → motor 序 → `policy_output`
- **`exit`**:`pass`
- **`checkChange`**:支持切到 PASSIVE / LOCO / LOCO_NEW / LOCO_NEW_ONNX,
  其余保持 `SKILL_BOX_TRANSPORT_V`

### 2. `policy/box_transport_velocity/config/BoxTransportVelocity.yaml`

所有向量都是 **policy 顺序**(Isaac 29-DoF):

- `kps / kds`:从训练 actuator 配置直接抄(腿髋 40.18 / 膝 99.10 / 踝 28.50,
  腰 14.25,肩 28.50,肘 14.25,腕 16.78)
- `default_angles`:`G1BoxTransportVelocityEnvCfg.__post_init__` 里的
  `init_state.joint_pos`(腿同 LocoNew,**手臂为抱箱姿态**:
  shoulder_pitch −0.6,shoulder_roll ±1.0,wrist_roll ±1.2 等)
- `action_scale`:每关节独立,16 个不同值。源自训练里的
  `JointPositionAction.scale=G1_ACTION_SCALE`,公式 `0.25 × effort_limit / stiffness`
- `joint2motor_idx`:跟 LocoMode/LocoNew 完全一致(同套 Isaac 29-DoF 顺序)
- 训练侧 `obs_clip_default=100`,`last_action_clip=12`,`history_length=5`,
  `num_obs=480`,`num_actions=29`
- 命令范围用 `UniformLevelVelocityCommandCfg.limit_ranges`(部署不再用早期的
  小 `ranges`,直接用最终上限)

### 3. `policy/box_transport_velocity/model/policy.onnx`

从训练日志拷过来,不入 git。

---

## 修改文件

### 4. `common/utils.py`
新增枚举:`FSMStateName.SKILL_BOX_TRANSPORT_V = 15`,
`FSMCommand.SKILL_BOX_TRANSPORT_V = 15`。

### 5. `FSM/FSM.py`
- `from policy.box_transport_velocity.BoxTransportVelocity import BoxTransportVelocity`
- `__init__` 里 `self.box_transport_velocity_policy = BoxTransportVelocity(...)`
- `get_next_policy` 加分支

### 6. `policy/loco_mode/LocoMode.py` / `policy/loco_new/LocoNew.py` / `policy/fixedpose/FixedPose.py`
各自 `checkChange` 加一条:
```python
elif self.state_cmd.skill_cmd == FSMCommand.SKILL_BOX_TRANSPORT_V:
    return FSMStateName.SKILL_BOX_TRANSPORT_V
```
让从 LocoMode、LocoNew、FixedPose 都能切进 box_transport。

### 7. `deploy_mujoco/deploy_mujoco.py` / `deploy_mujoco/deploy_mujoco_keyboard_input.py`
绑定 `B+R1`(手柄)和 `b+r1`(键盘)→ `SKILL_BOX_TRANSPORT_V`。
`deploy_real.py` **不绑**——硬件不安全,这套策略只做 sim2sim。

### 8. `CLAUDE.md`(本仓库 + 父目录)
扩了 Safety 段:`BoxTransportVelocity` 列入 sim2sim-only 名单,
绑 `b+r1`,标注 480-dim per-term history 结构。

---

## 关键的两处坑(以及怎么爬出来的)

### 坑 1:obs history 的 IsaacLab 语义

`ObservationGroupCfg.history_length=5, flatten_history_dim=true,
concatenate_terms=true` 在 IsaacLab 内部做的事:

1. 每个 term 独立维护一个 `CircularBuffer(shape=(H, dim))`
2. **第一次 append 时**,buffer 全部 H 个 slot 都被填成同一帧
   (避免开局有"假零")
3. 每个 term 的 buffer 被 reshape 成 `(H * dim,)`,**oldest 在前**
4. 然后所有 term 按 dataclass 字段顺序 concat 成最终 obs

部署侧必须复现以上四点。`_prime`(第一帧)和 `_push`(后续滚动)就是
为这个语义写的。最容易踩的是 reshape 顺序——`np.reshape(-1)` 默认行优先,
正好等价于 `[oldest, ..., newest]` 拼接,跟 IsaacLab 一致。

源码参考:
- `IsaacLab/source/isaaclab/isaaclab/utils/buffers/circular_buffer.py`
- `IsaacLab/source/isaaclab/isaaclab/managers/observation_manager.py:380-430`

### 坑 2(更致命):OOD 切入瞬间炸机

**症状**:从 LocoMode 切到 BoxTransportVelocity 的瞬间,关节迅速拧成一团掉地。

**根因**:`box_transport_velocity_env_cfg.py` 里
```python
reset_robot_joints = EventTerm(func=mdp.reset_joints_by_scale,
                               params={"position_range": (1.0, 1.0), ...})
```
是**乘法**复位:`pos = default × U(1.0, 1.0)`,即"严格等于默认角"。
对 0-default 的手臂关节(肩 yaw、肘),0×任何值还是 0,所以训练时这些关节
**永远从 0 开始**,从未见过"上来就 ±1 rad"的 OOD 状态。

部署时切入瞬间,手臂的 `joint_pos_rel = qj_obs - default = (loco 的 0) - (box 的 ±1)
= ∓1 rad`,是训练分布(±0.05)的 ~20×。策略输出垃圾 action,叠加 kp=99 的
膝盖 PD,瞬间炸机。

**部署侧补丁**:`BoxTransportVelocity.enter()` 里抓一帧 `state_cmd.q`,
然后 `run()` 头部跑 2 秒线性插值到 `default_angles`,期间不喂 ONNX、不写 history。
插完再启动推理。这相当于"先把姿态拉回训练分布内,再让策略接管"。

效果:不再瞬间炸,策略会**尝试稳定身体**,但仍然站不稳——因为 ramp 完之后
`joint_pos_rel ≈ 0`,但身体在落地过程中已经偏离 base_height 阈值,
策略本身的鲁棒性不足。

**这是训练侧问题,不是部署侧能彻底解决的。**

---

## 训练侧的同步修复(`upper_lower` 仓库)

为了根上解决 OOD 与鲁棒性,改了 `box_transport_velocity_env_cfg.py`:

| # | 项目 | 原值 | 新值 | 原因 |
|---|---|---|---|---|
| 1 | `PolicyCfg.history_length` | 5 | **0** | 部署侧 ring buffer 是 sim2sim 链路里最容易出 bug 的环节;单帧策略一般也更易泛化 |
| 2 | `reset_robot_joints` | `reset_joints_by_scale (1.0, 1.0)` | **`reset_joints_by_offset (-0.3, 0.3)`** | 加性 offset 才能让 0-default 关节也有随机化(根因修复) |
| 3 | `box_dropped` termination | minimum_height=0.3,存在 | **删除** | 让策略练"箱子掉了之后稳住自己" |
| 4 | `push_robot.velocity_range` | ±0.1 | **±0.3** | 起始更激进 |
| 4 | `push_vel_levels.max_vel` | 1.0 | **2.0** | 课程上限抬高 |
| 5 | `randomize_actuator_gains` | 不存在 | **新增 startup, ±20% scale** | MuJoCo PD 与 IsaacLab implicit actuator 不完全一致,标准 sim2sim 防御 |

注意:**critic obs 的 `history_length=5` 保持不动**——只是 deploy 用不到,
critic 多帧 value 估计仍然有用。

---

## 部署侧待办(等新 ONNX 出来再做)

新 ONNX 训练完后(history_length=0,obs 96 维):

1. `BoxTransportVelocity.py`:删掉 6 个 `hist_*`、`_push`、`_prime`、
   `_history_primed`,obs 改单帧 concat,结构跟 `LocoNew.run()` 几乎一样
2. yaml 改:`history_length: 1`(可保留字段做记录),`num_obs: 96`
3. ramp-in 逻辑保留——它对所有"切入时姿态偏远"场景都有用,不依赖 history
4. 重跑 sim2sim 验证流程

---

## 验证流程(单帧版上线后用)

```
python deploy_mujoco/deploy_mujoco_keyboard_input.py
```

```
start         # FixedPose 缓降(2s)
b+r1          # BoxTransport: ramp fixedpose-default → box-default(2s)→ 推理
              # 控制台看 "ramping ..." → "ramp complete, starting policy inference."
vel 0 0 0     # 静止站立
vel 0.3 0 0   # 慢速前进(训练 lin_x 范围内)
vel 0.5 0 0   # 上限速度
vel 0 0 0.2   # 偏航上限
a+r1          # 切回 LocoMode 验证 hot-swap
b+r1          # 再切回来验证幂等
l3            # 急停
exit
```

回归检查:`x+r1`(Dance)、`y+l1`(BeyondMimic)、`x+l1`(LocoNew)
仍然能跑,确认新增的 `state_cmd.base_lin_vel` 字段没影响其它策略。
