# DualAgentTracking Sim2Sim 集成记录

## 背景

`upper_lower` 里新训练了一版**运动跟踪**双 agent 策略：上半身沿用抱箱
velocity 的 upper actor,下半身换成 motion tracking actor,实时跟一条
preprocessed 参考轨迹(npz)。训练 cfg:

- `upper_lower/source/amp_tasks/amp_tasks/dual_agent/robots/g1/dual_agent_env_cfg.py`
- `upper_lower/source/amp_tasks/amp_tasks/dual_agent/robots/g1/dual_agent_train_env_cfg.py`
- 当前 `motion_file = demo_walk_file = "data/demo/lafan/walk4_subject1.npz"`
- experiment: `g1_dual_agent`,目录 `logs/rsl_rl/g1_dual_agent/<ts>_joint_train/`

部署目标:`FSMStateName.DUAL_AGENT_TRACK`,ONNX 推理,键位 `a+l1`(手柄 A+L1 /
键盘 `a+l1`)。`b+l1` / `x+l1` / `y+l1` 预留给未来的 tracking demo,每个新
demo **自带一份 ONNX + 自带一份 motion npz**(训练时 motion 和 policy 绑定,
不能共享 ONNX 切 motion)。

与 `DualAgentBoxTransVel` 的关键差异:

| | `DualAgentBoxTransVel` | `DualAgentTracking` |
|---|---|---|
| 下身 task | velocity tracking | motion tracking |
| `lower_obs` 维度 | 96 | **109** |
| 下身 obs 组成 | `ang_vel(3)+grav(3)+cmd(3)+qpos(29)+qvel(29)+last_a(29)` | `lower_cmd(30)+grav(3)+ang_vel(3)+qpos(29)+qvel(29)+last_a(15)` |
| 下身 obs scale | 有 `ang_vel×0.2`、`dof_vel×0.05`、各种 clip | **全部 raw**,训练 cfg 没 scale(关节量都是裸值) |
| `last_action` 维度 | 29(upper/lower 共用) | **15**(只有下身 last_action) |
| 额外输入 | vel_cmd(来自手柄/键盘) | 一条 motion reference npz |
| 额外状态注入 | - | - |
| 控制节奏 | 50 Hz | 50 Hz,**motion fps 必须等于 50**(1 tick = 1 motion frame) |

历史维度:早期 121-dim tracking / 99-dim velocity 版本在 lower actor obs
里包含 `motion_anchor_pos_b(3)`、`motion_anchor_ori_b(6)`、`base_lin_vel(3)`。
2026-04-21 retrain 把这三项从 actor 移到 critic(privileged),actor 只靠
IMU + encoder + motion phase 推断误差修正方向。收敛慢 ~2x,最终精度等价。
目的是真机 IMU 拿不到 anchor 世界位姿和 body-frame 线速度。

注意:tracking 的 `base_ang_vel` / `joint_pos_rel` / `joint_vel_rel` 在训练
`LowerBodyPolicyCfg` 里都没有 `scale=` 和 `clip=`,只有 upper 那 6 项才走
scale + clip。**不要照抄 DualAgentBoxTransVel 的下身 scale**。

---

## 新增/修改文件

### 部署侧

#### 1. `policy/dual_agent_tracking/DualAgentTracking.py`

`FSMState` 子类,onnxruntime 后端,双 input + motion reference。关键点:

- **类属性** `needs_transport_box = True`:训练场景里机器人也是抱箱行走,
  deploy 沿用 `BoxTransportVelocity`/`DualAgentBoxTransVel` 那套 1s 硬 pin
  悬吊逻辑;deploy entrypoint 只读类属性,零侵入。
- **`MotionBuffer`**:从 yaml 指定的 npz 加载 `lower_joint_pos (T,15)` /
  `lower_joint_vel (T,15)` / `fps`,每 tick `advance()` 一次,wrap 到开头。
  注:2026-04-21 retrain 之后下身 actor 不再用 anchor,所以 npz 里的
  `torso_pos_w / torso_quat_w` 字段不再被读;preprocess 脚本仍然写出来,
  保留是为了换回旧策略时不用重跑 preprocess。
- **`__init__`**:
  - 加载 yaml + npz
  - 单帧 scratch 两块:`upper_obs_flat (96)` / `lower_obs_flat (109)`
  - 断言 ONNX 输入名含 `upper_obs`+`lower_obs`,输出含 `actions`,维度
    96 / 109 / 29
  - warm-up 5 次
- **`enter`**:
  - 按 `joint2motor_idx` 把 kps/kds/default_angles/**action_scale** 重排到
    MuJoCo 顺序(和 DualAgentBoxTransVel 相同)
  - 清 last_action、reset motion clock
  - 快照当前 `state_cmd.q` 做 ramp-in(yaml `ramp_time=0.5`,25 ticks)
- **`run`** 流程:
  1. ramp-in 阶段:线性插值到 default,不喂 ONNX、不走 motion
  2. 读 robot state(`gravity_ori / ang_vel / q / dq / vel_cmd`;deploy loop
     仍然每 tick 写 `anchor_pos_w/anchor_quat_w/base_lin_vel`,但本策略不读)
  3. motor→Isaac 重排 `qj_obs/dqj_obs`
  4. 读 motion:`lower_cmd_pos(15) + lower_cmd_vel(15)` 直接来自 npz
  5. Upper obs 96 维(单帧,带 scale + clip,和 LocoMode 完全一致):
     ```
     [0:3]    base_ang_vel × 0.2,            clip ±100
     [3:6]    projected_gravity              raw,  无 clip
     [6:9]    velocity_commands × cmd_scale  raw
     [9:38]   joint_pos_rel × 1.0            clip ±100
     [38:67]  joint_vel × 0.05               clip ±100
     [67:96]  last_action(29)                clip ±12
     ```
  6. Lower obs 109 维(**无 scale、无 clip**):
     ```
     [0:15]    lower_cmd_pos           (motion 当前帧 joint_pos)
     [15:30]   lower_cmd_vel           (motion 当前帧 joint_vel)
     [30:33]   gravity                 raw
     [33:36]   ang_vel                 raw
     [36:65]   joint_pos_rel           raw
     [65:94]   joint_vel               raw
     [94:109]  last_action_lower       Isaac 0..14
     ```
  7. 一次 `sess.run(["actions"], {"upper_obs": u, "lower_obs": l})`,两个
     input 进 ONNX 前再 `clip ±100` 兜底
  8. 输出 `action_mujoco` 已是 MuJoCo 顺序;gather 回 Isaac 顺序存
     `self.action_isaac`,下一帧 lower.last_action 取前 15 位;upper.last_action
     用完整 29 位
  9. `q_cmd_motor = action_mujoco * action_scale_reorder + default_angles_reorder`
     直接写 `policy_output.actions`
  10. `motion_buffer.advance()`

- **`checkChange`**:支持切到 PASSIVE / LOCO / LOCO_NEW / LOCO_NEW_ONNX /
  SKILL_BOX_TRANSPORT_V / DUAL_AGENT_BOX_TRANS_VEL,其余保持 `DUAL_AGENT_TRACK`

#### 2. `policy/dual_agent_tracking/config/DualAgentTracking.yaml`

- `kps/kds/default_angles/action_scale/joint2motor_idx` 完全复用
  `DualAgentBoxTransVel.yaml`(同一套 G1 29-DoF 动作空间、抱箱 default pose)
- `policy_path: "dual_agent_combined.onnx"`,`motion_file: "walk_tracking_ref.npz"`
- `num_obs_upper: 96`,`num_obs_lower: 109`,`num_actions: 29`,
  `history_length_upper: 1`(单帧,**不再是 5 帧 history**)
- Upper 侧 scale:`ang_vel_scale: 0.2`、`gravity_scale: 1.0`、
  `cmd_scale: [1,1,1]`、`dof_pos_scale: 1.0`、`dof_vel_scale: 0.05`、
  `obs_clip_default: 100.0`、`last_action_clip: 12.0`(和 box_trans_vel 一致)
- `anchor_body_name: torso_link`:**仅做引用**,代码里不读 — 留着是为了
  跟训练 cfg 名字对得上,以及未来有可能要加 anchor 回 actor 时少改一处
- `ramp_time: 0.5`、`control_dt: 0.02`

#### 3. `policy/dual_agent_tracking/model/dual_agent_combined.onnx`(user-provided)

训练方导出(见下文 §4),不入 git。

#### 4. `policy/dual_agent_tracking/motion/walk_tracking_ref.npz`(user-provided)

训练方 preprocess 生成(见下文 §4),不入 git。

#### 5. 修改的共享文件

- `common/utils.py`:`FSMStateName.DUAL_AGENT_TRACK = 17`、
  `FSMCommand.DUAL_AGENT_TRACK = 17`
- `FSM/FSM.py`:import + instance attr + `get_next_policy` 分支
- `policy/loco_mode/LocoMode.py`:`checkChange` 加 `DUAL_AGENT_TRACK` 出口
- `deploy_mujoco/deploy_mujoco.py`、`deploy_mujoco/deploy_mujoco_keyboard_input.py`:
  - 每 tick 注入 `state_cmd.anchor_pos_w = d.xpos[anchor_body_id]`、
    `state_cmd.anchor_quat_w = d.xquat[anchor_body_id]`(留作给未来重新引入
    anchor obs 的策略用;当前 tracking actor 不读)
  - 键位 `a+l1` → `FSMCommand.DUAL_AGENT_TRACK`
  - box 生成 pelvis-frame offset `(0.32, 0.0, 0.14)`(和抱箱策略共用)
- `deploy_real.py`:从 2026-04-21 retrain 起,actor obs 已经把 `base_lin_vel /
  anchor_pos_b / anchor_ori_b` 三项移到 critic privileged,真机 IMU + encoder
  够用。绑 `a+l1` 的步骤见 `refer/DualAgentTracking-Sim2Real-Guide.md`。

### 训练侧

#### 6. `upper_lower/scripts/factoryIsaac/dual_agent_tracking_preprocess_motion.py`

独立的 preprocess 脚本(unitree_isaaclab conda env 里跑),从训练用的原始 npz
(含 `joint_pos (T,29)` / `joint_vel (T,29)` / `body_pos_w (T,nB,3)` /
`body_quat_w (T,nB,4)` / `fps`)抽出部署所需 5 个字段写入新 npz。硬编码
`TORSO_BODY_IDX_IN_NPZ = 15`(30-body URDF 顺序,去掉 world 之后 torso_link
在 idx 15,和 MuJoCo 里 torso_link = body 16 对上)。

> **重训了一版新策略 / 要换箱子怎么办?**
> 运维流程(导出 ONNX、拷贝 artifact、调箱子几何和生成位置)单独抽到了
> `refer/Sim2Sim-Ops-Guide.md`。设计/代码路径说明留在这份里。

---

## 验证流程(walk demo)

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1
python deploy_mujoco/deploy_mujoco_keyboard_input.py
```

```
start         # FixedPose 缓降
a+r1          # LocoMode
a+l1          # DualAgentTracking 入场:
              #   "DualAgentTracking: ramping ..." (0.5s)
              #   "DualAgentTracking: ramp complete, starting policy inference."
              #   "BoxTransport: spawned box, pinned for 1.0s."
              #   箱子出现在手心,1s 后解除硬 pin
              # 机器人开始按 walk demo 行走(控制台 motion frame 递增)
a+r1          # 切回 LocoMode:"BoxTransport: parked box.",臂部回中性
a+l1          # 再切回来验证幂等,箱子重 spawn,motion clock 归零
l3            # E-stop
exit
```

回归:`b+r1`(BoxTransportVelocity)、`x+r1`(DualAgentBoxTransVel)、
`y+r1`(BeyondMimic)仍然能跑,确认 `needs_transport_box=True` 和 motion
buffer 没有污染其它策略。

---

## 坑点汇总

1. **`lower_obs` 下半身动力学项(grav/ang_vel/qpos_rel/qvel/last_a)全是 raw**
   ——照抄 DualAgentBoxTransVel 的 96 维 lower 会错把 `ang_vel×0.2` 喂进
   tracking lower,机器人慢慢 drift 到崩。训练 cfg 是权威。`lower_cmd_pos /
   lower_cmd_vel` 也直接来自 motion npz,不要二次缩放。
2. **`last_action` 在 lower obs 里只有 15 维**(下身 action),不是 29 维。
   Isaac 顺序的前 15 位即 `action_isaac[:15]`;upper obs 用完整 29 维。
3. **motion 50 fps = policy 50 fps 是硬约束**。MotionBuffer 每 tick
   `advance()`,wrap 到 0。训练若换 20 fps,部署侧要么插值要么降 policy
   频率。
4. **upper 单帧、不要 history**:2026-04-21 retrain 之后 upper actor 也降到
   单帧(96 维),和老的 5 帧 480 维 ONNX 不兼容。yaml 里的
   `history_length_upper` 留了 1 占位,代码里没真的开 ring buffer。换回老
   ONNX 必须同时把 `num_obs_upper` 改回 480、`history_length_upper` 改回 5
   并恢复 ring buffer 代码。
5. **anchor / base_lin_vel 仍然由 deploy loop 注入**到 `state_cmd`,但
   tracking actor 不读它们(只在 critic 训练时用,部署的 ONNX 里没有这层)。
   留着是为了让别的 policy(以后有需要)可以复用同一套 deploy loop。
6. **torso body idx = 15** 是 preprocess 时 npz 里去掉 world 之后的顺序,
   MuJoCo body list 里 torso_link 是 body 16(world=0、robot root=1 往后数)。
   `scene.xml` 若改动 body 顺序(例如加了 prop),preprocess 脚本断言会直接
   fail —— 这条只跟 preprocess 相关,deploy 不读 anchor 不会触发。
7. **motion 文件不入 git**(model 和 motion 都是 artifact)。
   `.gitignore` 已覆盖 `policy/*/model/*.pt`、`policy/*/model/*.onnx`、
   `policy/*/motion/*.npz`。
8. **真机部署**:见 `refer/DualAgentTracking-Sim2Real-Guide.md`。obs 简化后
   IMU + encoder 够用,但箱子要靠人手递,绳吊逻辑只在 MuJoCo 里有意义。
