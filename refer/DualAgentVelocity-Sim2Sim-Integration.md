# DualAgentVelocity Sim2Sim 集成记录

## 背景

`upper_lower` 里新训练了一版"双 agent 抱箱行走"策略:上下半身由两个独立
actor 控制,合并成一个 29-DoF 动作。训练 cfg 在
`upper_lower/source/amp_tasks/amp_tasks/dual_agent/robots/g1/dual_agent_velocity_env_cfg.py`,
play 脚本是 `scripts/factoryIsaac/dual_agent_play.py --task velocity`。

任务语义和单网络的 `BoxTransportVelocity` 完全一致(机器人抱着 0.3m/1.5kg
的箱子,按 velocity command 行走;场景里的箱子几何参数也相同:
`pos=(0.32, 0.0, 1.05)`、立方、1.5kg、默认摩擦),区别只是架构:
- 上半身:480 维(96×5 历史) → 29 维 action,只用 `[15:29]` 的臂部分量
- 下半身:99 维(单帧,头带 `base_lin_vel`)→ 15 维 action,即 `[0:15]`
  腿 + 腰
- `CombinedActor` 把两者合并成 29 维,然后 **在 ONNX 图内部** reorder 到
  MuJoCo motor 顺序输出

目标:在 `FSMDeploy_G1` 里新增 `FSMStateName.DUAL_AGENT_VEL`,ONNX 推理,
绑定 `a+l1`,做 sim2sim 验证。复用 `BoxTransportVelocity` 那套 MuJoCo 箱子
生成 + 1s 硬 pin 悬吊机制。

与 `BoxTransportVelocity`(单网络 96-dim)的关键差异:

| | `BoxTransportVelocity`(单网络) | `DualAgentVelocity`(双 agent) |
|---|---|---|
| 架构 | 单 actor,全 29-DoF | 两个 actor,合并 29-DoF |
| ONNX 输入 | `obs[1,96]` | `upper_obs[1,480]` + `lower_obs[1,99]` |
| ONNX 输出 | `actions[1,29]` raw,**Isaac 顺序** | `actions[1,29]` raw,**MuJoCo 顺序** |
| 上身 obs 历史 | 无(单帧) | **5 帧 per-term ring buffer** |
| 下身 obs | (含在 96 中) | 独立 99 维,含 `base_lin_vel(3)` |
| action 处理 | 散射 `action_motor[j2m[i]] = q_cmd[i]` | ONNX 已重排,直接用 |
| `last_action` obs | 直接用 Isaac 顺序 raw action | **gather** MuJoCo 输出回 Isaac 顺序 |
| 真机可部署 | 可(若 IMU 提供足够信息) | **否**(需要 `base_lin_vel`) |

---

## 新增文件

### 1. `policy/dual_agent_velocity/DualAgentVelocity.py`

`FSMState` 子类,onnxruntime 后端,单 session 双 input。关键点:

- **类属性** `needs_transport_box = True`:deploy 循环据此决定是否传送
  `transport_box` body 到手心。`BoxTransportVelocity` 也被加了同属性,
  两者共用 deploy 侧的箱子逻辑。
- **`__init__`**:
  - 加载 yaml,建 6 个 per-term ring buffer 给 upper(ang_vel / gravity /
    cmd / qpos / qvel / last_action,shape `[H=5, term_dim]`)
  - Lower 单帧 99 维 scratch `self.lower_obs_flat`
  - 断言 ONNX 输入名含 `upper_obs` + `lower_obs`,输出含 `actions`,
    维度分别 480 / 99 / 29
  - warm-up 5 次
- **`enter`**:
  - 按 `joint2motor_idx` 重排 kps/kds/default_angles/**action_scale** 到
    MuJoCo 顺序(`BoxTransport` 只重排前三个,因为那边 action 在 Isaac 顺序
    做 scale+default 再散射;这里 action 已是 MuJoCo 顺序,所以 scale 也要
    提前在 MuJoCo 顺序上)
  - 清零所有 history,置 `_history_primed = False`
  - 快照当前 `state_cmd.q` 启动 ramp-in(yaml 里 `ramp_time=0.02`
    即"硬切"——训练随机化足够了,和最新的 BoxTransport 同策略)
- **`run`** 流程:
  1. ramp-in 阶段:线性插值到 default(MuJoCo 顺序),不喂 ONNX、不写 history
  2. 读 robot state,motor→Isaac 重排 `qj_obs` / `dqj_obs`
  3. 逐项 scale + clip(upper 和 lower 共用 ang_vel/gravity/cmd/qpos/qvel/
     last_action 项,只差 lower 多一个头部 `base_lin_vel`,**训练里这项没有
     scale**)
  4. 上身 6 个 per-term ring buffer:第一帧 `_prime` 全填同值,之后 `_push`
     滚动;拼接顺序 `[ang_vel×5, gravity×5, cmd×5, qpos×5, qvel×5, act×5]`
     (per-term oldest→newest,行优先 reshape)
  5. 下身单帧 concat `[lin_vel, ang_vel, gravity, cmd, qpos, qvel, last_a]`
  6. 一次 `sess.run(["actions"], {"upper_obs": u, "lower_obs": l})`
  7. 输出 `action_mujoco` 已是 MuJoCo 顺序的 raw action
  8. **gather 回 Isaac 顺序存 `self.action_isaac`** 供下一帧 last_action obs:
     `self.action_isaac[i] = action_mujoco[joint2motor_idx[i]]`
  9. `q_cmd_motor = action_mujoco * action_scale_reorder + default_angles_reorder`
     直接写 `policy_output.actions`
- **`checkChange`**:支持切到 PASSIVE / LOCO / LOCO_NEW / LOCO_NEW_ONNX /
  SKILL_BOX_TRANSPORT_V,其余保持 `DUAL_AGENT_VEL`

### 2. `policy/dual_agent_velocity/config/DualAgentVelocity.yaml`

`kps / kds / default_angles / action_scale / joint2motor_idx` **完全复用
BoxTransportVelocity.yaml**——两个 task 共享同一套 `G1_ACTION_SCALE`、
`G1_OPENSOURCE_CFG` 和抱箱初始姿态。

新字段:
- `ang_vel_scale: 0.2`、`lin_vel_scale: 1.0`、`gravity_scale: 1.0`、
  `cmd_scale: [1,1,1]`、`dof_pos_scale: 1.0`、`dof_vel_scale: 0.05`、
  `obs_clip_default: 100.0`、`last_action_clip: 12.0`
  (与训练 cfg `UpperBodyPolicyCfg` / `LowerBodyPolicyCfg` 的 `scale=` 参数
  一一对应;**`base_lin_vel` 没有 `scale=`,所以 `lin_vel_scale=1.0`,
  clip 靠通用 `obs_clip_default`**)
- `history_length_upper: 5`、`num_obs_upper: 480`、`num_obs_lower: 99`、
  `num_actions: 29`
- `ramp_time: 0.02`、`ramp_kp/kd_scale: 1.0`(硬切)

### 3. `policy/dual_agent_velocity/model/dual_agent_combined.onnx`

用训练仓库的导出脚本产出:

```bash
python scripts/factoryIsaac/dual_agent_export_onnx.py \
  --upper_policy logs/rsl_rl/g1_dual_agent_velocity/<ts>/upper_model_<N>.pt \
  --lower_policy logs/rsl_rl/g1_dual_agent_velocity/<ts>/lower_model_<N>.pt \
  --task velocity \
  --output_dir /home/.../FSMDeploy_G1/policy/dual_agent_velocity/model
```

文件不入 git。

---

## 修改文件

### 4. `common/utils.py`
新增 `FSMStateName.DUAL_AGENT_VEL = 16`、`FSMCommand.DUAL_AGENT_VEL = 16`。

### 5. `FSM/FSMState.py`
基类加类属性 `needs_transport_box = False`。这是抽象化 deploy 侧的箱子开关
——任何未来需要在 MuJoCo 里生成箱子的策略只要覆盖成 `True`,不用改 deploy
entrypoint。

### 6. `FSM/FSM.py`
- `from policy.dual_agent_velocity.DualAgentVelocity import DualAgentVelocity`
- `__init__` 里 `self.dual_agent_velocity_policy = DualAgentVelocity(...)`
- `get_next_policy` 加分支

### 7. `policy/loco_mode/LocoMode.py`
`checkChange` 加 `FSMCommand.DUAL_AGENT_VEL → FSMStateName.DUAL_AGENT_VEL`,
让从 LocoMode 可以直接切进 DualAgent(用户主力切换路径)。

### 8. `policy/box_transport_velocity/BoxTransportVelocity.py`
类属性 `needs_transport_box = True`(替代之前 deploy 里硬编码
`cur.name == SKILL_BOX_TRANSPORT_V` 的判断)。

### 9. `deploy_mujoco/deploy_mujoco.py` / `deploy_mujoco/deploy_mujoco_keyboard_input.py`
- 绑定 `A+L1`(手柄)和 `a+l1`(键盘)→ `FSMCommand.DUAL_AGENT_VEL`
- **箱子判断从硬编码名字改成读属性**:
  ```python
  is_box = getattr(cur, "needs_transport_box", False)
  ```
  这样 `transport_box` 的 spawn/despawn + 1s 硬 pin 悬吊对 BoxTransport 和
  DualAgent 都自动生效,不需要 `or` 枚举
- `deploy_real.py` **不绑**——下身策略需要 `base_lin_vel`,真机 IMU 拿不到

### 10. `CLAUDE.md`
Safety 段扩写:标注 DualAgentVelocity sim2sim-only,说明
`needs_transport_box` 约定,一个 `transport_box` body 被两个抱箱策略共用。

---

## 关键技术点

### 点 1:action 顺序约定反过来了

现有 `BoxTransportVelocity` 的 ONNX 输出是 Isaac 顺序的 raw action,部署端:
```python
q_cmd_policy = action * action_scale + default_angles   # 都是 Isaac
action_motor[j2m[i]] = q_cmd_policy[i]                  # 散射到 MuJoCo
```

`dual_agent_export_onnx.py` 的 `CombinedActor.forward()` 末尾做了:
```python
combined_action = combined_action[:, self.reorder_idx]
```
其中 `reorder_idx` 是 `MUJOCO_TO_ISAAC_IDX`,结果 **ONNX 输出已是 MuJoCo
顺序的 raw action**。部署端改成:
```python
q_cmd_motor = action_mujoco * action_scale_reorder + default_angles_reorder
self.policy_output.actions = q_cmd_motor
```
scale 和 default 提前在 `enter()` 里就重排到 MuJoCo 顺序。

### 点 2:last_action obs 必须 gather 回 Isaac

训练里 `last_action` 这个 ObsTerm 记录的是 **reorder 之前的原始 action**
(Isaac 顺序),而部署拿到的 ONNX 输出是 reorder 之后(MuJoCo 顺序)。
所以每帧多一步 gather:
```python
for i in range(29):
    self.action_isaac[i] = action_mujoco[self.joint2motor_idx[i]]
```
`self.action_isaac` 只用来喂下一帧 obs 的 last_action 项(对 upper 和 lower
都一样),不参与控制。两个 actor 都看到同一份 29 维 last_action(训练时
一致)。

### 点 3:两个 actor 共享 per-term 项,只差 lin_vel

upper_obs 和 lower_obs 里的 `ang_vel_s / gravity_s / cmd_s / joint_pos_rel
/ joint_vel_s / last_action_s` 是**同一份计算结果**,lower 额外多一个
`lin_vel_s` 头。所以 run() 里这六项只算一遍,各自拼一次。上身塞进 ring
buffer,下身直接塞进 99 维扁平。

### 点 4:`needs_transport_box` 替代硬编码

deploy 循环里原本是:
```python
is_box = (cur.name == FSMStateName.SKILL_BOX_TRANSPORT_V)
```
加第二个抱箱策略时最自然的做法是写 `or cur.name == DUAL_AGENT_VEL`,但这条
判断未来每加一个就要改一次。改成读类属性 `needs_transport_box` 后,deploy
entrypoint 对新策略是**零侵入**的:策略类自己声明"我要箱子"。

---

## 训练 vs 部署的参数校验

导出脚本 `dual_agent_export_onnx.py` 对 upper / lower 的输入维度做硬断言,
yaml 里的 `num_obs_upper=480` / `num_obs_lower=99` 必须和训练 cfg 一致,
否则 `__init__` 第一步就会 assert fail。好处是 sim2sim 跑起来之前就能发现
维度错配。

对 `joint2motor_idx` 没有自动校验——但部署侧所有抱箱类策略用同一套(G1
Isaac 29-DoF → MuJoCo 顺序),直接从 BoxTransport 抄过来即可。若训练侧改动
joint 顺序,这里需同步。

---

## 验证流程

```
python deploy_mujoco/deploy_mujoco_keyboard_input.py
```

```
start         # FixedPose 缓降
a+r1          # LocoMode,臂部在中性位置
a+l1          # DualAgentVelocity:ramp 到抱箱姿态(~1 tick)→ 推理
              # 控制台应看到 "ramping ..." → "ramp complete"
              # 紧接着 "BoxTransport: spawned box, pinned for 1.0s"
              # 箱子出现在两手之间并悬吊 1 秒
              # 1 秒后解除硬 pin,箱子交由策略抱住
vel 0.3 0 0   # 慢速前进验证 lower actor 跟踪 velocity
vel 0 0 0.2   # 偏航上限
vel 0.5 0 0   # 前进上限
a+r1          # 切回 LocoMode:箱子 park 回(100,100),手臂恢复中性位
a+l1          # 再切回来验证幂等:重新 ramp → 新的箱子 spawn
b+r1          # 切到单网络 BoxTransport:箱子逻辑无缝复用
              # (因为都走 needs_transport_box 属性)
l3            # E-stop
exit
```

回归检查:`x+r1`(Dance)、`y+l1`(BeyondMimic)、`x+l1` / `y+r1`
(LocoNew / LocoNewOnnx)仍然能跑,确认新增 FSMState 和 `needs_transport_box`
属性没有影响其它策略的状态路由。

---

## 可能的后续工作

1. **真机部署路径**:若要上真机,必须换一个不依赖 `base_lin_vel` 的下身
   policy(例如 Loco-G1-29dof 的标准 velocity 版,obs 96 维不含 lin_vel),
   上身保留抱箱 upper actor。改动只在 `DualAgentVelocity.py`(不构造
   `lin_vel_s`、`lower_obs_flat` 改 96 维)和 yaml(`num_obs_lower: 96`)。
2. **ramp_time 回调**:若换到鲁棒性较差的新策略,`ramp_time: 0.02` 硬切
   可能不够用,恢复 `BoxTransport` 早期的 `ramp_time: 2.0` +
   `ramp_kp_scale: 3.0` 即可。
3. **更多抱箱策略**:任何未来"需要箱子在手心"的策略,只需在 FSMState 子类
   上加一行 `needs_transport_box = True`,scene.xml 和 deploy entrypoint
   都不用动。
