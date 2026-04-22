# DualAgentTracking 真机部署指南

把 sim2sim 验证通过的 `DualAgentTracking`(iter-15000 of
`2026-04-21_20-01-44_joint_train`)放到真 G1 上跑。本文只讲真机相关动作 ——
sim2sim 集成 / 运维细节见 `refer/DualAgentTracking-Sim2Sim-Integration.md`
和 `refer/Sim2Sim-Ops-Guide.md`。

---

## 0. 为什么这版策略可以上真机

老的 121-dim lower 版本在 actor obs 里包含三项**真机拿不到**的量:

- `motion_anchor_pos_b(3)` / `motion_anchor_ori_b(6)`:需要世界坐标系下的
  `torso_link` 位姿;真机 IMU 只能给姿态(quat),拿不到世界位移。
- `base_lin_vel(3)`:body-frame 线速度;真机没有这个估计器(`deploy_real.py`
  L134 注入零向量,可以看到 TODO)。

2026-04-21 retrain 把这三项移到 critic privileged signals,actor 只剩
**IMU 姿态 + 角速度 + 关节 q/dq + motion phase + 上一帧 action**,这些真机
全部直接可观测。所以现在可以走 `deploy_real.py`。

仍然真机**拿不到**的是箱子 —— 但 IsaacLab 训练本来就只用箱子作为静态质量
载荷,policy 没有视觉输入,所以"箱子在哪"这个信息没进 obs。真机要做的
只是物理上把箱子塞到机器人手里,policy 自己不关心。

---

## 1. 准备工作(一次性)

### 1.1 artifact 就位

在部署机器人那台主机上:

```
FSMDeploy_G1/
└── policy/dual_agent_tracking/
    ├── model/dual_agent_combined.onnx    # 96/109/29 三 IO,Sim2Sim 跑通的同一份
    └── motion/walk_tracking_ref.npz      # 同一条 walk demo,50 fps
```

两份 artifact 直接从 sim2sim 验证用的目录拷过来,**不要在真机端重新导出**
——sim2sim 通过的就是这两份字节流,任何重导出都引入未验证的差异。

### 1.2 sanity check(不连机器人)

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1
conda activate unitree_sim2sim   # 部署侧 onnxruntime + mujoco env

python -c "
import onnxruntime as ort, numpy as np
sess = ort.InferenceSession('policy/dual_agent_tracking/model/dual_agent_combined.onnx',
                            providers=['CPUExecutionProvider'])
ins = {i.name: i.shape[-1] for i in sess.get_inputs()}
print('inputs:', ins)                       # 期望 {'upper_obs': 96, 'lower_obs': 109}
out = sess.run(['actions'], {'upper_obs': np.zeros((1,96), np.float32),
                             'lower_obs': np.zeros((1,109), np.float32)})[0]
print('action shape / norm:', out.shape, np.linalg.norm(out))
"
```

零输入下 action 范数应该在 1~10 量级。任何报错(维度不匹配、找不到 input
名)说明你拿错了 ONNX。

---

## 2. 在 deploy_real.py 里绑 a+l1

`deploy_real.py` 里目前只绑了 `LocoMode`(`a+r1`)、`SKILL_1`(`x+r1`)、
`PASSIVE`(`F1`)、`POS_RESET`(`start`)。需要加 **L1 组**给 tracking。

### 2.1 编辑 `deploy_real/deploy_real.py`

在 L107~113 的按键扫描段后面加:

```python
# L1 group: tracking-family policies
if self.remote_controller.is_button_pressed(KeyMap.A) and \
   self.remote_controller.is_button_pressed(KeyMap.L1):
    self.state_cmd.skill_cmd = FSMCommand.DUAL_AGENT_TRACK
```

注意:`is_button_pressed` 不是 edge-trigger 而是 level-trigger,所以**轻按
一次**A+L1 会持续触发若干 tick。`FSMState.checkChange` 内部会把
`skill_cmd` 重置成 `INVALID`,所以这是无害的,但操作上仍然按一下就放开,
不要长按。

### 2.2 不需要改的地方

- **不要给 `state_cmd.anchor_pos_w` / `state_cmd.anchor_quat_w` 注入数据。**
  当前 actor 不读它们;`StateAndCmd` 里这两个字段在真机入口里也压根没
  写过(只有 MuJoCo 入口才写),保持现状即可。
- **不要给 `base_lin_vel` 估速度。** L134 的 `np.zeros(3)` 占位足够 ——
  tracking actor 不读这一项;唯一会读它的 `LOCO_NEW` 系列在真机上本来就
  禁用。
- FSM wiring(`FSM.py` / `LocoMode.checkChange`)不用改 —— sim2sim 集成
  时已经接好了。

---

## 3. 真机运行流程

### 3.1 启动前检查

| 项 | 要求 |
|---|---|
| 机器人架空 | 第一次跑必须挂安全绳/吊架,脚不沾地 |
| 周围空间 | walk demo 大约前进 2~3 m,左右散开 ≥1 m |
| 箱子准备 | 0.3 m 立方体,~1.5 kg(和训练 / sim2sim 一致),提前递到操作员手边 |
| 网络 | DDS 跑通(`ChannelFactoryInitialize` 不报错) |
| 电池 | ≥60%,tracking 双 actor 推理 + walking 比 LocoMode 耗能高 |

### 3.2 启动序列

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1
conda activate unitree_sim2sim
python deploy_real/deploy_real.py
# 终端打印 "Successfully connected to the robot."
# 等到 "Enter zero torque state. Waiting for the start signal..."
```

按键序列:

```
start         → POS_RESET,FSM 进 FixedPose,缓降到 default 抱箱姿
                (注意:这一段已经是抱箱手型,操作员目视确认手臂没卡)
a+r1          → LocoMode,机器人原地站立
                (此时人**还没递箱子**,两手空摆但手型保持抱箱)
a+l1          → DualAgentTracking 入场
                控制台:
                  "DualAgentTracking: ramping ..."(0.5s 缓 ramp 到 default)
                  "DualAgentTracking: ramp complete, starting policy inference."
                ramp 完成的瞬间 → 操作员**两手把箱子塞进机器人手心**,
                与 policy 输出的抱箱手型对齐。3~5 秒内手感就能让 policy
                的合拢力把箱子稳住。
                                 然后 policy 按 walk demo 开始迈步。
F1            → PASSIVE(E-stop),所有电机 damping
                **任何异常立刻按这个**
```

### 3.3 切换回 LocoMode / 终止

```
a+r1          → 切回 LocoMode,机器人停下、手型回中性 → 操作员把箱子接回手里
                (真机没有"传送箱子"机制,sim2sim 里那段 box despawn 在真机
                 上 no-op,因为 needs_transport_box 只影响 MuJoCo deploy loop)
F1            → PASSIVE
select        → 退出程序
```

---

## 4. 安全考虑

### 4.1 绳吊 vs. 真机的差别

sim2sim 里 ramp 完成后会有 1s 的"硬 pin 绳吊",给手合拢的时间。真机**没
有**这个机制 —— `needs_transport_box` 这个类属性只被 `deploy_mujoco/*.py`
读取,`deploy_real.py` 完全忽略。所以真机的等价操作是"操作员的手"。

- ramp 完成的瞬间 = 你递箱子的瞬间。早了 policy 还在 PD-hold 到 default,
  手不会主动合拢;晚了 policy 已经开始走步,手在动。
- 第一次跑建议:让另一个人盯 `DualAgentTracking: ramp complete` 这条
  console line,第一次"塞箱子"动作两人配合更稳。

### 4.2 步态收敛差异

iter-15000 在 IsaacLab 里收敛得很干净,但 `dual_agent_play.py --num_envs 16`
的统计数值会**略好于真机** —— sim 的 PD 是理想 PD,真机有齿轮回差、
布线阻抗、电池电压下垂。预期差异:

- 步态频率正常;
- 落脚位置可能偏前/偏后 ~3 cm;
- pelvis 倾角 RMS 可能比 sim 大 1~2°;
- 长时间(>30 s)行走如果开始打转或往一侧倾斜,先 E-stop,再回来检查 IMU
  零偏校准。

### 4.3 Abort 触发条件

任何一条满足就**立刻 F1**:

- 机器人单脚悬停 >0.5 s(policy 卡了);
- pelvis 倾角目视 >15°;
- 任一关节抖动幅度 >5° / cycle;
- 箱子从手里滑出来(失去载荷反作用力,policy 会瞬间高估推力 → 容易扑街);
- 操作员任何不确定。

---

## 5. 验证梯子

不要一上来就走步。按这个顺序加压:

| 阶段 | 设定 | 通过标准 |
|---|---|---|
| **5.1 桌面静态** | 机器人架空,不递箱子,只跑到 ramp_complete 后**立刻 F1** | 确认手型正确,关节没飞 |
| **5.2 架空递箱子** | 同上,ramp 完成后递箱子,policy 走 5~10 s | 没掉箱子,关节温度正常 |
| **5.3 落地原地** | 脚踩地,但绳子保持轻拉,递箱子,policy 走 5 s 后 `a+r1` 切回 | 站得住、能正常切换 |
| **5.4 解除绳子,短距离** | 操作员手扶,递箱子,走 1~2 m,`a+r1` | 不偏不倒 |
| **5.5 完整 demo** | 5.4 通过 ≥3 次后 | walk demo 整段 |

每升一级都重新跑一次 `dual_agent_play.py` 在训练机上对比,确认 sim 和真机
的差异曲线没有漂移到不可控。

---

## 6. 跟 sim2sim 不同的地方一览

| | sim2sim(MuJoCo) | sim2real |
|---|---|---|
| 入口脚本 | `deploy_mujoco/deploy_mujoco_keyboard_input.py` 或 `deploy_mujoco.py` | `deploy_real/deploy_real.py` |
| 键位 | 键盘 `a+l1` 或手柄 A+L1 | 仅手柄 A+L1 |
| 箱子 | 自动 spawn + 1s 硬 pin 绳吊 | 操作员手递,无绳吊 |
| `anchor_pos_w/quat_w` | 每 tick 注入(actor 不读) | 不注入(actor 也不读) |
| `base_lin_vel` | 注入真值(actor 不读) | 注入零(actor 不读) |
| E-stop | `l3` / `select` | `F1` / `select` |
| 失败成本 | 重启 sim | 摔机器人 + 滑箱子 |

---

## 7. 出事之后

- **机器人摔了**:F1 → 检查 IMU 校准 → 关节零位重校 → 回 §5.1 重来。不要
  跳级。
- **箱子滑了**:别去抓!F1,policy 在失去载荷的瞬间力矩会瞬间反向,容易
  打脸或撞胸。
- **走偏 / 转圈**:在 sim2sim 里 walk demo 是直走的,真机出现偏航要先看
  是不是 IMU 零偏。如果 sim 里也偏,说明 motion 本身就偏(LAFAN 里 walk4
  并不严格直行),不是真机问题。

---

## 8. 后续工作(可选)

- **多 motion 切换**:每个新 tracking demo 一份独立 ONNX + npz + state 类,
  L1 组还有 `b+l1 / x+l1 / y+l1` 三个键位。新 state 类的克隆步骤见
  `Sim2Sim-Ops-Guide.md` §2,对真机的额外动作只是"在 deploy_real.py 里再
  绑一组按键 + checkChange 加分支"。
- **加 `base_lin_vel` 估计器**:如果将来要部署回老的 121-dim 策略或者新
  策略需要这个 obs,真机要么外挂 VIO,要么从 IMU 加速度积分(漂)+ 接触
  state 修正。当前 tracking 用不到。
