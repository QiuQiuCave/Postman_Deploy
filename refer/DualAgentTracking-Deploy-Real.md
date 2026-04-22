# deploy_real.py 真机使用手册

**目的**:从零把 Unitree G1 开机,到完整走完一遍 DualAgentTracking(walk demo
抱箱行走)。配套的安全梯子与 abort 触发条件见
`refer/DualAgentTracking-Sim2Real-Guide.md`;本文聚焦"按什么键、出什么现象"。

---

## 0. 前提检查清单

**硬件**:

- [ ] Unitree G1 开机、进 debug 模式(开机双击电源键 → L2+A 进 lying → L2+B 起身)
- [ ] 腰部固定支架如果装着,按官方步骤解锁
- [ ] 安全绳 / 吊架 已挂(第一次跑必挂,不妥协)
- [ ] 0.3 m / ~1.5 kg 立方体箱子准备好在操作员手边
- [ ] 周围 2~3 m × 1 m 清空

**网络**:

- [ ] 机器人网线接部署机,网卡名已填到 `deploy_real/config/real.yaml`
  的 `net:` 字段(默认 `enp2s0`,不同机器可能是 `enp3s0` / `eno1` 等,
  `ip link` 查)
- [ ] DDS 能通:`ping` 机器人管理口 IP 能回,且下一步 `python deploy_real.py`
  打得出 `Successfully connected to the robot.`

**软件**:

- [ ] §1.1~1.2 的系统库和 cyclonedds 已装(见 `UV-Deploy-Setup.md`)
- [ ] `uv sync` 装完,`uv run python -c "import unitree_sdk2py"` 无报错
- [ ] `policy/dual_agent_tracking/model/dual_agent_combined.onnx` 和
  `policy/dual_agent_tracking/motion/walk_tracking_ref.npz` 都在(`git pull`
  应该带下来了)
- [ ] 电量 ≥60%(双 actor + 行走比 LocoMode 耗电高)

---

## 1. 键位总览

`deploy_real.py` 当前绑定的全部按键:

| 按键 | 触发动作 | 对应 FSMState |
|---|---|---|
| **F1** | E-stop(damping) | `PassiveMode` |
| **Start** | 复位到 default pose | `FixedPose`(POS_RESET 命令) |
| **A + R1** | 行走模式 | `LocoMode` |
| **X + R1** | 舞蹈(SKILL_1) | `Dance` — 真机上稳定,hardware-verified |
| **A + L1** | 抱箱跟踪(walk demo) | `DualAgentTracking` ⭐ |
| **Select** | 退出程序(不是 E-stop,只退 Python) | - |

**摇杆**(任何 loco-family state 生效):

| 摇杆 | 对应 `vel_cmd` | 量纲 |
|---|---|---|
| 左摇杆 Y(前后) | `vel_cmd[0]` | 前进 / 后退 |
| 左摇杆 X(左右)× -1 | `vel_cmd[1]` | 左移 / 右移 |
| 右摇杆 X × -1 | `vel_cmd[2]` | 偏航(左转 / 右转) |

> 所有 loco-family 的速度缩放由各策略 yaml 的 `cmd_range.lin_vel_x/y` +
> `ang_vel_z` 控制,摇杆输出 [-1, 1] 映射到这些区间。

**未绑定**(有意不绑,真机不能跑):

- `b+r1 / x+r1 / y+r1`(BoxTransport / DualAgentBoxTransVel / BeyondMimic
  的 sim2sim 版)—— 这几个 policy 的 lower actor obs 有 `base_lin_vel`,
  真机 IMU 没有 body-frame 线速度估计器
- `b+l1 / x+l1 / y+l1`:预留给未来的 tracking demo,每条 demo 一份
  ONNX + 一份 motion npz

---

## 2. 启动 deploy_real.py

```bash
cd ~/code/postman/Postman_Deploy
uv run python deploy_real/deploy_real.py
```

**预期输出**:

```
Successfully connected to the robot.
Locomotion policy initializing ...
loco_new_mode policy initializing (backend=pt) ...
...  (FSM 构造所有策略,约 3~5 秒)
DualAgentTracking policy initializing (backend=onnx, dual-input) | motion frames=8195 @ 50Hz | duration=163.90s
initalized all policies!!!
current policy is  passive_mode
Enter zero torque state.
Waiting for the start signal...
```

程序停在 `Waiting for the start signal...`,这时机器人所有关节 **零力矩
(damping)**,你可以用手掰关节验证(应该能被缓阻尼感地压动)。

---

## 3. 完整抱箱跟踪流程

### 3.1 起身到 LocoMode

```
Start        → "current policy is  fixedpose_mode"
               FixedPose 缓降到 default 抱箱姿(上肢已经摆成抱箱手型,
               下肢标准站立)。约 2s 完成。
A + R1       → "current policy is  loco_mode"
               LocoMode 接管,原地站立,可以用左摇杆走起来测试。
               这一步操作员**还没递箱子**。
```

### 3.2 切到 DualAgentTracking,操作员递箱子

**关键配合**:ramp 完成的瞬间必须把箱子递到手心里。

```
A + L1       → "current policy is  dual_agent_tracking_mode"
               终端连续打出:
                 "DualAgentTracking: ramping to default pose over 0.50s
                  (25 ticks) before policy inference starts."
               0.5 秒后:
                 "DualAgentTracking: ramp complete, starting policy
                  inference."
               ★ 看到这行立刻递箱子 ★
               机器人手型已经是抱箱姿,两手间距刚好 0.3 m。把箱子平举
               塞进两手之间,操作员松手,policy 的合拢力把箱子稳住。
```

**递箱子动作要点**:

- 箱子平举,中心对准机器人胸前 ~30 cm(大约 pelvis 前方 0.32 m)
- 一次递进去,不要反复调整
- 松手前感受一下机器人两手是否已经合拢;如果还没稳,继续托 1~2 秒
- 如果箱子滑了,**立刻 F1**,不要去抓

### 3.3 行走

箱子稳住之后,`vel_cmd` 开始生效(Walk demo 本身也有 base velocity
reference,但 yaml 里 `cmd_range` 留着给人为遥控调节):

```
左摇杆向前  → 缓慢前进
左摇杆向后  → 后退
右摇杆 X   → 原地偏航
```

Walk demo motion 是 8195 帧 / 163.9 s 循环的,走完一轮 motion clock wrap
回头继续。行走时终端不打日志(避免刷屏);想看 motion frame 可以在代码
里给 `DualAgentTracking.run()` 加 print。

### 3.4 切回 / 退出

```
A + R1       → 切回 LocoMode,policy 停住、手型回中性
               → 操作员把箱子接回来(真机没有传送箱子的机制,
                  sim2sim 里的 "parked box" 在真机上是 no-op)
F1           → PASSIVE / E-stop(所有电机 damping,机器人会慢慢坐下)
Select       → 退出 python 程序(之前必须已经 F1 或物理把机器人放稳,
                否则脚本退出瞬间失去控制)
```

**重进**:同一次运行里可以反复 `a+l1 → a+r1 → a+l1`,每次 re-entry 都会
重新 ramp + 重置 motion clock(`MotionBuffer.reset()`),箱子要人再递一次。

---

## 4. 第一次跑必须走"安全梯子"

跟 sim2sim 通过**不等于**真机敢裸跑。按
`refer/DualAgentTracking-Sim2Real-Guide.md` §5 的 5 级梯子走:

| 阶段 | 设定 | 通过标准 |
|---|---|---|
| **5.1 架空静态** | 吊起,不递箱子,`a+l1` 后立刻 F1 | 手型正确,关节不飞 |
| **5.2 架空递箱子** | 吊起,递箱子,跑 5~10 s | 不掉箱子,关节温度正常 |
| **5.3 落地原地** | 脚着地但绳子轻拉,递箱子,5 s 后切回 LocoMode | 能站能切 |
| **5.4 解绳短距离** | 人手扶,递箱子,走 1~2 m,`a+r1` | 不偏不倒 |
| **5.5 完整 demo** | 5.4 过 3 次以上才做 | walk demo 整段跑完 |

每阶段完了再上下一阶段。出事回退两级。

---

## 5. 立刻 abort 的触发条件

见到任何一条 **立即 F1**,不要犹豫:

- 单脚悬停 > 0.5 s(policy 卡)
- pelvis 目视倾角 > 15°
- 任一关节单个 cycle 抖动 > 5°
- 箱子从手里滑出去(policy 瞬间失去载荷反力,容易扑街)
- 关节温度告警(Unitree 监控软件上看)
- 操作员任何不确定

---

## 6. 常见报错

| 报错 | 原因 | 解法 |
|---|---|---|
| `Could not find interface enp2s0` | 网卡名不对 | `ip link` 查实际网卡名,改 `deploy_real/config/real.yaml` 的 `net:` |
| 长时间卡在 `Successfully connected to the robot.`,不进 `Enter zero torque state` | DDS topic 订阅了但没收到 `rt/lowstate` | 机器人端的 motion_control_service 没起 / 或者你同一个网卡上还有别的 ROS/DDS 进程抢订阅。`ros2 topic hz` 检查一下 `/lowstate` |
| `import unitree_sdk2py` 报 `cannot import name 'b2'` | `external/unitree_sdk2_python/` 没 clone 或没 editable | 见 `UV-Deploy-Setup.md` §3 |
| `libddsc.so.0: cannot open shared object file` | `LD_LIBRARY_PATH` 没设 | 见 `UV-Deploy-Setup.md` §1.2 最后两行 |
| ramp 完成但机器人立刻往一侧倒 | IMU 零偏校准飘了 | Unitree 工具重新做 IMU 校准,或者重启机器人 |
| ramp 完成后手根本没合拢 | 箱子递得太晚,policy 已经进入 walk phase | F1,回 3.1 重来,下次紧跟 "ramp complete" 这条打印 |
| `control loop over time.` 刷屏 | CPU 不够 / 电脑太忙 | 关掉浏览器、IDE,用 `top` 看 python 是不是 >100% CPU。onnxruntime 双 actor 在 i5/i7 笔记本上 2 ms/tick 够用,如果超标换机器 |

---

## 7. 改代码后

如果你加了新策略(新 L1 键位),至少改这 4 个地方才能在 deploy_real 上
触发(R1 组已经完整):

1. `common/utils.py`:`FSMStateName` + `FSMCommand` 各加一个枚举值
2. `FSM/FSM.py`:`__init__` 里 `self.xxx_policy = XxxPolicy(...)` + `get_next_policy` 加分支
3. `policy/loco_mode/LocoMode.py`:`checkChange` 加 `FSMCommand.XXX → FSMStateName.XXX`
   (从 LocoMode 能切进新 state)
4. **`deploy_real/deploy_real.py`**:在 "L1 group" 注释块下仿 `a+l1`
   加一行按键 → `FSMCommand.XXX` 绑定

deploy_mujoco 的两个入口(xbox + 键盘)也要对应加,否则 sim2sim 跟真机
行为会不一致,回归时容易坑。
