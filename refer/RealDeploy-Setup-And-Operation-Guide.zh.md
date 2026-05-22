# G1 三条 selected tracking clips 真机部署指南

本文面向当前已经在 sim2sim 侧筛选出来的三条短参考 motion,把它们通过
`FSMDeploy_G1` 的 `deploy_real.py` 上到 G1 真机。本文既是环境安装手册,也是
现场操作 checklist。实验前建议直接按本文从上到下过一遍。

当前三条 runtime clips:

| 技能 | 真机按键 | runtime motion | 片段 | 长度 | SHA256 |
|---|---|---|---|---|---|
| walk tracking | `A+L1` | `policy/dual_agent_tracking/motion/walk_tracking_ref.npz` | `walk4_subject1 1:17-1:40` | `1150` frames / `23.00s` | `8c970e646fa59af71dd9e95959d0098496b4af81b08a6bd4ee82028b08dcfd55` |
| run tracking | `B+L1` | `policy/dual_agent_run_tracking/motion/run_tracking_ref.npz` | `run1_subject2 2:15-2:22` | `350` frames / `7.00s` | `d1a6c74a8c1ea640882dd6b71fde8b79e2c4ebb8556620aa65956ef93955720d` |
| jump tracking | `X+L1` | `policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz` | `jumps1_subject2_first75 0:23-0:30` | `350` frames / `7.00s` | `9df2a6e567626e4939a95a2609f6d22e61bab868fe2d300ed45f5d0209199488` |

这三条都共用同一套 staged box workflow:

```text
Start -> A+R1 -> B+R1 -> 操作员递箱子 -> X+R1 -> A/B/X+L1
```

其中最后一步根据目标技能选择 `A+L1`、`B+L1` 或 `X+L1`。

---

## 0. 当前进展快照

截至 2026-05-23,当前 sim2real 准备工作已经推进到"可以按安全梯子上真机
验证"的状态:

- 已从长参考 motion 中裁出三条适合 6m x 8m 场地初测的 selected clips:
  walk `1:17-1:40`, run `2:15-2:22`, jump `0:23-0:30`。
- 三条 selected clips 已复制到对应 runtime 路径,策略会直接读取这些短片段:
  `walk_tracking_ref.npz`, `run_tracking_ref.npz`, `jump_tracking_ref.npz`。
- 三条原始长 runtime reference 已在各自 `motion/` 目录中保留 `.full_...`
  备份,后续可以恢复完整 motion。
- `reference_motion_bank/selected/` 已作为人工筛选 motion bank,记录了当前
  active runtime 和备用 jump 片段。
- `deploy_real.py` 已接入 staged box workflow 和 L1 tracking-family:
  `A+R1 -> B+R1 -> X+R1 -> A/B/X+L1`。
- `real.yaml` 已打开 walk/run/jump 三个 selected clips 的真机门控,保留 dance
  关闭。
- `deploy_real.py` 会在启动时打印每条 tracking runtime motion 的帧数和时长,
  现场能直接发现是否拿错文件。
- 真机日志默认开启,每次运行会写入 `logs/real_deploy/real_deploy_*.csv`。
- 已完成静态校验:Python 编译通过,FSM 可初始化,三条 runtime SHA256 与
  motion bank 记录一致,三条 motion 长度分别为 `1150 / 350 / 350` frames。

尚未完成的是正式真机验证。下一步必须按本文 §12 的 safety ladder 从架空开始:
先 walk,再 run,最后 jump。尤其 run 的参考净位移约 `6.9m`,必须沿 8m 长边布置;
jump 必须先吊起或安全绳承重测试。

---

## 1. 当前代码适配状态

真机入口在 `deploy_real/deploy_real.py`:

- `A+R1` -> `LocoMode`
- `B+R1` -> `BoxHandoffStand`
- `X+R1` -> `BoxHoldStand`
- `A+L1` -> `DualAgentTracking` / walk
- `B+L1` -> `DualAgentRunTracking` / run
- `X+L1` -> `DualAgentJumpTracking` / jump
- `Y+L1` -> `DualAgentDanceTracking` / dance,当前不属于三条 deploy clips

`deploy_real/config/real.yaml` 当前已经打开三条 selected clips:

```yaml
enable_box_handoff_stand: true
enable_box_hold_stand: true
enable_walk_tracking: true
enable_run_tracking: true
enable_jump_tracking: true
enable_dance_tracking: false
```

运行 `deploy_real.py` 时会打印当前真机门控和 runtime motion 长度,用于确认现场
没有拿错配置:

```text
RealDeploy enabled skill gates:
  B+R1 BoxHandoffStand: True
  X+R1 BoxHoldStand:    True
RealDeploy tracking runtime clips:
  A+L1 walk: enabled=True | motion=1150 frames @ 50Hz (23.00s)
  B+L1 run: enabled=True | motion=350 frames @ 50Hz (7.00s)
  X+L1 jump: enabled=True | motion=350 frames @ 50Hz (7.00s)
  Y+L1 dance: enabled=False | motion=...
```

如果现场只想测 walk,把 run / jump 的开关重新改成:

```yaml
enable_run_tracking: false
enable_jump_tracking: false
```

---

## 2. 现场风险边界

当前强硬件证据仍以 walk + box 为主。run 和 jump 已经接入 sim2real 入口并使用
短 clip,但必须按本文安全梯子单独验证,不能因为 walk 通过就直接裸跑。

参考运动空间估计:

| 技能 | 净位移 | 平面 bbox | 参考路径长 | 场地建议 |
|---|---:|---:|---:|---|
| walk | `1.30m` | `3.61m x 2.07m` | `9.71m` | 可以在 6m x 8m 区域内测,但要看转向和箱子是否带偏 |
| run | `6.90m` | `7.14m x 0.44m` | `7.66m` | 必须沿 8m 长边,前后留缓冲,不建议沿 6m 方向 |
| jump | `3.40m` | `1.55m x 3.36m` | `5.28m` | 需要横向留足 4m 以上,优先架空和安全绳承重测试 |

任何阶段出现以下情况,立即 `F1`:

- pelvis 快速前倾、后仰或侧倾
- 箱子从手中明显下滑
- 单侧肩、肘、腕出现高频抖动
- 膝关节突然塌陷或脚尖明显绊地
- 控制台连续出现 `control loop over time.`
- 操作员看不清下一步或场地边界不够

`Select` 只退出 Python,不是急停。急停优先级是:

```text
F1 damping -> 物理急停 / 断使能 -> Select 退出程序
```

---

## 3. 部署机系统依赖

以下按 Ubuntu 部署机写。纯真机部署不需要 GPU。

### 3.1 apt 依赖

```bash
sudo apt update
sudo apt install -y cmake build-essential git curl
sudo apt install -y libglfw3 libglew2.2 libegl1
sudo apt install -y libgl1 libglib2.0-0
```

`libglfw3` / `libglew2.2` / `libegl1` 主要是 MuJoCo viewer 需要。真机只跑
`deploy_real.py` 理论上可以不用,但同一台机器最好能做 sim2sim smoke test。

### 3.2 CycloneDDS native 库

`unitree_sdk2py` 依赖本地 `libddsc.so`。先安装 CycloneDDS,再同步 Python 环境:

```bash
cd ~
git clone --depth 1 --branch releases/0.10.x \
  https://github.com/eclipse-cyclonedds/cyclonedds.git
cd cyclonedds
mkdir -p build install
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$HOME/cyclonedds/install" -DBUILD_EXAMPLES=OFF
cmake --build . --target install --parallel "$(nproc)"

echo 'export CYCLONEDDS_HOME="$HOME/cyclonedds/install"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"' >> ~/.bashrc
source ~/.bashrc
```

检查:

```bash
echo "$CYCLONEDDS_HOME"
ls "$CYCLONEDDS_HOME/lib"
```

至少应该能看到 `libddsc.so`。

### 3.3 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

---

## 4. 拉代码与 Python 环境

```bash
cd ~/code/postman
git clone <FSMDeploy_G1 remote URL> FSMDeploy_G1
cd FSMDeploy_G1

mkdir -p external
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git \
  external/unitree_sdk2_python

uv sync
```

仓库的 `pyproject.toml` 会以 editable 方式安装
`external/unitree_sdk2_python`。如果这个目录不存在,`unitree_sdk2py` 很容易在
运行时缺 message package。

环境 smoke test:

```bash
uv run python - <<'PY'
import mujoco
import onnxruntime
import torch
import unitree_sdk2py
from common.ctrlcomp import StateAndCmd, PolicyOutput
from FSM.FSM import FSM

fsm = FSM(StateAndCmd(29), PolicyOutput(29))
print("deploy env ok")
PY
```

期望能看到所有 policy 初始化,最后打印 `deploy env ok`。

---

## 5. Artifact 与 motion 检查

真机前不要只相信 git 分支名,要检查 runtime 文件本身。

### 5.1 检查模型文件

```bash
ls -lh policy/box_handoff_stand/model/policy.onnx
ls -lh policy/box_hold_stand/model/policy.onnx
ls -lh policy/dual_agent_tracking/model/dual_agent_combined.onnx
ls -lh policy/dual_agent_run_tracking/model/dual_agent_combined.onnx
ls -lh policy/dual_agent_jump_tracking/model/dual_agent_combined.onnx
```

### 5.2 检查三条 selected clips

```bash
python - <<'PY'
from pathlib import Path
import hashlib
import numpy as np

items = [
    ("walk", Path("policy/dual_agent_tracking/motion/walk_tracking_ref.npz")),
    ("run", Path("policy/dual_agent_run_tracking/motion/run_tracking_ref.npz")),
    ("jump", Path("policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz")),
]

for name, path in items:
    data = np.load(path, allow_pickle=True)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    frames = len(data["lower_joint_pos"])
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    root = data["torso_pos_w"]
    xy = root[:, :2]
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    span = xy.max(axis=0) - xy.min(axis=0)
    path_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    print(name, path)
    print("  frames", frames, "duration", frames / fps, "sha256", sha)
    print("  net_xy %.3fm, bbox %.3fm x %.3fm, path %.3fm" %
          (net, span[0], span[1], path_len))
PY
```

期望结果应和本文开头的表一致。

### 5.3 ONNX 输入维度检查

```bash
uv run python - <<'PY'
import numpy as np
import onnxruntime as ort

for model in [
    "policy/dual_agent_tracking/model/dual_agent_combined.onnx",
    "policy/dual_agent_run_tracking/model/dual_agent_combined.onnx",
    "policy/dual_agent_jump_tracking/model/dual_agent_combined.onnx",
]:
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    inputs = {i.name: i.shape[-1] for i in sess.get_inputs()}
    print(model, inputs)
    out = sess.run(
        ["actions"],
        {
            "upper_obs": np.zeros((1, inputs["upper_obs"]), dtype=np.float32),
            "lower_obs": np.zeros((1, inputs["lower_obs"]), dtype=np.float32),
        },
    )[0]
    print("  output", out.shape, "norm", float(np.linalg.norm(out)))
PY
```

期望三条 tracking policy 都是:

```text
upper_obs: 96
lower_obs: 109
actions: (1, 29)
```

---

## 6. real.yaml 配置

编辑:

```bash
deploy_real/config/real.yaml
```

### 6.1 网卡

```yaml
net: enp2s0
```

用下面命令查部署机实际有线网卡:

```bash
ip link
```

常见名字是 `enp2s0`、`enp3s0`、`eno1`。这里必须和真机有线连接的网卡一致。

### 6.2 控制周期

```yaml
control_dt: 0.02
```

三条 motion 都是 50Hz,和 `0.02s` 控制周期匹配。不要在真机前临时改成别的值。

### 6.3 技能门控

当前三条 selected clips 的 deploy profile:

```yaml
enable_box_handoff_stand: true
enable_box_hold_stand: true
enable_walk_tracking: true
enable_run_tracking: true
enable_jump_tracking: true
enable_dance_tracking: false
```

如果只做 walk 首次硬件复现,建议临时关闭 run / jump:

```yaml
enable_run_tracking: false
enable_jump_tracking: false
```

关闭后按 `B+L1` / `X+L1` 只会在控制台打印 blocked,不会进入策略。

### 6.4 日志

```yaml
log_real_run: true
real_log_dir: "logs/real_deploy"
real_log_interval: 1
```

每次运行会生成:

```text
logs/real_deploy/real_deploy_YYYYmmdd_HHMMSS.csv
```

CSV 字段包括 FSM 状态、最近一次命令、loop time、按钮状态、实际 q/dq、发送的
q_cmd、kp、kd。第一次真机实验必须保留日志。

### 6.5 target limiter

```yaml
max_target_delta: 0.0
```

`0.0` 表示不限制 policy 输出。如果切换瞬间非常硬,可以设置一个小值临时试验,
例如 `0.03` rad/tick,但这会改变控制器行为,必须重新走安全梯子。

---

## 7. sim2sim 最终预检

真机前先跑一次 MuJoCo keyboard 入口:

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1
uv run python deploy_mujoco/deploy_mujoco_keyboard_input.py
```

walk 预检:

```text
start
a+r1
b+r1
x+r1
a+l1
exit
```

run 预检:

```text
start
a+r1
b+r1
x+r1
b+l1
exit
```

jump 预检:

```text
start
a+r1
b+r1
x+r1
x+l1
exit
```

观察重点:

- `BoxHandoffStand` 不应该 spawn 箱子。
- `BoxHoldStand` ramp 结束后 MuJoCo 会把箱子放到手中。
- 切到 tracking 后 motion clock 从第 0 帧开始。
- 切回 `a+r1` 后应该能离开 tracking。
- 没有 non-finite、assert、ONNX dim mismatch。

sim2sim 只验证软件链路和策略形态,不代表真机可以跳过安全梯子。

---

## 8. 真机启动前硬件 checklist

人员:

- 一人只负责手柄,不递箱子。
- 一人负责递箱子和接箱子。
- 一人观察电脑终端和安全绳,条件允许时再加一人看场地边界。

机器人:

- G1 已开机并进入可 DDS 控制状态。
- 安全绳 / 吊架已挂好。
- 双脚站在防滑、平整区域。
- 机器人朝向已经按目标技能调整:
  - run: 朝 8m 长边。
  - jump: 横向留 4m 以上空间。
  - walk: 前后、侧向都留余量。

箱子:

- 尺寸和训练 / sim2sim 使用的箱子接近。
- 重量不要临时加重。
- 表面不能太滑。
- 递箱人手不要卡在机器人手和箱子之间。

电脑:

- 有线网卡接机器人。
- `real.yaml` 的 `net` 是实际网卡。
- 没有另一个 `deploy_real.py`、ROS2 或 DDS 进程抢 topic。
- 电源接好,不要电池低电量跑。

---

## 9. 启动真机程序

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1
uv run python deploy_real/deploy_real.py
```

预期输出:

```text
Successfully connected to the robot.
...
initalized all policies!!!
Real deploy log: ...
RealDeploy enabled skill gates:
...
Enter zero torque state.
Waiting for the start signal...
```

如果卡在 lowstate:

- 查 `real.yaml` 的 `net`
- 查网线
- 查机器人当前模式
- 确认没有别的进程占用 DDS topic

程序启动后处于零力矩等待 `Start`。这时不要碰 `A/B/X + L1`。

---

## 10. 手柄按键总表

| 按键 | 真机动作 | 说明 |
|---|---|---|
| `F1` | `PASSIVE` damping | 急停优先使用 |
| `Start` | `POS_RESET` / `FixedPose` | 从 zero torque 进入默认站立 |
| `A+R1` | `LocoMode` | 原始 loco,也用于 tracking 后退出 |
| `B+R1` | `BoxHandoffStand` | 打开/保持递箱手型,无 tracking |
| `X+R1` | `BoxHoldStand` | 夹箱站立,无 tracking |
| `A+L1` | walk tracking | 当前 selected walk clip |
| `B+L1` | run tracking | 当前 selected run clip |
| `X+L1` | jump tracking | 当前 selected jump clip |
| `Y+L1` | dance tracking | 当前 `real.yaml` 关闭 |
| `Select` | 退出 Python | 退出前先 `F1` |

组合键建议按法:先按住 `R1` 或 `L1`,再短按 `A/B/X/Y`,看到终端打印命令后松开。
`deploy_real.py` 对组合键做了 latch,按住不放不会每个 tick 重复触发。

---

## 11. 通用 staged box 操作流程

这套流程适用于 walk / run / jump,区别只在最后一个 L1 组合键。

### 11.1 进入默认站立

按:

```text
Start
```

观察:

- 机器人从 zero torque 进入固定站立。
- 没有异常抖动。

失败处理:

- 姿态不对或脚底打滑,`F1`,重新摆正机器人。

### 11.2 进入原始 loco

按:

```text
A+R1
```

观察:

- 终端打印 `A+R1->LOCO`。
- 机器人能稳定站住。
- 第一次不要推摇杆。

### 11.3 进入递箱姿态

按:

```text
B+R1
```

观察:

- 终端打印 `B+R1->BOX_HANDOFF_STAND`。
- 手臂进入递箱姿态。
- 机器人仍然原地站住。

递箱:

- 递箱人从正前方或侧前方把箱子放到两手之间。
- 不要把箱子猛塞进手腕。
- 手保持在箱子附近,不要马上离开。

### 11.4 进入夹箱站立

按:

```text
X+R1
```

观察 5-10 秒:

- 箱子被夹住。
- 肩、肘、腕没有高频抖动。
- pelvis 没有明显弯腰或侧倒。
- 脚底没有明显滑动。

如果箱子不稳:

```text
A+R1
操作员接回箱子
F1
```

不要直接切 tracking。

### 11.5 切入目标 tracking

walk:

```text
A+L1
```

run:

```text
B+L1
```

jump:

```text
X+L1
```

切入后终端应打印类似:

```text
RealDeploy command: B+L1->DUAL_AGENT_RUN_TRACK
Switched to  dual_agent_run_tracking_mode
DualAgentRunTracking: ramping to default pose over 0.10s ...
DualAgentRunTracking: ramp complete, starting policy inference.
```

注意:

- tracking 的 motion clock 在 ramp 结束后才开始走。
- 第一次每条技能只跑 1-2 秒就 `F1`,验证切换瞬间是否安全。
- run 的 7 秒 reference 净位移约 6.9m,不要一上来完整跑完。
- jump 即便只有 7 秒,也必须先架空验证。

### 11.6 退出

推荐正常退出:

```text
A+R1
操作员接回箱子
F1
Select
```

异常退出:

```text
F1
操作员接回箱子
必要时物理急停
```

---

## 12. 三条技能各自的安全梯子

每条技能单独验证,不要共用结论。

### 12.1 walk tracking 安全梯子

| 阶段 | 设置 | 操作 | 通过标准 |
|---|---|---|---|
| W1 | 架空,不递箱 | `Start -> A+R1 -> B+R1 -> X+R1 -> A+L1 -> F1` | 切换瞬间无大抽动 |
| W2 | 架空,递箱 | 完整 staged box 后 `A+L1` 1-2 秒 | 能夹箱,手臂不抖 |
| W3 | 轻触地,安全绳承重 | `A+L1` 1-2 秒 | 脚底不滑,pelvis 不倒 |
| W4 | 地面,人手扶箱 | `A+L1` 3-5 秒 | 箱子不滑,可 `A+R1` 退出 |
| W5 | 地面完整短片 | 允许跑完 23 秒或提前退出 | 能稳定结束或稳定切回 loco |

### 12.2 run tracking 安全梯子

run 的位移接近 7m,场地必须沿 8m 长边布置。

| 阶段 | 设置 | 操作 | 通过标准 |
|---|---|---|---|
| R1 | 架空,不递箱 | `... -> B+L1 -> F1` | 切换瞬间腿部不飞 |
| R2 | 架空,递箱 | `... -> B+L1` 1 秒 | 手臂能稳住箱子 |
| R3 | 轻触地,安全绳承重 | `B+L1` 0.5-1 秒 | 无前冲失控 |
| R4 | 地面,人手扶箱 | `B+L1` 1-2 秒后 `A+R1` | 能退出,不冲出边界 |
| R5 | 地面短跑 | 分段增加到 3 秒、5 秒、7 秒 | 每次都有足够刹停/接箱空间 |

run 第一次不要完整跑 7 秒。现场应该先确定从切入点到前方边界至少有 7m,
再考虑完整 clip。

### 12.3 jump tracking 安全梯子

jump 的风险主要是垂向冲击、落地姿态和横向位移。

| 阶段 | 设置 | 操作 | 通过标准 |
|---|---|---|---|
| J1 | 架空,不递箱 | `... -> X+L1 -> F1` | 切换瞬间无大幅收腿/甩腿 |
| J2 | 架空,递箱 | `X+L1` 1 秒 | 手臂夹箱稳定 |
| J3 | 安全绳承重,脚轻触地 | `X+L1` 0.5-1 秒 | 脚接触不过度横扫 |
| J4 | 地面,强辅助 | `X+L1` 1-2 秒 | 落地没有跪膝/侧倒趋势 |
| J5 | 地面短片 | 分段增加到 3 秒、5 秒、7 秒 | 能稳定切回 `A+R1` |

jump 不建议在没有吊架或安全绳的情况下第一次尝试。

---

## 13. 日志检查

运行结束后找最新 CSV:

```bash
ls -lt logs/real_deploy | head
```

快速看状态切换:

```bash
python - <<'PY'
import csv
from pathlib import Path

path = sorted(Path("logs/real_deploy").glob("real_deploy_*.csv"))[-1]
print(path)
last = None
with path.open() as f:
    for row in csv.DictReader(f):
        key = (row["fsm_state"], row["last_command"])
        if key != last:
            print(row["time_s"], row["fsm_state"], row["last_command"],
                  "loop", row["loop_time_s"], "overtime", row["overtime_count"])
            last = key
PY
```

看是否有连续超时:

```bash
python - <<'PY'
import csv
from pathlib import Path

path = sorted(Path("logs/real_deploy").glob("real_deploy_*.csv"))[-1]
mx = 0
with path.open() as f:
    for row in csv.DictReader(f):
        mx = max(mx, int(row["overtime_count"]))
print("max overtime_count =", mx)
PY
```

如果 `max overtime_count` 接近或超过 `max_control_over_time`,不要继续硬件实验。

---

## 14. 切换 selected motion

候选 motion bank 在:

```text
reference_motion_bank/selected/
```

当前三条 runtime 已经复制到对应 policy 的 `motion/` 目录。以后如果又筛选了
新的片段,规则是:

- 文件名写清楚来源、时间窗、目标策略和 runtime 文件名。
- 先放到 `reference_motion_bank/selected/`。
- 再复制到对应 runtime 路径。
- 更新 `reference_motion_bank/selected/README.md`。
- 真机前重新做 sim2sim 预检。

当前把 bank 里的三条 selected clips 复制到 runtime 的命令:

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1

cp reference_motion_bank/selected/walk4_subject1_1m17_1m40_23s__DualAgentTracking__walk_tracking_ref.npz \
   policy/dual_agent_tracking/motion/walk_tracking_ref.npz

cp reference_motion_bank/selected/run1_subject2_2m15_2m22_7s__DualAgentRunTracking__run_tracking_ref.npz \
   policy/dual_agent_run_tracking/motion/run_tracking_ref.npz

cp reference_motion_bank/selected/jumps1_subject2_first75_0m23_0m30_7s__DualAgentJumpTracking__jump_tracking_ref.npz \
   policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz
```

恢复完整 runtime reference 的备份:

```bash
cp policy/dual_agent_tracking/motion/walk_tracking_ref.full_8195_before_crop_1m17_1m40_2026-05-23.npz \
   policy/dual_agent_tracking/motion/walk_tracking_ref.npz

cp policy/dual_agent_run_tracking/motion/run_tracking_ref.full_11890_before_crop_2m15_2m22_2026-05-23.npz \
   policy/dual_agent_run_tracking/motion/run_tracking_ref.npz

cp policy/dual_agent_jump_tracking/motion/jump_tracking_ref.full_9166_before_crop_1m45_2m08_2026-05-23.npz \
   policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz
```

---

## 15. 常见问题

### 15.1 找不到网卡

报错类似:

```text
Could not find interface enp2s0
```

处理:

```bash
ip link
```

把 `deploy_real/config/real.yaml` 里的 `net` 改成实际有线网卡。

### 15.2 CycloneDDS 动态库找不到

报错类似:

```text
libddsc.so.0: cannot open shared object file
```

检查:

```bash
echo "$CYCLONEDDS_HOME"
echo "$LD_LIBRARY_PATH"
ls "$CYCLONEDDS_HOME/lib"
```

缺失就重新执行本文 3.2。

### 15.3 Unitree SDK import 缺 message package

检查:

```bash
ls external/unitree_sdk2_python/unitree_sdk2py
uv sync
```

不要只 `pip install unitree_sdk2py`,要使用仓库里的 editable SDK。

### 15.4 L1 组合键被 blocked

控制台如果打印:

```text
RealDeploy blocked B+L1->DUAL_AGENT_RUN_TRACK: enable_run_tracking=false
```

说明 `real.yaml` 里对应门控是 false。确认你确实要测试该技能后再打开。

### 15.5 切 tracking 瞬间弯腰或软一下

先 `F1`,不要连续尝试。排查:

- 是否从 `BoxHoldStand` 稳定状态切入。
- 箱子是否太重或太滑。
- 机器人脚底是否站偏。
- 当前 runtime motion 是否是你想测的短片。
- CSV 中切换前后 `q_cmd` 是否有单关节大跳。
- 是否临时设置了 `max_target_delta` 并改变了策略动态。

### 15.6 控制循环超时

如果连续打印:

```text
control loop over time.
```

程序会在连续超时达到 `max_control_over_time` 后发送 damping。继续实验前排查:

- CPU 是否被其他进程占满。
- 是否开着多个 MuJoCo viewer。
- 是否同时运行两个 deploy 进程。
- 部署机是否在省电模式。

---

## 16. 实验记录建议

每次真机实验至少记录:

- 日期、场地、操作者
- git commit 或 `git status --short`
- `real.yaml` 内容
- 三条 runtime motion SHA256
- 实际按键序列
- 是否架空 / 安全绳承重 / 裸跑
- 箱子尺寸和重量
- 终端输出和 CSV 日志路径
- 成功、失败和中断原因

推荐在实验记录里写成:

```text
2026-xx-xx
profile: selected walk/run/jump clips
flow: Start -> A+R1 -> B+R1 -> X+R1 -> A+L1
result: ...
log: logs/real_deploy/real_deploy_...
video: ...
notes: ...
```

这样后续写论文或排查 sim2real 差异时,不会丢掉关键上下文。
