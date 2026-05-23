# 新部署电脑 Agent 环境配置与基础验收指南

本文给在实验部署电脑上运行的 Claude Code / coding agent 使用。目标是从
GitHub 拉取本仓库,完成部署环境配置,并在不接触真机控制的前提下完成基础功能
验收。通过本文检查后,再由实验人员在新电脑上做 MuJoCo sim2sim,最后才进入
真机 deploy 流程。

本文是新机器入口。不要优先读旧版 README 里的 GAE_Mimic 快速开始,那部分保留
历史信息,键位和当前 dual-agent box workflow 已经不同。

## 0. 当前部署包内容

当前仓库包含已经验证过的 runtime policy artifact。新电脑只需要 pull 仓库,
不用再从训练仓库导出 ONNX。

| 入口 | 策略 | runtime policy | runtime motion | 当前片段 |
|---|---|---|---|---|
| `A+R1` / `a+r1` | loco | `policy/loco_mode/model/policy_29dof.pt` | 无 | 原始 loco |
| `B+R1` / `b+r1` | handoff stand | `policy/box_handoff_stand/model/policy.onnx` | 无 | 张手站立接箱 |
| `X+R1` / `x+r1` | hold stand | `policy/box_hold_stand/model/policy.onnx` | 无 | 夹箱站立 |
| `A+L1` / `a+l1` | walk tracking | `policy/dual_agent_tracking/model/dual_agent_combined.onnx` | `policy/dual_agent_tracking/motion/walk_tracking_ref.npz` | `walk4_subject1 1:17-1:40` |
| `B+L1` / `b+l1` | run tracking | `policy/dual_agent_run_tracking/model/dual_agent_combined.onnx` | `policy/dual_agent_run_tracking/motion/run_tracking_ref.npz` | `run1_subject2 2:15-2:22` |
| `X+L1` / `x+l1` | jump tracking | `policy/dual_agent_jump_tracking/model/dual_agent_combined.onnx` | `policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz` | `jumps1_subject2_first75 0:23-0:30` |
| `Y+L1` / `y+l1` | sprint tracking | `policy/dual_agent_dance_tracking/model/dual_agent_combined.onnx` | `policy/dual_agent_dance_tracking/motion/dance_tracking_ref.npz` | `sprint1_subject2 0:05-0:12` |

注意: `Y+L1` 仍复用历史 `dual_agent_dance_tracking` 目录和 enum,但 active
artifact 已经是 sprint tracking,不是 dance。

## 1. 禁止事项

- 不要直接运行 `deploy_real/deploy_real.py` 做真机实验。
- 不要删除或覆盖 `policy/*/model/` 和 `policy/*/motion/` 下的 runtime 文件。
- 不要把本机生成的 `logs/`、`.venv/`、`external/` 提交回 GitHub。
- 不要自行修改 `deploy_real/config/real.yaml` 的技能门控来开启新技能;真机前由
  实验人员决定。

## 2. 系统依赖

目标系统按 Ubuntu 写,不需要 GPU。MuJoCo viewer 需要图形环境;真机部署不依赖
GPU。

```bash
sudo apt update
sudo apt install -y cmake build-essential git curl
sudo apt install -y libglfw3 libglew2.2 libegl1
sudo apt install -y libgl1 libglib2.0-0
```

## 3. 安装 CycloneDDS native 库

`unitree_sdk2py` 依赖本地 `libddsc.so`。必须先安装 CycloneDDS,再 `uv sync`。

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

echo "$CYCLONEDDS_HOME"
ls "$CYCLONEDDS_HOME/lib"
```

`ls` 至少应该看到 `libddsc.so`。

## 4. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

## 5. 拉仓库和 Unitree SDK

把 `<REPO_URL>` 换成当前 GitHub 仓库地址。

```bash
mkdir -p ~/code/postman
cd ~/code/postman
git clone <REPO_URL> FSMDeploy_G1
cd FSMDeploy_G1

mkdir -p external
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git \
  external/unitree_sdk2_python
```

`external/unitree_sdk2_python` 必须存在。仓库的 `pyproject.toml` 以 editable
方式安装它,绕开 Unitree SDK 上游普通 wheel 会漏包的问题。

## 6. 同步 Python 环境

```bash
cd ~/code/postman/FSMDeploy_G1
uv sync
```

这一步会按 `uv.lock` 创建 `.venv/` 并安装固定版本依赖:

- Python `>=3.10,<3.12`
- `mujoco==3.3.7`
- `onnxruntime==1.23.2`
- CPU-only `torch==2.7.0`
- `numpy<2`
- `unitree-sdk2py` editable from `external/unitree_sdk2_python`

以后 pull 新代码后也可以重复执行:

```bash
git pull --ff-only
uv sync
```

## 7. 一键部署包验收

先跑仓库自带的只读验证脚本:

```bash
cd ~/code/postman/FSMDeploy_G1
uv run python tools/validate_deploy_package.py
```

预期最后输出:

```text
VALIDATION PASSED: deploy package is ready for MuJoCo smoke test.
```

这个脚本会检查:

- `mujoco` / `onnxruntime` / `torch` / `unitree_sdk2py` 等 import
- 必要 `.pt`、`.onnx`、`.npz` 是否存在
- walk/run/jump/sprint 四条 runtime motion 的帧数和 SHA256
- 四个 dual-agent ONNX 是否满足 `upper_obs=96`, `lower_obs=109`,
  `actions=(1, 29)`
- FSM 是否能完整初始化全部 policy

如果这个脚本失败,先不要继续 sim2sim。根据错误补依赖或检查 artifact 是否被
错误覆盖。

## 8. 手动检查当前短 motion

需要人工复核时可运行:

```bash
uv run python - <<'PY'
from pathlib import Path
import hashlib
import numpy as np

items = [
    ("walk", Path("policy/dual_agent_tracking/motion/walk_tracking_ref.npz")),
    ("run", Path("policy/dual_agent_run_tracking/motion/run_tracking_ref.npz")),
    ("jump", Path("policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz")),
    ("sprint_y_l1", Path("policy/dual_agent_dance_tracking/motion/dance_tracking_ref.npz")),
]

for name, path in items:
    data = np.load(path)
    fps = int(np.asarray(data["fps"]).reshape(-1)[0])
    frames = data["lower_joint_pos"].shape[0]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    torso = data["torso_pos_w"]
    xy = torso[:, :2]
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    span = xy.max(axis=0) - xy.min(axis=0)
    path_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    print(name, path)
    print(f"  frames={frames}, duration={frames / fps:.2f}s, fps={fps}, sha256={sha}")
    print(f"  torso net_xy={net:.3f}m, bbox={span[0]:.3f}m x {span[1]:.3f}m, path={path_len:.3f}m")
PY
```

当前期望:

| motion | frames | duration | SHA256 |
|---|---:|---:|---|
| walk | `1150` | `23.00s` | `8c970e646fa59af71dd9e95959d0098496b4af81b08a6bd4ee82028b08dcfd55` |
| run | `350` | `7.00s` | `d1a6c74a8c1ea640882dd6b71fde8b79e2c4ebb8556620aa65956ef93955720d` |
| jump | `350` | `7.00s` | `9df2a6e567626e4939a95a2609f6d22e61bab868fe2d300ed45f5d0209199488` |
| sprint_y_l1 | `350` | `7.00s` | `1f2f10098c2fe0aea93b09afcf2211815e4378f84993ab0feb6a8ae190a6e2db` |

## 9. MuJoCo sim2sim 基础测试

有显示器/远程桌面时运行:

```bash
cd ~/code/postman/FSMDeploy_G1
uv run python deploy_mujoco/deploy_mujoco_keyboard_input.py
```

窗口和终端起来后,在终端输入:

```text
start
a+r1
b+r1
x+r1
a+l1
```

这条链路检查 staged handoff 到 walk tracking。终端应能看到状态切换:

```text
Switched to fixed_pose
Switched to Loco_mode
Switched to box_handoff_stand_mode
Switched to box_hold_stand_mode
Switched to dual_agent_tracking_mode
```

接着可以重启进程,分别测:

```text
start
a+r1
b+l1
```

```text
start
a+r1
x+l1
```

```text
start
a+r1
y+l1
```

`y+l1` 应显示 sprint:

```text
Dual Agent Tracking: sprint (sim2sim, ONNX)
Switched to dual_agent_sprint_tracking_mode
DualAgentSprintTracking: ramp complete, starting policy inference.
```

基础验收只要求:

- 程序能启动 viewer
- 能进入对应 state
- 没有 ONNX 输入维度错误
- 没有 `NaN`、Python traceback 或 viewer 立即崩溃

动作质量和真机可行性由实验人员后续在本机重新 sim2sim 观察。

## 10. 真机前配置检查

不要主动开始真机 deploy。实验人员要求时,先检查:

```bash
ip link
sed -n '1,80p' deploy_real/config/real.yaml
```

`deploy_real/config/real.yaml` 里的:

```yaml
net: enp2s0
```

必须改成连接 G1 的实际有线网卡名。当前真机门控默认:

```yaml
enable_box_handoff_stand: true
enable_box_hold_stand: true
enable_walk_tracking: true
enable_run_tracking: true
enable_jump_tracking: true
enable_dance_tracking: false
```

`enable_dance_tracking=false` 表示 `Y+L1` 真机入口关闭。即使 sim2sim 中
`y+l1` 是 sprint,真机也不要默认开启它。

真机完整操作流程看:

```text
refer/RealDeploy-Setup-And-Operation-Guide.zh.md
```

## 11. 后续更新策略的 pull 流程

训练/主控电脑更新策略后会 push 到 GitHub。部署电脑只做:

```bash
cd ~/code/postman/FSMDeploy_G1
git pull --ff-only
uv sync
uv run python tools/validate_deploy_package.py
```

如果 validate 通过,再运行 MuJoCo keyboard sim2sim。不要在部署电脑上手工替换
policy artifact,除非实验人员明确要求。

## 12. 常见错误

### `Failed to build cyclonedds`

说明第 3 节没做或 `CYCLONEDDS_HOME` 没生效。重新:

```bash
source ~/.bashrc
echo "$CYCLONEDDS_HOME"
ls "$CYCLONEDDS_HOME/lib/libddsc.so"*
uv sync
```

### `ImportError: cannot import name 'b2' from unitree_sdk2py`

说明 `external/unitree_sdk2_python` 不存在,或者没有重新 `uv sync`:

```bash
ls external/unitree_sdk2_python/unitree_sdk2py
uv sync
```

### `OSError: libddsc.so.0: cannot open shared object file`

说明运行时 `LD_LIBRARY_PATH` 没带上 CycloneDDS:

```bash
source ~/.bashrc
echo "$LD_LIBRARY_PATH"
```

### MuJoCo viewer 打不开

先确认有图形环境:

```bash
echo "$DISPLAY"
```

再确认依赖:

```bash
sudo apt install -y libglfw3 libglew2.2 libegl1 libgl1 libglib2.0-0
```

### validate motion hash 不一致

说明 runtime `.npz` 被改过或 pull 的不是预期 commit。执行:

```bash
git status -sb
git pull --ff-only
uv run python tools/validate_deploy_package.py
```

如果仍然不一致,停止并向主控电脑反馈当前 commit:

```bash
git rev-parse HEAD
```
