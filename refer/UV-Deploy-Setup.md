# UV 部署环境配置(新机器)

用 `uv` 管理部署环境,替代之前又大又杂的 `unitree_sim2sim` conda env。

- **Python**: 3.10(锁在 `>=3.10,<3.12`,uv 会自己拉 python-build-standalone)
- **核心包**:`mujoco==3.3.7`、`onnx==1.19.1`、`onnxruntime==1.23.2`、
  `torch==2.7.0+cpu`、`numpy<2`、`pygame`、`pyyaml`
- **Unitree SDK**:editable 从 `external/unitree_sdk2_python/` 本地路径装
  (upstream 的 setup.py 有 bug,非 editable 装会缺 `b2/g1/h1/comm` 子包)
- **不装**:CUDA torch、torchvision、torchaudio、onnx exporters、conda

所有依赖已经在 `pyproject.toml` + `uv.lock` 里冻结,跨机器字节一致。

---

## 1. 一次性系统依赖(Ubuntu)

### 1.1 apt 包

```bash
# mujoco viewer 依赖(纯跑 deploy_real 可跳过)
sudo apt install -y libglfw3 libglew2.2 libegl1

# OpenCV 从 unitree_sdk2py 带进来的运行时依赖
sudo apt install -y libgl1 libglib2.0-0

# build cyclonedds 需要
sudo apt install -y cmake build-essential git
```

### 1.2 cyclonedds native 库(必装,uv sync 前)

`unitree_sdk2py` 依赖 `cyclonedds==0.10.2` Python binding,这个 Python 包
在装的时候要找 C++ 库 `libddsc.so`。PyPI 上**没有预编译 wheel**,必须本地
build + install 后再 `uv sync`,否则 uv 会在 `Failed to build cyclonedds==0.10.2
— Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH`
处挂掉。

```bash
cd ~
git clone --depth 1 --branch releases/0.10.x \
  https://github.com/eclipse-cyclonedds/cyclonedds.git
cd cyclonedds
mkdir -p build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$HOME/cyclonedds/install" -DBUILD_EXAMPLES=OFF
cmake --build . --target install --parallel $(nproc)

# 持久化到 shell:build 时读 CYCLONEDDS_HOME,运行时读 LD_LIBRARY_PATH
echo 'export CYCLONEDDS_HOME="$HOME/cyclonedds/install"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:${LD_LIBRARY_PATH:-}"' >> ~/.bashrc
source ~/.bashrc
```

Build ~3~5 分钟。装到 `$HOME/cyclonedds/install/` 不需要 sudo。

> **能不能装 /usr/local 拿 sudo 省事?** 可以,但那台机以后别装别的版本
> cyclonedds,否则冲突。用户目录 + CYCLONEDDS_HOME 更干净。

---

## 2. 装 uv

[官方安装脚本](https://docs.astral.sh/uv/getting-started/installation/),不需要 sudo:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# 执行完按提示 source 一下(通常是 source $HOME/.local/bin/env)
uv --version   # 确认能用,这套验证过 0.11.x
```

---

## 3. 拉代码 + 拉 Unitree SDK

```bash
# 主仓库
git clone <你的 FSMDeploy_G1 remote URL> FSMDeploy_G1
cd FSMDeploy_G1

# Unitree SDK 装在仓库内 external/ (gitignored)
mkdir -p external
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git \
  external/unitree_sdk2_python
```

> **为什么要放 `external/`**:upstream `unitree_sdk2_python` 的 `__init__.py`
> 里 `from . import b2, g1, h1, comm`,但这几个子目录**没有 `__init__.py`**,
> 用 `find_packages()` 装普通 wheel 会把它们漏掉,运行时报"circular import"。
> 只有 `pip install -e` / `uv sync --editable path=...` 才能绕开 —— 因为
> editable 不打 wheel,直接暴露源码树。`pyproject.toml` 里已经配好这条路径,
> 别改。

---

## 4. 同步环境

```bash
uv sync
```

这一步 uv 会:

1. 自动拉 Python 3.10 (如果系统没有对应版本)
2. 在 `.venv/` 建虚拟环境
3. 按 `uv.lock` 下载并装全部 42 个 pinned 包
4. 把 `external/unitree_sdk2_python/` 挂成 editable

第一次完整装大概 1~2 分钟(看网速);以后 `uv sync` 检测到 lock 没变几秒就过。

**验证**:

```bash
uv run python -c "
import mujoco, onnxruntime, torch, unitree_sdk2py
from common.ctrlcomp import StateAndCmd, PolicyOutput
from FSM.FSM import FSM
fsm = FSM(StateAndCmd(29), PolicyOutput(29))
print('all good')
"
```

能看到每个策略 "initializing ..." 并最终打出 `initalized all policies!!!`,
说明 ONNX / TorchScript / MuJoCo / SDK 四条路径都通了。

---

## 5. 放策略文件

模型和 motion 不入 git,每个策略自己 `model/` + `motion/` 目录下的
artifact 要手动拷过来。以 `DualAgentTracking` 为例(就是你刚刚 sim2sim
验证通过的那份):

```bash
# 从训练机/已验证机 scp 过来
scp <source_machine>:~/code/postman/FSMDeploy_G1/policy/dual_agent_tracking/model/dual_agent_combined.onnx \
    policy/dual_agent_tracking/model/

scp <source_machine>:~/code/postman/FSMDeploy_G1/policy/dual_agent_tracking/motion/walk_tracking_ref.npz \
    policy/dual_agent_tracking/motion/
```

其他策略(LocoMode `.pt`、Dance `.onnx` 等)按需拷。

---

## 6. 跑

所有 `python` 都换成 `uv run python`,或者先 `source .venv/bin/activate`
激活再跑普通命令。

```bash
# MuJoCo + 键盘输入(不需要手柄)
uv run python deploy_mujoco/deploy_mujoco_keyboard_input.py

# MuJoCo + Xbox 手柄
uv run python deploy_mujoco/deploy_mujoco.py

# 真机(需要插电话线 DDS 通 + Unitree G1 开机)
uv run python deploy_real/deploy_real.py
```

键位和状态机逻辑跟之前完全一样,见 `README.md` / `CLAUDE.md` Safety 段。
DualAgentTracking 真机流程单独见 `refer/DualAgentTracking-Sim2Real-Guide.md`。

---

## 7. 常用 uv 操作

| 场景 | 命令 |
|---|---|
| 装/同步环境(装完 `pyproject.toml` 改了也跑这条) | `uv sync` |
| 只加装一个包不 lock | `uv pip install <pkg>` |
| 加一个永久依赖(会改 pyproject + lock) | `uv add <pkg>` |
| 升级某个包到新版本(改 lock) | `uv lock --upgrade-package <pkg>` |
| 看当前环境装了啥 | `uv pip list` |
| 彻底重装 | `rm -rf .venv && uv sync` |
| 不激活 venv 直接跑 | `uv run python xxx.py` |
| 激活 venv 当普通 python 用 | `source .venv/bin/activate` |

---

## 8. 排坑

- **`Failed to build cyclonedds==0.10.2 — Could not locate cyclonedds`**
  → §1.2 没做,native 库没 build。按 §1.2 从源码 build cyclonedds 并
  `export CYCLONEDDS_HOME`,再重跑 `uv sync`。

- **`ImportError: cannot import name 'b2' from ... unitree_sdk2py`**
  → `external/unitree_sdk2_python/` 没 clone,或 clone 了但 `uv sync`
  没跑。检查 `ls external/unitree_sdk2_python/unitree_sdk2py/b2/` 应该看到
  `__init__.py` 等文件;再 `uv sync` 重新装。

- **运行时 `OSError: libddsc.so.0: cannot open shared object file`**
  → `LD_LIBRARY_PATH` 没包含 `$CYCLONEDDS_HOME/lib`。检查 §1.2 最后两行
  是否都加到 `~/.bashrc`,然后 `source ~/.bashrc`。

- **`mujoco` import 报 GLFW / OpenGL 错**
  → §1 的系统库没装。纯 headless 推理如果也报,那是 egl 没装,补
  `sudo apt install libegl1`。

- **`torch.cuda.is_available() == True`**
  → 你装成 CUDA torch 了,CPU-only 的锁被绕过。
  `uv sync --reinstall-package torch` 重装;如果还不对,手动
  `rm -rf .venv && uv sync`。pyproject 里已经显式走 `pytorch-cpu` 索引,
  正常不会踩到。

- **Python 版本 uv 找不到**
  → 服务器系统 python 特别老,uv 会自动下载 python-build-standalone。
  如果网络受限,提前 `uv python install 3.10`。

- **权限 / 目录不对**
  → uv 所有东西都写在项目 `.venv/` 里,不污染系统。要彻底清就 `rm -rf
  .venv`,重来。
