<div align="center">
  <h1 align="center">RoboMimic Deploy - 增强版</h1>
  <p align="center">
    <span>🇨🇳 中文</span> | <a href="README.md">🌎 English</a>
  </p>
</div>

<p align="center">
  <strong>基于状态切换机制的宇树G1机器人(29自由度)多策略部署框架，集成了增强的GAE_Mimic动作跟踪能力</strong>
</p>

---

## 当前部署电脑入口

如果是在新的实验部署电脑上由 agent 拉取本仓库并配置环境,请优先阅读:

```text
refer/NewMachine-Agent-Setup-And-SmokeTest.zh.md
```

这份文档覆盖 `uv` 环境、CycloneDDS、Unitree SDK、artifact 验收、MuJoCo
smoke test 和后续 `git pull` 更新流程。当前实际键位以该文档和运行时终端
输出为准。

---

## 📢 关于本仓库

本仓库是基于 [ccrpRepo/RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy) 的优秀工作进行增强的版本。

### 🙏 致谢

我们衷心感谢原作者为机器人社区做出的杰出贡献。他们的工作为人形机器人的多策略部署提供了坚实的基础。

**原始仓库**: [https://github.com/ccrpRepo/RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy)

---

## ✨ 新增功能 - GAE_Mimic 集成

我们将 **GAE_Mimic**（通用动作编码模仿学习）这一前沿的动作跟踪和模仿学习策略集成到了本部署框架中。

### 🎯 GAE_Mimic 特性

- **动作重定向**: 从参考轨迹进行高级动作跟踪
- **通用编码**: 使用基于四元数的变换实现鲁棒的动作表示
- **实时执行**: 通过ONNX模型推理实现高效部署
- **多动作支持**: 兼容多种动作数据集（LAFAN1等）

### 🚀 快速开始 - GAE_Mimic

#### 1. 文件准备

在运行 GAE_Mimic 之前，需要手动放置以下文件：

**ONNX 模型**:
- 位置: `policy/gae_mimic/model/`
- 默认文件名: `policy.onnx`

**动作数据**:
- 位置: `policy/gae_mimic/motion/lafan1/`
- 默认文件名: `walk1_subject2.npz`
- 必需的数据格式:
  - `joint_pos`: (帧数, 关节数)
  - `joint_vel`: (帧数, 关节数)
  - `body_pos_w`: (帧数, 身体数, 3)
  - `body_quat_w`: (帧数, 身体数, 4)
  - `body_lin_vel_w`: (帧数, 身体数, 3)
  - `body_ang_vel_w`: (帧数, 身体数, 3)

#### 2. 触发 GAE_Mimic

**在 MuJoCo 仿真中**:
```bash
python deploy_mujoco/deploy_mujoco.py
# 按 Xbox 手柄的 B + L1
```

**键盘控制（无手柄）**:
```bash
python deploy_mujoco/deploy_mujoco_keyboard_input.py
# 输入命令: b+l1
```

**真机部署**:
```bash
python deploy_real/deploy_real.py
# 按手柄的 B + L1
```

#### 3. 键盘命令参考

| 命令 | 功能 | 说明 |
|------|------|------|
| `l3` | 被动模式 | 阻尼保护 |
| `start` | 位置复位 | 恢复默认姿态 |
| `a+r1` | 运动模式 | 行走模式 |
| `x+r1` | 技能 1 | 舞蹈 |
| `y+r1` | 技能 2 | 功夫 |
| `b+r1` | 技能 3 | 踢腿 |
| `y+l1` | 技能 4 | BeyondMimic |
| `b+l1` | **技能 GAE** | **GAE_Mimic** ⭐ |
| `vel x y z` | 设置速度 | 例如 `vel 0.5 0 0.2` |
| `exit` | 退出程序 | 退出仿真 |

---

## 📚 文档

详细的安装、配置和使用说明，请参考：

- **[中文教程](TUTORIAL_zh.md)** - 完整的安装和操作指南
- **[English Tutorial](TUTORIAL.md)** - Complete setup and operation guide
- **[GAE_Mimic 迁移说明](policy/gae_mimic/MIGRATION_NOTES.md)** - GAE_Mimic 集成的技术细节

---

## 🏗️ 项目结构

```
RoboMimic_Deploy/
├── policy/
│   ├── passive/              # 被动阻尼模式
│   ├── fixedpose/            # 固定位置复位
│   ├── loco_mode/            # 运动策略
│   ├── dance/                # 查尔斯顿舞蹈
│   ├── kungfu/               # 武术动作
│   ├── kungfu2/              # 备选功夫
│   ├── kick/                 # 踢腿动作
│   ├── beyond_mimic/         # BeyondMimic 跟踪
│   └── gae_mimic/            # ⭐ GAE_Mimic 跟踪（新增）
│       ├── config/           # 配置文件
│       ├── model/            # ONNX 模型（用户提供）
│       └── motion/           # 动作数据（用户提供）
├── FSM/                      # 有限状态机控制器
├── deploy_mujoco/            # MuJoCo 仿真部署
├── deploy_real/              # 真机部署
└── common/                   # 共享工具
```

---

## 🛠️ 支持的策略

| 策略名称 | 描述 | 状态 |
|---------|------|------|
| **PassiveMode** | 阻尼保护模式 | ✅ 稳定 |
| **FixedPose** | 位控复位 | ✅ 稳定 |
| **LocoMode** | 稳定行走 | ✅ 稳定 |
| **Dance** | 查尔斯顿舞蹈 | ✅ 真机验证通过 |
| **KungFu** | 武术动作 | ⚠️ 仅限仿真 |
| **KungFu2** | 备选功夫 | ⚠️ 仅限仿真 |
| **Kick** | 踢腿动作 | ⚠️ 仅限仿真 |
| **BeyondMimic** | 动作跟踪 | ⚠️ 实验性 |
| **GAE_Mimic** | 高级动作跟踪 | ⭐ 新增 |

---

## ⚠️ 重要注意事项

### 机器人兼容性
- 本框架设计用于**具有三自由度腰部的宇树G1机器人**
- 如果安装了腰部固定件，请按照官方说明解锁
- **建议拆除手掌**以避免舞蹈动作时的碰撞

### 安全指南

⚠️ **重要警告**: 
- **务必在真机部署前充分进行仿真测试（Sim2Sim）**
- **确保采取适当的安全保护措施**（紧急停止装置、安全吊带、清理工作区域）
- **本代码不提供任何安全保证** - 使用风险自负
- 开发者不对使用本软件造成的任何损坏或伤害承担责任

1. **先在仿真中测试**，再部署到真机
2. 按 `F1` 或 `Select` 紧急停止（被动模式）
3. 查尔斯顿舞蹈（R1+X）是真机上最稳定的策略
4. 其他动作**建议仅在仿真中使用**

### 已知限制
- 不直接兼容 Orin NX 平台（请使用 unitree_sdk2 + ROS 方案）
- Mimic 策略在复杂地形上可能无法保证 100% 成功率
- 舞蹈开始/结束时可能需要人工稳定

---

## 🔧 安装

### 快速开始

```bash
# 创建虚拟环境
conda create -n robomimic python=3.8
conda activate robomimic

# 安装 PyTorch
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# 克隆仓库
git clone https://github.com/YOUR_USERNAME/RoboMimic_Deploy.git
cd RoboMimic_Deploy

# 安装依赖
pip install numpy==1.20.0
pip install onnx onnxruntime

# 安装 Unitree SDK
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

详细说明请参见 [TUTORIAL_zh.md](TUTORIAL_zh.md)。

---

## 🎥 视频教程

[观看视频教程](https://www.bilibili.com/video/BV1VTKHzSE6C/?vd_source=713b35f59bdf42930757aea07a44e7cb#reply114743994027967)

---

## 📝 许可证

本项目保持与原始仓库相同的许可证。

---

## 🤝 贡献

我们欢迎贡献！如果您发现问题或想添加新功能，请：

1. Fork 本仓库
2. 创建您的特性分支
3. 提交 Pull Request

---

## 📧 联系方式

关于以下问题：
- **原始框架**: 请参考 [ccrpRepo/RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy)
- **GAE_Mimic 增强**: 在本仓库中提交 Issue

---

<div align="center">
  <sub>基于 ccrpRepo 的杰出工作用 ❤️ 构建</sub>
</div>
