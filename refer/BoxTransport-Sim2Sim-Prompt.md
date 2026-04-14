# Box Transport ONNX sim2sim 适配 Prompt

将以下内容提供给 sim2sim 适配开发/Agent，用于实现 Box Transport Velocity ONNX 在 MuJoCo 中的部署。

---

## Prompt for sim2sim Adapter

你需要在 MuJoCo 中部署一个 Box Transport Velocity 策略 ONNX。该策略让人形机器人 G1 抱着箱子并跟踪速度命令行走。

### 1. ONNX 基本信息

- **模型类型**: `box_transport_velocity`（单网络全身体控制）
- **输入**: `obs`，形状 `[1, 480]`，float32
- **输出**: `actions`，形状 `[1, 29]`，float32，**已是 Mujoco 关节顺序**
- **控制频率**: 50Hz（control_dt=0.02s），decimation=4（每 4 个仿真步执行一次策略）

### 2. Isaac 与 MuJoCo 关节顺序差异（重要）

Isaac Sim 和 MuJoCo 的关节顺序不同：

- **Isaac**：按关节类型分组（pitch/roll/yaw），左右交替
- **MuJoCo**：按身体部位分组（左腿→右腿→腰→左臂→右臂）

**ONNX 输出已是 Mujoco 顺序**，可直接写入 MuJoCo 的 qpos/actuator。但**观测中的关节相关项必须是 Isaac 顺序**。

### 3. 观测构建（480 维）

观测 = 5 步历史 × 96 维/步。每步 96 维包含：

| 顺序 | 项 | 维度 | 说明 |
|------|-----|------|------|
| 1 | base_ang_vel | 3 | 基座角速度 (body frame)，×0.2，clip(-100,100) |
| 2 | projected_gravity | 3 | 重力在 body frame 的投影（归一化） |
| 3 | velocity_commands | 3 | [vx, vy, ω] 速度命令 |
| 4 | joint_pos_rel | 29 | (当前关节角 - 默认角)，**Isaac 顺序**，clip(-100,100) |
| 5 | joint_vel_rel | 29 | 关节角速度，**Isaac 顺序**，×0.05，clip(-100,100) |
| 6 | last_action | 29 | 上一步策略输出，**Isaac 顺序**，clip(-12,12) |

**历史**：`[t-4, t-3, t-2, t-1, t]` 拼接，共 480 维。初始化时用首帧填充至 5 步。

**关节顺序转换**：从 MuJoCo 读取的 joint_pos、joint_vel 是 Mujoco 顺序；last_action 来自策略输出，也是 Mujoco 顺序。观测要求 **Isaac 顺序**，需用 `isaac_to_mujoco_idx` 重排：

```python
# 从 ONNX metadata 读取 isaac_to_mujoco_idx (或 joint_names 自建)
# joint_pos_isaac[j] = joint_pos_mujoco[isaac_to_mujoco_idx[j]]
joint_pos_rel_isaac = (joint_pos_mujoco - default_pos_mujoco)[isaac_to_mujoco_idx]
joint_vel_rel_isaac = joint_vel_mujoco[isaac_to_mujoco_idx] * 0.05
last_action_isaac = last_action_mujoco[isaac_to_mujoco_idx]  # last_action 来自上一步 ONNX 输出
```

### 4. 动作后处理

```python
# 从 metadata 读取
action_scale = 0.25  # float(metadata["action_scale"])
default_joint_pos_mujoco = np.array(metadata["default_joint_pos_mujoco"].split(","), dtype=np.float32)

# ONNX 输出 actions 已是 Mujoco 顺序
target_joint_pos = default_joint_pos_mujoco + actions * action_scale
# target_joint_pos 可直接用于 MuJoCo actuator
```

### 5. 从 ONNX 元数据读取配置

```python
import onnx
model = onnx.load("box_transport_velocity.onnx")
meta = {p.key: p.value for p in model.metadata_props}

# 必须使用的字段:
# - joint_names: Mujoco 顺序关节名 (用于 policy_joint_names / act_maps)
# - isaac_to_mujoco_idx: 观测重排，Mujoco->Isaac 查表 (csv 转 int array)
# - default_joint_pos_mujoco: 默认关节角 Mujoco 顺序 (csv 转 float array)
# - action_scale: "0.25"
# - control_dt, control_decimation
```

### 6. policy_joint_names 配置

sim2sim 中 `policy_joint_names` 必须使用 **Mujoco 顺序**，与 ONNX 输出一致：

```python
policy_joint_names = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
```

### 7. 参考文档

详细说明见 `docs/DualAgent-ONNX-Export-Guide.md` 第 11 节「Box Transport Velocity」。
