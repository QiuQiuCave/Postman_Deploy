# WristPath-v1 全身真实接触抓箱实验

## 范围

这是 **MuJoCo 仿真实验，不是实机测试，也不是严格抓取验收**。
使用最新 `model_1099.pt`（本轮 1100 次后训练更新）的 ONNX，按训练的
`wrist_world_v2` 手腕 link 原点接收世界位置和 wxyz 四元数目标。
保留原来的固定手腕和严格门控抓箱基线；新探索入口独立实现。

- 身体策略 50 Hz，独立 0/1 手部控制 100 Hz，真实接触和力矩积分 1 kHz。
- 完整带手模型，28 个身体驱动、12 个手部驱动；10 个手部 mimic 等式。
- 无基座/手腕固定、无 mocap、无辅助力、无 IK 或 qpos 轨迹回放。
- 箱子真实自由运动：左右 26 cm、前后 24 cm、高 16 cm、质量 0.4 kg；孔 12 × 5.5 cm。
- 箱底初始 XY 为 `(0.33, 0)` m；桌面 Z 为 `0.9609189696536513` m。
- 采用已冻结的 closer5/down5、下斜 20° + 内斜 yaw 15° 归档；插入 60 mm，
  拇指初始化 75°，手指闭合 curl 0.8 rad。抬升段期望上移 2 cm、向后 3 cm。
- 归档只驱动手腕目标；其中的箱体期望位姿仅用于测量，绝不改写真实箱体状态。

`common/r2v2_wrist_path_contact.py` 的 NumPy 采样器与训练原生 WristPathData
对齐；同时检查路径 SHA256 与 ONNX 的 `path_trajectory_sha256` 完全一致。
进入实验前，`require_parity()` 绑定 checkpoint、ONNX、部署适配器和数值对照证据。

## 完整测试，而不是碰撞即停止

按照用户要求，准备 → 外展 → 转腕 → 预对齐 → 插入 → 停稳 → 闭合 → 抬升 → 保持，
由固定仿真时钟推进，共 **58.24 s**。不会因碰箱、碰桌、末端误差、抓空、滑移、
箱体倾斜/掉落或软件约束残差阈值而中断，也不会追着被推走的箱子重算路径。

真实 MuJoCo 关节限位、碰撞、mimic 等式和力矩限制没有放宽。
关节限位/mimic/自碰/手箱穿透超阈值写入 `constraint_events`，使严格验收不通过。
仅跌倒/非脚部着地、非有限状态/模拟器警告或检测到意外辅助力/夹具时终止。
该探索模式不能直接作为部署到实机的安全控制器。

`COMPLETE` 只表示流程跑完，手部 `command=1` 只表示执行闭合。
实际拿起必须有双侧手指承重、箱底离台 ≥8 mm、无桌/地支撑、无额外身体托举，
且相对滑移 <15 mm、旋转滑移 <5°、箱体倾斜 ≤8°并持续稳定 2 s。
严格成功还要求手腕 <5 mm/<3°、低速和全程无约束阈值超限。
连续稳定按真实时间差计算，并检查最后一个终止时刻；先拿起再掉落不算最终成功。

## 2026-09-12 实测

产物目录：`/root/autodl-tmp/Postman_Deploy/wrist_path_contact_20260912/`。

- `parity/report.json`：240 帧、2 次 reset 数值对照通过；端到端动作最大差 `1.12e-5`。
- `full_contact_trial/`：最初软件手指越限门控在 0.04 s 截断的原始证据，未删除。
- `full_contact_continue_trial/`：完整 58.24 s 真实接触视频和 100 Hz 轨迹；
  `source_snapshot/` 保存实际执行的控制器及 renderer 版本。
- `final_metrics_recheck/`：补严末时刻/连续时间/穿透成功判据后的无视频复测。

无视频复测与视频实验全部 5826 个记录时刻的身体关节、力矩、策略动作、基座位置、
箱体位姿、离台高度和手腕误差逐值完全相同。判据补严没有改变本次物理轨迹或失败结论。
视频为 1280 × 800、30 fps、1809 帧（60.30 s，含终态静帧），已完整解码检查。

完整视频没有跌倒，但 **没有拿起箱子**：闭合阶段没有持续双侧手指接触；
末态箱子仍触台，桌面向上承重约 3.857 N，而箱重约 3.924 N。
末态双手合计向上作用力约 0.067 N，手指对孔梁的有效向上作用力均为 0。
末态手腕误差左 17.14 mm/14.54°、右 13.62 mm/12.65°。
箱—手腕相对位移变化最大 18.96 mm；由于未建立抓稳基线，这不是已抓住后滑落的证据。

初始 HOME 姿态存在拇指与孔梁重叠，峰值约 12.85 mm；0.04 s 时手指越限残差
0.01003 rad，0.05 s 时恢复到软件阈值内。这段初始接触没有隐藏或剪掉，不能把
其引起的箱体移动当作主动抓取效果。视频末尾 2 s 是明确标注的静帧，不是额外稳定证据。

## 复现

在仓库目录执行，输出目录必须为空或不存在：

```bash
.venv-r2v2/bin/python -u deploy_mujoco/r2v2_wrist_path_contact.py \
  --path-manifest /root/autodl-tmp/AMO_R2/logs/rsl_rl/r2v2_crate_wrist_path_v1_28dof/2026-09-11_path-v1-lift2-rear3-from4502/path_snapshot/manifest.json \
  --reach-config /root/autodl-tmp/Postman_Deploy/wrist_path_contact_20260912/policy/reach_config.yaml \
  --parity-report /root/autodl-tmp/Postman_Deploy/wrist_path_contact_20260912/parity/report.json \
  --output /root/autodl-tmp/Postman_Deploy/wrist_path_contact_new_trial
```

可加 `--no-video` 仅生成真实仿真轨迹、状态切换和指标。
退出码 0 仅表示正常完成运行及产物保存，包括被正确记录的安全终止；应读
`schedule_completed`、`physical_pickup_verified` 和 `strict_success` 分别判断结果。
本实验未改训练权重、未启动训练。
