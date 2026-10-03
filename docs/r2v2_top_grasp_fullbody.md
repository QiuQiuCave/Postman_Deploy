# R2V2 全身可乐上端抓取：接口与验证边界

2026-09-23 根据用户选择新增[正手抓取配置](r2v2_front_grasp_fullbody.md)，取消大幅翻腕，原上端配置保留。本页以下为上端抓型及其历史验证记录；正手当前只完成静态检查，冻结策略文件缺失尚未运行动态实验。

本入口将已记录的纯手部上端抓型迁移到新完整机器人，先做 AIR 空手路径验证，只有同场景、同策略、同资产的完整 AIR 验收通过才能进入 CONTACT。它不是纯手部视频换一个全身外观，也不会回放机器人关节轨迹。

## 实现

- `common/r2v2_top_grasp_path.py` 与 `deploy_mujoco/config/r2v2_top_grasp_calibration.json`：从成功纯手部实验提取腕目标相对于**初始物体**的位姿；包含来源文件和标定 SHA256，不包含机器人 qpos 回放。
- `common/r2v2_top_grasp_fullbody.py`：自由浮基座、原 28 维身体策略、独立左右手指控制，以及实际末端/接触判据驱动的状态机。
- `deploy_mujoco/r2v2_top_grasp_fullbody.py`：正侧面全身与固定斜近景双画面，失败也输出视频、状态、目标和测量日志。
- `deploy_mujoco/config/r2v2_top_grasp_fullbody.json`：台高 0.80 m、罐中心 XY=(0.36, 0.18) m、绕罐轴整体旋转 180° 的候选。
- `deploy_mujoco/config/r2v2_top_grasp_fullbody_yaw90.json`：同台高，罐中心 XY=(0.32, 0.18) m、整体旋转 +90° 的候选；搬移朝左外侧。

两个配置中的 `reach_config` / `parity_report` 是本机数据盘上的冻结文件。迁移到其他机器必须带上相应模型、ONNX、配置和数值对照证据，并更新路径；不能拿其他策略的报告替代。

冻结来源为 `R2V2-Reach-CrateWristPayload-v2-28DoF` 的 `model_5999.pt`，使用已经数值对照通过的 ONNX，不导出新网络、不改变训练仓库：

- checkpoint SHA256：`bb82a46b4c4ba62cec8ec51116298b62532b147d2b3f03c5e3d8e8ab272c79c6`
- ONNX SHA256：`a4200572a78c46963105f00503c75e52110c8fc47e4f356315e19cc37952155e`
- endpoint：`wrist_world_v2`，即 `left_hand_roll_link` / `right_hand_roll_link` 的原点世界位姿，无旧 TCP 偏移。

物体沿用已经独立抓取成功的直径 40 mm、高 120 mm、质量 100 g 仿真基线，带可乐外观；不表示真实装满 330 mL 可乐的规格。机器人原始限位、惯量、碰撞、手指 mimic 和执行器上限保留。身体策略 50 Hz、手指参考 100 Hz、接触仿真与力矩控制 1 kHz。

## 路径与两种模式

公共准备：`RESET_SETTLE → HOME → STAND(20 s) → SAFE_OUT → TURN_WRIST`。站稳后锁定右腕实际世界位姿。左腕先外展到世界 (0.16, 0.36, max(1.10, hover_z)) m，再在该位置转腕。

AIR：`HOVER → APPROACH → GRASP → PROBE → UPRIGHT → LIFT → TRANSLATE → PLACE → RETREAT`。

- 台面和罐子为静态、无接触的线框占位；机器人始终自由站立，身体自碰撞和地面接触真实存在；双手一直为张手指令 0。
- 一个独立模型副本仅做几何相交诊断，检查真实机器人与虚拟台面的重叠；不会将该副本的力或状态写回动力学仿真。
- 目标每段只设置一次，由原策略的参考轨迹接口处理，不能每帧重置轨迹。
- 左腕实际误差 <5 mm、<3°、线速度 <2 cm/s，右腕 <2 cm、<5°，持续 0.3 s 才通过运动段；HOME 单独采用已有待机容差。单段最长 10 s。
- 配置默认允许 **AIR 精度不达标后继续诊断下一段**，但该段及整次验收永久标为失败，绝不能据此开启 CONTACT。关节过度越限、身体失稳、非脚部着地、深度自碰撞或数值异常仍立即结束。

CONTACT 在上述接近路径后增加实际闭合、试提、握持复验、扶正、抬升、搬移、触台释放与撤离。闭合不是抓到物体：必须同时满足对置接触、离台、承重与腕物相对位姿稳定性。实际右腕持续偏离锁定世界位姿 0.3 s 会失败。搬运失败不在空中自动松手。

真实 UPRIGHT/PLACE 使用当时测得的腕—物体变换计算目标，不能盲目播放纯手部实验的扶正和放置目标。计算后的目标若偏离已筛选的名义 AIR 目标超过 2 cm 或 5°，停止并要求重新筛选。AIR 名义路径不会覆盖任意抓歪后的姿态，也不能证明受载握持稳定。

所有路径的初始锚点固定：

`T_world_wrist_goal = T_world_object_initial × T_object_initial_wrist_goal`

不能每帧乘实时移动物体位姿，否则会变成不断追逐物体并改变实际搬运距离。接近由记录抓取位姿上方 3 cm 的点过渡，hover 高 10 cm；名义总抬升 8 cm、平移 10 cm。这一版不含货箱，也未声称这段抬升足以越过 16 cm 箱沿。

## 运行

在仓库目录使用已配置环境，输出放数据盘：

```bash
.venv-r2v2/bin/python deploy_mujoco/r2v2_top_grasp_fullbody.py \
  --config deploy_mujoco/config/r2v2_top_grasp_fullbody_yaw90.json \
  --mode air \
  --output /root/autodl-tmp/Postman_Deploy/top_grasp_fullbody_20260915/air_yaw90
```

输出目录必须为空或尚不存在；重跑使用新的目录后缀。`--no-video` 只跳过渲染，仍执行物理和输出原始日志。CLI 返回 0 表示录制过程正常结束，**不是抓取或验收成功**，应检查 `report.json` 的 `air_passed` / `success` / `failure`。

只有 AIR 全部通过后才可将其 `report.json` 路径填写到同场景配置的 `air_evidence`，再以 `--mode contact` 启动。证据绑定完整阶段顺序、实测终态与停稳时长、碰撞结果、配置内容、策略、控制代码、模型资产、手部配置及 MuJoCo 版本。当前失败报告不能用于 CONTACT。

产物包括 `top_grasp_fullbody.mp4`、`report.json`、100 Hz `trace.json`、`targets.json`、`transitions.json`、阶段截图。细绿/蓝轴是目标，短粗 RGB 轴是实际手腕。视频最后 2 s 定格明确标注，不计入仿真或稳定时间。

## 当前结果与下一步

正式输出位于 `/root/autodl-tmp/Postman_Deploy/top_grasp_fullbody_20260915/`。两方位的正式 AIR 录像均完成且全视频解码通过。两组 HOME 与 20 s STAND 通过；SAFE_OUT 10 s 结束时左腕位置误差为 8.175 mm（姿态 0.402°），未通过 5 mm 门槛，因此即使继续诊断也不再可能获得整次通过。随后均在 TURN_WRIST 触发停止，没有启动真实 CONTACT，也未开启训练。

| 记录目录 | 仿真结束时间 | 左腕终态误差 | 右腕终态误差 | 停止原因 |
| --- | --- | --- | --- | --- |
| `air_yaw180` | 35.097 s | 94.0 mm / 84.7° | 109.7 mm / 32.9° | 左臂 yaw 实际 -2.02223 rad，比真实下限 -1.97222 rad 超出 0.05001 rad，触发越限容差 |
| `air_yaw90` | 36.210 s | 196.3 mm / 58.3° | 175.9 mm / 31.2° | 身体倾角 35.072°，足部相对初始位置最大偏移 288.4 mm |

两个视频分别约 37.13 s / 38.27 s，含末尾 2 s 标记定格；全程双手指令均为 0。上表是失败时的实测误差，不是前述训练覆盖差角，也不能称为已完成抓取的误差。`peaks` 是检查时累计值，停止前的最终同步可能更晚，分析终态请读 `final_metrics`，分析完整采样序列请读 `trace.json`。

验证：全量 R2V2 回归在当时收集的 1083 项全部通过；之后补充的保护测试与本次路径/渲染测试共 191 项另行通过。模型场景测试逐项比较了原机器人惯量、限位、碰撞、mimic 与全部 40 个执行器映射；这些单元测试不替代真实抓握验收。

离线训练覆盖核对表明，冻结策略的本轮后训练是抬箱路径附近的局部/FULL 路径采样和 HOME 附近复习，并不是任意 SO(3) 朝向采样。与归档左腕路径目标逐个比较旋转测地距离，上端 GRASP 在 yaw 180° 时最近差角约 135.71°，yaw +90° 时仍约 84.95°。这些数字是对归档目标的最近距离，不是实际 Reach 误差或形式化可达性证明。当前抓型要求大幅翻转手腕局部 Z 轴，单纯绕罐轴换方位不能消除这个差距。

覆盖核对输入来自冻结训练目录 `/root/autodl-tmp/AMO_R2/logs/rsl_rl/r2v2_crate_wrist_payload_v2_28dof/2026-09-12_payload-v2-lift4-from4000/` 的 `params/env.yaml`、`code_snapshot/` 和 `path_snapshot/planned_path.npz`。比较方法是对其中 4682 个左腕目标旋转逐个求 SO(3) 测地距离的最小值；没有对所有插值、随机扰动及祖先策略见过的姿态求精确支持集距离。FULL 仍是路径采样；REHEARSAL 在 HOME 附近各轴 ±10°；持箱阶段额外扰动仅约 ±2 mm / 各轴 ±0.5°。

额外做了有界的离线静态 IK，不运行策略、不执行物理步、不将解用于录像或控制。在真实模型、右腕 HOME、双脚初始锚点约束下，yaw +90° 的 **GRASP 单点**找到了保留至少 2° 真实关节限位裕量的候选：左腕误差约 0.001 mm / 0.002°，基座倾角 4.28°，无桌面接触和新增身体间接触（仅保留源手掌—拇指约 0.171 mm 浅接触）。质心检查仅为足部 AABB 内，不是完整动态稳定性证明。

相同搜索裕量的 **TURN_WRIST** 虽能达到约 0.235 mm / 0.470°，找到的候选存在腰部—左肩约 2.794 mm 穿透，本轮未找到满足无明显自碰撞要求的过渡解。有限次数求解失败不证明不存在解；GRASP 单点有解也不证明从站姿存在无碰撞路径或现有策略可以跟踪。诊断归档 `offline_ik_audit.json` 与真实 AIR 轨迹分开，禁止将静态 witness 解释为 CONTACT 验收。

接下来需要先验证翻腕及上端抓取的全身运动学/碰撞可行性，再选择自然的抓取方位和过渡路径；若可行但策略无法跟踪，再由用户确认对这些路径及邻近姿态后训练。不得将当前纯手部成功或空手失败包装成全身抓取成功。
