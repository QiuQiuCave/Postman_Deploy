# R2V2 正手桌面抓取 → 箱口上方 → 松手落箱

本实验使用冻结的全身 Reach 策略和独立二值手指状态机。仅做仿真，不训练、不修改源机器人惯量、关节限位或碰撞，不使用手腕夹具、物体焊接、位姿回放或额外外力。目标是左手从桌面抓起饮料，抬过箱沿，平移至箱口上方后释放；不要求饮料最终直立，也不包含从箱内抓取。

## 冻结输入与代码入口

数据根目录：`/root/autodl-tmp/Postman_Deploy/front_drop_20260927`。

| 输入 | 绝对路径 |
| --- | --- |
| 策略 checkpoint | `/root/autodl-tmp/Postman_Deploy/front_drop_20260927/policy/model_9999.pt` |
| 部署 ONNX | `/root/autodl-tmp/Postman_Deploy/front_drop_20260927/policy/policy_front_manipulation_v1.onnx` |
| 部署控制配置 | `/root/autodl-tmp/Postman_Deploy/front_drop_20260927/policy/reach_config.yaml` |
| 归档轨迹及场景 | `/root/autodl-tmp/Postman_Deploy/front_drop_20260927/policy/path_manifest.json` |
| 接口数值对照 | `/root/autodl-tmp/Postman_Deploy/front_drop_20260927/parity/report.json` |
| 原始训练目录 | `/root/autodl-tmp/AMO_R2/logs/rsl_rl/r2v2_front_manipulation_v1_28dof/2026-09-24_front-drop-v1-from117` |

`model_9999.pt` 是零基 iteration 9999，即完成 **10000 次更新**，不是只训练了 9999 次。

- Checkpoint SHA-256：`9c314ce914e0f0893cd79b355db3de808cedec7e874c548cb577bf43c148edc9`。
- ONNX SHA-256：`a395375a663bd67ea1afca19b5df248186cec9e7a7a47e0368dc0ca319ffcb39`。
- 轨迹规范化内容 SHA-256：`bd0f23607eb4836b833d5341c5c673076c6b508b25c6c4d76f677d2704cf221b`；这是去除 `content_sha256` 后规范 JSON 的哈希，不是文件字节哈希。

代码入口：

- `/root/code/amo/Postman_Deploy/deploy_mujoco/r2v2_front_drop.py`：运行、双视角视频、日志和轨迹导出。
- `/root/code/amo/Postman_Deploy/deploy_mujoco/config/r2v2_front_drop.json`：任务配置。
- `/root/code/amo/Postman_Deploy/common/r2v2_front_drop.py`：目标插值、状态机、验收及安全检查。
- `/root/code/amo/Postman_Deploy/common/r2v2_front_drop_scene.py`：真实场景和独立碰撞诊断模型。
- `/root/code/amo/Postman_Deploy/common/r2v2_reach_policy.py`：观测/历史/ONNX/28 维身体动作映射。
- `/root/code/amo/Postman_Deploy/common/r2v2_hand_control.py`：独立 `0=张开 / 1=闭合` 手指轨迹。
- `/root/code/amo/Postman_Deploy/tools/verify_r2v2_front_drop_policy.py`：训练—部署数值对照。

## 控制与坐标约定

- 任务：`R2V2-Reach-FrontManipulation-v1-28DoF`；范围：`tabletop_to_crate_drop_v1`。
- 末端协议：`wrist_world_v2`；轨迹协议：`front_manipulation_path_v1`。
- 位置为世界坐标米，四元数顺序为 **WXYZ**。末端是真实 `left_hand_roll_link` / `right_hand_roll_link` 原点，对应 `left_wrist` / `right_wrist` site，不是旧偏移 TCP，也不是饮料中心。
- 抓取目标满足 `T_world_wrist = T_world_can × inverse(T_wrist_can)`。归档初始相对位置为 `[0.145, -0.035, 0] m`，相对姿态为单位四元数。
- 左腕执行归档轨迹；右腕目标位置和姿态在世界系固定，身体其他关节仍由策略调节。
- 身体推理 **50 Hz**；手指轨迹 **100 Hz**；力矩施加与接触物理 **1000 Hz**。完整安全/接触指标检查 **100 Hz**，非有限值检查 **1000 Hz**。
- 身体 28 个动作和双手 12 个独立电机通道不重叠。手指保留真实关节和 mimic 约束，沿用 75° 拇指初始化；不通过策略网络控制。
- 目标位置使用与训练相同的三次 smoothstep，姿态使用最短弧 SLERP。只插值目标，不写回身体/物体实际运动状态。阶段末可等待真实到位，不跳过失败条件。

数值对照覆盖两次真实训练环境 reset、480 个观测样本和 24 个观测项各自的 10 帧历史。观测/历史最大绝对误差约 `3.58e-7`；同观测动作误差约 `2.86e-6`；独立重建观测后的动作误差约 `2.09e-5`，均通过既定容差。训练 MuJoCo 3.7.0、部署 MuJoCo 3.3.7；**接口一致不等于接触动力学或抓取成功**，仍需部署 AIR 和 CONTACT 分别验证。

## 场景与物理边界

从归档 `provenance.scene` 读取，不在运行时偷偷更改：

- 桌面高度 `0.98 m`，板中心 `[0.545, 0.255, 0.96] m`，半尺寸 `[0.195, 0.365, 0.02] m`，桌子固定。
- 可乐外观物体中心 `[0.40, 0.13, 1.041] m`，碰撞圆柱 **直径 40 mm × 高 120 mm，质量 100 g**。这是已校准的仿真基线，**不是实际 330 ml 可乐罐规格或满罐质量**；涂装不改变碰撞和质量。
- 开顶货箱外底中心 `[0.45, 0.44, 0.98] m`，前后深 `24 cm`、左右宽 `36 cm`、高 `16 cm`，质量 `0.4 kg`，壁厚 `4 mm`，底厚 `5 mm`。孔尺寸 `12 × 5.5 cm`，把手梁向内加厚至 `10 mm`。
- 忠实保留归档布局：箱体近侧 X 边比桌沿突出 `2 cm`；没有为了演示静默移动箱子。
- CONTACT 的罐和箱均具有真实 free joint；只有原始手指 mimic joint equalities，没有 weld 或 mocap。
- AIR 的道具固定、隐藏且不接触；线框只作位置参考，手指始终张开。因此 AIR 通过不能宣称抓起、负载保持或落箱成功。
- AIR 另外使用独立 `MjModel/MjData` 做桌箱碰撞检测：道具碰撞在 XML 编译时启用，且与 AIR 的机器人 qpos 布局相同；只调用 `mj_forward`，不积分、不施力、不改 live state。不能只给已经编译为无碰撞的道具修改 geom 掩码，因为 body 聚合掩码及碰撞 BVH 也缺失。

## CONTACT 状态机与真实成功判据

路径为 `STAND → HOME → SAFE_OUT → APPROACH → GRASP → CLOSE → PROBE → VERIFY_GRASP → LIFT_CLEAR → TRANSFER_ABOVE → DROP_RELEASE → OPEN → WAIT_DROP → RETREAT_HIGH → RETURN_HOME → FINAL_HOLD → COMPLETE`。

- **到位**：双腕位置误差 `<5 mm`、姿态误差 `<3°`、线速度 `<2 cm/s`、角速度 `<0.15 rad/s`，脚漂移 `<2 cm`，身体速度/姿态稳定，持续至少 `0.3 s`。每段还须满足原轨迹最短时长。
- **闭合**：进入 CLOSE 时只下发一次左手 `1`；至少等待 `2.5 s`，并确认拇指与对侧手指接触及腕部停稳。`CLOSED` 本身不代表抓到。
- **试提验证**：物体真实离桌至少 `8 mm`、不接触桌面/地面/货箱、手部承载竖直力至少为物重 `80%`；拇指与对侧手指法向对向角至少 `120°`，物体速度停稳、倾斜不超过 `30°`，相对手腕滑移 `<15 mm / 5°`，持续 `1 s` 才设置 `grasp_verified`。
- **抬升与平移**：保持闭合，持续监测抓持、滑移、限位、身体稳定性和碰撞。真实姿态及速度达标后才推进下一段。
- **允许松手**：必须已验证抓持；整个倾斜圆柱包络处于实际箱口内，XY 余量至少 `5 mm`；物体最低点和手部最低点距箱沿至少 `10 mm`；箱子触桌、倾斜 `<5°`、线/角速度停稳。箱口边界按向内突出的把手梁计算，手部高度按倾斜箱体最高箱沿计算。
- **落箱成功**：释放后手部脱离，圆柱完整包络位于内腔，货箱接触支撑力至少为物重 `80%`，罐不直接接触桌面或地面，罐和箱均停稳，持续 `1 s`；撤回后最终保持阶段再次验证。落箱不要求饮料直立。
- 每阶段最多 `10 s`。超时、抓空、掉落、严重穿透、跌倒或数值异常记录失败；搬运失败不在空中自动张手。AIR 包括启动阶段的任何桌箱几何穿透均判失败。

## 启动方式及旧证据失效说明

当前显式配置为 `startup_mode="hold_home"`：前 `0.4 s` 由同一个冻结 Reach 策略跟踪既有 HOME 世界目标，而不是每帧把目标改为当前手腕位置。它是**任务目标初始化调整**，不是更换权重、关节位置回放、碰撞禁用或训练。

理由：原 `follow_current` 启动会让左小指扫过桌面。保留的 CONTACT v1 在 `t=0.14 s` 因穿透 `3.436 mm` 停止，尚未抓取；对原 AIR 保存轨迹做正确的独立碰撞诊断，发现 STAND 在 `t=0.24 s` 曾穿透 `27.83 mm`。旧 AIR 当时只修改 geom 碰撞掩码，没建立 body/BVH，漏报了这段碰撞。

因此 `/root/autodl-tmp/Postman_Deploy/front_drop_20260927/air_preflight_v1/report.json` 内即使写着成功，**也不是有效接触试验准入证据**。保留原报告供追溯，禁止据此宣称启动安全。失败视频与轨迹保留在 `/root/autodl-tmp/Postman_Deploy/front_drop_20260927/contact_v1`，没有覆盖或删除。

## 运行方法

使用仓库现有部署虚拟环境。输出目录必须为空；下例记录最终交付所用命令。如果 `air_final` / `contact_final` 已存在，复现时必须换用新的目录名，并同步修改 CONTACT 的证据路径，不能覆盖既有结果。

先运行 AIR，可去掉 `--no-video` 录制空手完整路径：

```bash
cd /root/code/amo/Postman_Deploy
MUJOCO_GL=egl .venv-r2v2/bin/python deploy_mujoco/r2v2_front_drop.py \
  --config /root/code/amo/Postman_Deploy/deploy_mujoco/config/r2v2_front_drop.json \
  --mode air --no-video \
  --output /root/autodl-tmp/Postman_Deploy/front_drop_20260927/air_final
```

确认 AIR 报告 `success=true`、`air_passed=true`、`phase="COMPLETE"` 且没有 failure 后，使用同一配置和代码运行 CONTACT：

```bash
cd /root/code/amo/Postman_Deploy
MUJOCO_GL=egl .venv-r2v2/bin/python deploy_mujoco/r2v2_front_drop.py \
  --config /root/code/amo/Postman_Deploy/deploy_mujoco/config/r2v2_front_drop.json \
  --mode contact \
  --air-evidence /root/autodl-tmp/Postman_Deploy/front_drop_20260927/air_final/report.json \
  --output /root/autodl-tmp/Postman_Deploy/front_drop_20260927/contact_final
```

CONTACT 自动要求匹配的 AIR 证据：轨迹、ONNX、checkpoint、任务配置（不含证据文件路径本身）、机器人源资产、手配置、身体适配器、状态机、手控制器、场景代码、指标代码及 MuJoCo 版本绑定一致。修改这些输入后须重新运行 AIR，不应手改报告或放宽准入条件。

每次输出 `report.json`、`transitions.json`、`trace.json`、`targets.json`；录视频时另有 `front_drop.mp4`、阶段截图和 `final.png`。视频含正侧面全身及斜侧近景；结尾 `2 s` 为明确标注的终止帧冻结，不是继续仿真。**进程退出码 0 仅表示记录完成，成功与否必须看报告的 `success / grasp_verified / release_commanded / drop_verified / failure`。**

## 当前验证结果（2026-09-27）

- 数值接口对照通过，报告在 `/root/autodl-tmp/Postman_Deploy/front_drop_20260927/parity/report.json`。
- 修复碰撞诊断并使用 `hold_home` 后，`air_hold_home_v2` 完成全部阶段，仿真 `34.42 s`，`success=true`、`air_passed=true`，无启动碰撞；这是部署空手 Reach 的单次通过，不是统计鲁棒性或抓取结论。报告：`/root/autodl-tmp/Postman_Deploy/front_drop_20260927/air_hold_home_v2/report.json`。
- **CONTACT v2 单次完整成功**：`45.40 s` 到达 COMPLETE，`success / grasp_verified / release_commanded / drop_verified` 均为 true，failure 和 runtime_error 均为 null。完整录像为 `/root/autodl-tmp/Postman_Deploy/front_drop_20260927/contact_v2/front_drop.mp4`，报告为 `/root/autodl-tmp/Postman_Deploy/front_drop_20260927/contact_v2/report.json`。
- v2 关键时间：`11.40 s` 闭合、`15.70 s` 试提、`18.72 s` 验证抓稳并开始抬升、`22.72 s` 转运、`29.72 s` 松手、`35.38 s` 验证落箱并撤回、`45.40 s` 完成。松手时圆柱距箱口 XY 最小余量 `72.26 mm`、最低点距箱沿 `30.24 mm`，握持位置滑移 `3.04 mm`；落箱后货箱对罐的竖直支持力约 `0.9816 N`，与 `100 g` 物重一致。
- 此后仅修正 `drop_snapshot` 中 `drop_verified` 标记滞后一帧的记录问题，没有更改物理和控制参数；由于控制器代码哈希改变，重新生成了匹配的最终证据。**`air_final` 在 34.42 s 通过，`contact_final` 在 45.40 s 完整成功**，报告中所有抓取、释放、落箱标记均为 true，没有 failure 或 runtime_error，两份报告的 bindings 完全一致。最终交付录像为 `/root/autodl-tmp/Postman_Deploy/front_drop_20260927/contact_final/front_drop.mp4`，指标为同目录 `report.json`；旧 v2 不作为新哈希的准入证据。
- 最终视频 47.48 s，1280 × 800、25 fps、1187 帧，包含末尾 2 s 标注冻结，完整解码通过。抓取停稳时左腕误差 `1.30 mm / 0.37°`，释放停稳时 `1.17 mm / 0.74°`；搬运期间最大相对滑移 `3.04 mm / 0.56°`。最终物体完整处于箱内、与手脱离，箱体支持力约 `0.981 N`。全程没有机器人与桌箱接触。
- v2 和 final 是同一固定初值场景的两次成功，不是随机化鲁棒性验证。`peaks.foot_drift_m=41.58 mm` 包含 STAND 阶段脚锚点重置前的峰值；HOME 后的路径最大脚漂移为 `18.80 mm`，两者不能混用。
- 最终相关回归测试 **348 项通过**，覆盖新状态机、场景、数值对照、身体策略、手部、桌面接触及视频接口。
- 场景模块 16 项测试通过，包含源机器人物理参数不变、身体/手部通道分离、真手腕 site、空气/接触自由度、独立诊断能检测人为设置的几何交叠且不改变 live AIR 状态。相关测试为 `/root/code/amo/Postman_Deploy/tests/test_r2v2_front_drop_scene.py`。
