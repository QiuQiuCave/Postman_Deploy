# R2V2 正手抓取：恢复水平腕姿态

2026-09-23 按用户要求新增正手配置，保留此前的上端抓取配置与记录。目标是取消大幅翻腕，沿已验证的 75° 手型、正手腕柱关系接近圆柱；不扩大机器人限位，不改变惯量、碰撞、手指力矩上限或身体策略。

## 配置与实现

- `deploy_mujoco/config/r2v2_front_grasp_fullbody.json`：新的 `grasp_style: front` 配置，调试 yaw 固定为 0°，内部四元数为 `[1, 0, 0, 0]`。
- `common/r2v2_front_grasp_path.py`：正手路径与归档初始腕柱关系读取。
- `common/r2v2_top_grasp_fullbody.py`：复用同一个世界手腕目标接口和实际状态判据；默认仍为 `upper`，需显式选取正手配置。
- `deploy_mujoco/r2v2_top_grasp_fullbody.py`：共用录制入口；正手画面标为 `FRONT GRASP`。
- `tests/test_r2v2_front_grasp_fullbody.py`：新增 34 项回归测试。

正手标定取自版本控制中的 `reference_motion_bank/r2v2_grasp/thumb75_cylinder40mm_100g/left/grasp_record.json`，读取 **initial** 关系，而不是受力后已经偏移、倾斜的握持关系：

`T_world_wrist = T_world_cylinder_initial × inverse(T_wrist_cylinder_initial)`

其中圆柱相对左腕平移为 `[0.145, -0.035, 0] m`，相对姿态为单位四元数。手腕指 `left_hand_roll_link` 原点，身体策略端点仍为 `wrist_world_v2`，没有旧 TCP 偏移。读取的归档只验证初始抓型，**下面的搬运路径是新规划，不是已经成功的全身运动记录**。

## 场景与路径

沿用直径 4 cm、高 12 cm、质量 100 g 的圆柱及可乐外观；这不是装满真实 330 mL 可乐的物理规格。本轮先做同台面搬放，不添加货箱。

正手腕原点与圆柱中心等高，因此重新选择 0.98 m 台高，使腕部处于较自然的待机高度附近。台面范围为 X `[0.36, 0.75]`、Y `[-0.10, 0.42]` m。初始圆柱中心 `[0.40, 0.13, 1.041]` m，底部有 1 mm 初始化间隙；名义放置中心 `[0.40, 0.03, 1.040]` m。

Y 不直接沿用此前候选的 0.18 m：静态 HOME 检查发现该摆放与初始左手穿插约 10.55 mm；改为 0.13 m 后消除该初始化干涉。

| 阶段 | 左腕世界 XYZ / m | 说明 |
| --- | --- | --- |
| SAFE_OUT / HOVER | `[0.255, 0.215, 1.081]` | 先到物体外侧并稍高 |
| APPROACH | `[0.255, 0.215, 1.041]` | 保持外侧 5 cm，降至抓取高度 |
| GRASP | `[0.255, 0.165, 1.041]` | 沿左手掌面法向 −Y 靠近 |
| PROBE / 名义 UPRIGHT | `[0.255, 0.165, 1.061]` | 试提 2 cm |
| LIFT | `[0.255, 0.165, 1.121]` | 总计抬高 8 cm |
| TRANSLATE | `[0.255, 0.065, 1.121]` | 向身体中线平移 10 cm |
| 名义 PLACE | `[0.255, 0.065, 1.040]` | 去掉初始 1 mm 落差 |
| RETREAT | `[0.215, 0.185, 1.080]` | 向后/外侧撤离，避免拇指拖带 |

所有名义姿态保持单位四元数，不绕指轴翻掌。左掌朝局部 −Y，不能误把手指纵向 +X 当成接近法向。为兼容原 AIR 阶段证据，内部保留 `TURN_WRIST` / `HOVER` 名称；在正手模式中前者仅保持水平姿态，录像显示 `HOLD_UPRIGHT (NO FLIP)`，后者显示 `ALIGN_OUTSIDE`。

真实握持后的扶正与放置仍由**实测腕柱关系**反算，不能只播放上表的名义姿态。超过既有 AIR 名义目标偏差保护（2 cm / 5°）则停止并重新筛选，不为了演示擅自放宽。身体保持 50 Hz，手部轨迹 100 Hz，物理与力矩 1 kHz；右腕世界锁定、到位停稳、接触/承重/滑移验证以及失败不在空中松手均保持。

## 本轮检查结果与限制

静态诊断目录：`/root/autodl-tmp/Postman_Deploy/front_grasp_20260923/geometry/`。

- `static_ik_report.json`：HOME 及 8 个不同任务关键点的离线全身逆解。原关节限位最小余量 **3.343°**，关键点无身体自穿插和机器人碰桌。源手掌—拇指约 0.171 mm 的轻微网格交叠保留，不以关闭自碰撞消除。
- 每段取 31 个关节插值点，最大身体交叠约 **0.0104 mm**，无机器人碰桌/非脚部着地，最大躯干倾角约 2.33°。不能称为严格连续无碰撞证明；质心只检查足部 AABB 范围，不是动态平衡证明。
- `static_grasp_ik.png`：正手姿态示意，图上明确标注 **STATIC IK ONLY / NOT POLICY ROLLOUT / HAND OPEN**。为便于辨识物体，静态诊断使用橙色圆柱外观；碰撞尺寸不变。
- 没有执行物理步、闭合、抬物或策略 rollout，静态解不用于驱动身体，也不作为 AIR 通过证据。
- 正手 34 项测试及原上端路径/全身/渲染回归合计 **225 项通过**。
- 全量 R2V2 回归为 **1134 通过、9 跳过、1 失败**；失败是旧 payload 归档测试找不到数据盘上的 `selected_path_posttrain_preflight_20260911/revised_lift2_rear3_v1/manifest.json`。没有删除或跳过该失败来制造全量通过，也未改动其测试。

运行前检查阻塞于缺失文件：原数据盘目录 `/root/autodl-tmp/Postman_Deploy/wrist_payload_contact_20260913/` 当前不存在。`air_preflight/report.json` 记录 `INITIALIZATION` 阶段缺少 `policy/reach_config.yaml`，没有产生任何动态实验帧；**不是正手动态测试失败，也不是通过**。

继续同一策略的验证需要恢复这套文件或提供其新路径：

1. `policy/model_5999.pt`，SHA256 `bb82a46b4c4ba62cec8ec51116298b62532b147d2b3f03c5e3d8e8ab272c79c6`。
2. `policy/policy_wrist_payload_v2.onnx`，SHA256 `a4200572a78c46963105f00503c75e52110c8fc47e4f356315e19cc37952155e`。
3. 对应 `policy/reach_config.yaml` 和 `parity/report.json`；若对照报告缺失需重新验证，不能伪造通过证据。

检查了含 Git 忽略项的本地训练/部署目录，现有较早任务的 checkpoint 不等于这版新整机 wrist 策略，未擅自替换，也未启动训练。

## 恢复文件后的运行

在仓库根目录执行；使用新的空输出目录，不覆盖检查记录：

```bash
.venv-r2v2/bin/python deploy_mujoco/r2v2_top_grasp_fullbody.py \
  --config deploy_mujoco/config/r2v2_front_grasp_fullbody.json \
  --mode air \
  --output /root/autodl-tmp/Postman_Deploy/front_grasp_20260923/air_verified_candidate
```

完整 AIR 精度及碰撞验收通过后，才将该 `report.json` 填入正手配置的 `air_evidence` 并运行 CONTACT。输出视频沿用共用入口的文件名 `top_grasp_fullbody.mp4`，正手模式由配置、画面标题与报告 `grasp_style` 明确区分。不能复用旧上端抓取 AIR 报告。
