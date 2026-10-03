# 上侧接近、抓取圆柱上段、搬移与松手实验

## 范围与当前结论（2026-09-13）

这是第一阶段**纯手部仿真**，不是全身 Reach 策略或实机实验。两个真实手腕为动态 free body，只有手腕通过 weld 跟踪 mocap 夹具；圆柱为自由刚体，仅由重力、桌面和手部接触运动。运行中不设置圆柱 qpos、不焊接圆柱、不施加额外外力。

已找到 40 mm 直径、120 mm 高、100 g 基线的完整桌面抓起—搬移—放下参数。58 mm 直径、145.4 mm 高、**假设满载质量 350 g** 的大规格尚未通过同样的完整验证；350 g 并非实测数据，物体是实心碰撞/刚体近似，不模拟液体和薄壁罐。

该抓型从**斜上方**接近，指伸展轴与竖直向下夹角为 60°，不是垂直伸直五指往下扣。闭合后如物体发生有限倾斜，先真实试提，再通过手腕动作扶正；不会直接改物体姿态。不能把名义桌面成功直接解释为能够无碰撞地放进箱子。

旧固定手腕正向抓握、全身桌面抓取和箱子实验保持原样；未改训练策略、未开启训练。

## 复现

在仓库根目录执行，输出应使用新的空目录，避免覆盖旧实验：

```bash
.venv-r2v2/bin/python deploy_mujoco/r2v2_top_grasp.py \
  --profile baseline_40mm_100g \
  --candidate-json deploy_mujoco/config/r2v2_top_grasp_upper_40mm.json \
  --output /root/autodl-tmp/Postman_Deploy/top_grasp_upper_new_run
```

将 profile 改成 `sleek_330ml_approx_full` 可复核大规格的失败，不会偷偷缩小尺寸或减轻质量。无头环境自动使用 EGL；`--no-video` 只运行仿真并保存数值记录。

批量候选使用显式 JSON 列表，每项为 `case_id`、`profile`、`candidate`：

```bash
.venv-r2v2/bin/python tools/sweep_r2v2_top_grasp.py \
  --cases-json /path/to/cases.json \
  --output /root/autodl-tmp/Postman_Deploy/top_grasp_sweep_new_run \
  --workers 4
```

渲染器正常退出只表示记录过程完成，**必须读取 report.json 的 success / failure**；成功与失败均可生成视频。

## 文件与接口

- `common/r2v2_top_grasp_scene.py`：真实手部、固定台面、自由圆柱与候选姿态的静态碰撞诊断。
- `common/r2v2_top_grasp.py`：成功判据驱动的实验状态机、接触/载荷/滑移测量。
- `deploy_mujoco/r2v2_top_grasp.py`：正侧面与斜近景双画面录像，状态日志和完整轨迹。
- `tools/sweep_r2v2_top_grasp.py`：最多 4 个 CPU worker 的有界动力学筛选。
- `deploy_mujoco/config/r2v2_top_grasp_upper_40mm.json`：上移后抓取上段的轻量基线。
- `deploy_mujoco/config/r2v2_top_grasp_baseline.json`：较低的中段包握成功对照，**不是上端抓取标定**。
- `deploy_mujoco/config/cylinder_profiles/`：物体尺寸、质量与来源说明。

`TopGraspCandidate` 的欧拉调试量均为度，位置均为米。姿态构造为 `Rz(yaw) @ Ry(spread_tilt) @ Rx(tilt) @ R_fingers_down`，内部使用旋转矩阵和 wxyz 四元数插值。`tilt` 沿指屈曲方向倾斜；`spread_tilt` 沿拇指/四指展开方向倾斜，两者不是同一自由度。总偏离向下方向不超过 60°。

`depth_m` 是指定的名义拇指局部 X 几何参考，不等于倾斜后实际接触距罐顶的深度。其负值会升高手腕；上段基线为 -45 mm，相比中段对照整体升高 45 mm。是否抓在上端应根据记录中的 `hand_object_contacts[].depth_below_object_top_m` 判断，而不是根据参数名字推断。

`lateral_m` 控制掌面间距，`across_m` 控制圆柱在四指排列方向的偏置；`open_curl_rad` 只覆盖当前候选四指张手角。75° 拇指对掌、0.04 rad 拇指张手角、原闭合角、Ruckig 速度/加速度/jerk、真实惯量/限位/碰撞与力矩上限均保持不变。

## 流程与验收

`READY → APPROACH → SETTLE → CLOSE → PROBE_LIFT → VERIFY → [UPRIGHT → VERIFY] → LIFT → HOLD_LIFT → TRANSLATE → HOLD_MOVED → LOWER → PLACE_SETTLE → OPEN → RETREAT → FINAL_HOLD → COMPLETE`

每段仅在进入时设置目标或发送二值手指指令。先试提 20 mm，必要时仅尝试一次实测关系驱动的扶正，再复验；通过后增加 60 mm 抬升、沿世界 X 搬移 100 mm。落放使用当时实测腕—物关系的逆，使圆柱竖直落桌，不盲目复用最初抓取姿态。

关键判据：

- “闭合”不等于抓住：拇指与四指有有效接触，作用在物体上的接触法向夹角至少 120°；这只是对置筛选，不是数学上的 force-closure 证明。
- 正式抓起要求最低点净空至少 8 mm、无桌/地面支撑、手部向上承重至少为物重 80%、物体倾角不超过 10°、低速并连续稳定 1 s。
- 扶正前的临时承重不算成功；扶正期间仍要求离台 8 mm、对置接触、腕物相对平移小于 15 mm、相对旋转小于 5°。失去握持立即失败，不在空中下发松手。
- 搬运期间监控实际接触和滑移；落桌稳定后才发 0，撤离后物体需独立稳定至少 1 s，最终 XY 误差不超过 20 mm、倾角不超过 10°。
- 每个状态设有限超时且不超过 10 s。手碰桌、物体落地、过量穿透、限位异常或数值异常均失败，不隐藏失败继续播放假成功。

仿真与力矩更新为 1 kHz；手部参考、观测日志和接触峰值采样为 100 Hz。所有“最大值”均是该采样率下的统计，不声称捕获每个 1 ms 瞬间的接触峰值。松手后的正常腕物分离不计入握持滑移。

## 实验记录

本轮输出根目录：`/root/autodl-tmp/Postman_Deploy/top_grasp_20260913/`。

- `diagnostic_baseline20/`：早期失败诊断；中指先推偏圆柱，两指最终同侧弱接触。当时旧 `opposed_contact` 仅表示两类手指同时有力，后续实现已补上真正法向夹角检查；不要混用其旧语义。
- `baseline_pick_place_final/`：中段包握成功对照。
- `upper_pick_place_final/`：上段轻量基线的完整视频与数值记录。
- `sleek_upper_attempt/`：同一上段目标、大规格 350 g 的真实失败录像。扶正时约 5.05° 相对旋转滑移触发停止，没有提前松手，也没有当作完成抓取。
- 各 `*_sweep/`：保留静态拒绝和动力学失败，不能把候选总数都算成实际完成的抓取次数；各轮状态机版本有演进，不适合汇总成统一成功率。

每段录像目录包含 `report.json`、100 Hz 的 `trace.json`、`targets.json`、`transitions.json`、阶段截图和 `top_grasp.mp4`。轨迹含双向世界位姿与 `T_wrist_object`、手指参考与实际关节/力矩、接触位置/法向力/向上支撑力、物体最低点和滑移。视频末尾 2 s 是明确标记的定格，不计入物理稳定时间。

上段名义参数最终实测：真实仿真 27.52 s；主要负载接触位于距罐顶约 0–21 mm，最大净空 82.36 mm，连续离桌约 15.07 s，搬运阶段最大滑移 4.39 mm，含落放阶段为 6.14 mm / 3.82°；最终 XY 误差 4.19 mm，手指脱离，圆柱独立稳定。扶正前曾倾斜 20.15°，不能表述为全程保持竖直。

`upper_local_sweep/` 额外测试深度 ±2 mm 和掌间距 ±2 mm 四项，3/4 完成，掌间距 -13 mm 项在落放/释放验收失败。这个很小的邻域筛选**不是鲁棒性证明**，更不是随机化批量验收。单元/场景/渲染与物体规格相关测试共 232 项通过，单元测试也不能替代真实抓握验证。

特别注意：当前上段放置姿态的腕原点距台面约 155.1 mm，仍比 160 mm 箱沿低约 4.9 mm，且腕在瓶体侧方偏移约 143 mm。因此下一步即使只拿轻量基线入箱，也必须重新做整手/箱沿碰撞与松手撤离检查，不能直接把当前轨迹搬过去。

## 下一阶段边界

先确认上段抓型及大规格握持，再增加 26 × 24 × 16 cm 箱子。应按物体最低点而非手腕高度规划“抬升越过箱沿 → 平移 → 下放”，同时检查掌部、手指张开扫掠与撤离路径；同台面至少需要超过 16 cm 箱沿并留余量，当前 8 cm 抬升不能直接复用。

接入全身前还需验证手腕目标/朝向是否处于策略可达范围，以及箱口与前臂碰撞余量。本实验的手腕夹具成功不证明策略能跟踪新目标；不会据此自动启用全身或新训练。

2026-09-15 新增独立全身入口，见 [全身上端抓取验证](r2v2_top_grasp_fullbody.md)。保留本纯手部基线不变；全身入口先执行 AIR 验证，通过后才允许真实 CONTACT。
