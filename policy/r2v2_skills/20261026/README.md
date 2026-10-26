# R2V2 权重交接（2026-10-26）

这是可随 `r2v2-fsm` 分支正常 clone/pull 的真实文件，不是 LFS 指针，
不需要访问原服务器。总计约 90 MiB，保留完整原始 checkpoint（含优化器）、
匹配 ONNX、训练配置、必要轨迹和小体积证据；不包括视频、大日志、环境或机器人资产。
代码分别在 `QiuQiuCave/HALO-mj:halo-r2` 和 `QiuQiuCave/Postman_Deploy:r2v2-fsm`。
仅仿真使用，不代表实机可用或整机搬箱闭环已成功。

## 选择哪个

| 目录 | PT / ONNX | 接口 | 用途和验收边界 |
|---|---|---|---|
| `loco_9999` | `model_9999.pt` / `policy.onnx` | 560 → 14 | 本次选择的步行基线，原固定臂模型训练 10000 次；已在新整机可乐搬运中用外部手臂 PD 保持完成一次后退两桌闭环。不能据此认定横移、转身或负载搬箱通过。 |
| `dual_wrist_vertical_grip_stop4` | `model_stop_4.pt` / `policy_wrist_payload_v2.onnx` | 1460 → 28，`wrist_world_v2` | **新整机双臂竖把轨迹实验候选，非稳定版**。从 front 9999 后训练仅 5 次，随机站立 21/24、保护停止。适合继续诊断/后训练，不要直接当作验收通过的搬箱策略。 |
| `front_drop_9999` | `model_9999.pt` / `policy_front_manipulation_v1.onnx` | 1460 → 28，`wrist_world_v2` | 已跑通固定场景单手可乐闭环的恢复基线。含双腕观测/目标和全身动作，但该专项训练不等于已经验证任意双臂搬箱能力。 |
| `legacy_dual_reach_12000` | `model_12000.pt` / `policy.onnx` | 1460 → 28，`legacy_tcp_v1` | 旧身体双臂限速 Reach 参考。**不是新模型手腕策略**；同维度不代表可互换，不作为新整机默认策略。 |

原 26 cm 孔箱 `WristPayload 5999` 未在本机保留的权重中找到，不在本包中，
不能拿其他文件冒充。新候选的箱壳是左右宽 36 cm × 前后深 24 cm × 高 16 cm，
竖把直径 4 cm、长度 12 cm、轴线 y=±24 cm，空箱 0.4 kg，桌面 z=0.98 m。
箱底世界位置 `[0.45,0.44,0.98]`；轨迹在固定任务世界中，
加 `[0.12,0.44,0]` 才是该部署场景的世界坐标。

## 拉取和校验

在本仓库根目录，使用已配置的部署环境：

```bash
git switch r2v2-fsm
git pull --ff-only origin r2v2-fsm
.venv-r2v2/bin/python tools/verify_r2v2_skill_bundle.py
```

`manifest.json` 与 `SHA256SUMS` 列出大小和哈希。ONNX 内包含归一化、
按名称排列的关节、默认姿态、动作缩放及 PD 参数，不要再归一化一次。
10 帧历史按观测项拼接，每项内部从旧到新，不是整帧堆叠。
`.pt` 是训练 checkpoint，不能使用 `torch.jit.load`；仅加载可信来源的文件。

可在兼容 mjlab 1.3 / rsl-rl 的训练解释器下进行额外数值复核：

```bash
uv run --no-project --python /path/to/reach-env/bin/python \
  python tools/verify_r2v2_skill_bundle.py --torch-parity
```

这是固定网络和 ONNX 的同输入比较，不替代原生观测重建对照或物理验收。
`inference_verification.json` 记录本次检查；各模型 `evidence/` 保留原始证据。
旧报告的绝对路径只用于溯源，不需要建立同名 `/root/...` 目录。
旧报告中的适配器哈希可能因代码迭代失效，不能删除校验来强行运行。

## 已有 Loco → Reach 可乐闭环

先安装新机器人资产（`r2v2_description/source/r2v2_with_hand`）及
`requirements-r2v2-reach-sim.txt` 对应环境。模型文件不是机器人网格。
新增配置只把已通过那次演示的两处权重目录改为仓库相对路径，
场景、姿态、速度、切换、判据均保持原样；请从仓库根目录执行：

```bash
.venv-r2v2/bin/python deploy_mujoco/r2v2_loco_cola.py \
  --config deploy_mujoco/config/r2v2_loco_cola_shared_weights.json \
  --stop-after place --output /tmp/r2v2_cola_shared_trial
```

输出目录必须不存在。是否成功查看 `report.json`，不能仅看退出码。
原单次闭环耗时 60.78 s、最终罐平面误差 6.87 mm；途中有约 15 cm 横向漂移和
短暂碰桌，故不是无碰撞、鲁棒性或双手搬箱证明。详见
[闭环记录](../../../docs/r2v2_loco_cola.md)。

## 双腕竖把候选怎么用

使用 `common.r2v2_reach_policy.ReachPolicy`，明确指定
`expected_endpoint_contract="wrist_world_v2"`。`set_target_world(side, p, q)`
的 p 是真实 `left_hand_roll_link` / `right_hand_roll_link` 原点的世界位置，
q 是 WXYZ 四元数；不加旧 TCP 偏移、不把抓手中心当成腕目标。
身体 28 路动作不包含手指。身体 50 Hz、独立手部 100 Hz、仿真/PD 1 kHz。
Loco 只观察/控制腿腰 14 路，不观察手臂或手指；切到 Loco 后手臂保持需要外部 PD，
不等于物理固定或手腕世界锁定。

在相邻 HALO-mj 仓库根目录，用其 Reach 专用 mjlab 1.3 环境重放：

```bash
uv run --no-project --python /path/to/reach-env/bin/python \
  python scripts/tools/r2v2_training_component.py reach_skill script \
  render_wrist_payload.py \
  --checkpoint ../Postman_Deploy/policy/r2v2_skills/20261026/dual_wrist_vertical_grip_stop4/model_stop_4.pt \
  --payload-mass-kg 0.4 --output-dir /tmp/r2v2_dual_wrist_trial
```

需匹配机器人资产、GPU 和依赖；参照 HALO 的组件安装文档，不能用根 mjlab 1.4
环境直接代替。包内 `path_snapshot/` 与 checkpoint 哈希绑定，不能随意编辑。
训练配置和 launcher 中原服务器路径是归档信息，不是可直接运行的跨机器启动器。

最近名义轨迹重放完成约 41.9 s，高位保持末秒双腕位置误差约 3.30 / 2.47 mm、
角度误差 1.25 / 1.19°；但脚部最大漂移 38.6 mm 超过 20 mm 门槛。
仿真箱体是线框，手指保持张开，0.4 kg 是腕部等效向下力，不是真实抓箱。
站立测试源策略为 23/24，新候选降为 21/24；未通过，训练已停止。
新候选已有导出同输入一致性，尚无其独立整机部署观测对照和真实抓箱验收。
本次只发布现存权重，没有重启训练、调整门槛或改写失败证据。
