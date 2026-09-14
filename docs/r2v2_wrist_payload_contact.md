# Payload-v2 最终策略的实体箱接触复测

使用从旧 `model_4000.pt` 开始、完成 6000 次新更新的
`R2V2-Reach-CrateWristPayload-v2-28DoF/model_5999.pt`。
这是仿真中的完整定时探索，不是实机，不是抓稳后才允许高抬的严格 FSM。
旧 WristPath-v1 实验和 `--grasp-plan` 入口保持原行为。

## 与原生承重视频的区别

原生训练播放保持固定张手、虚拟桌箱，四个持箱阶段使用 0.4 kg 等效重力。
本接触实验使用完整可动手指和实际自由箱体，箱子质量 0.4 kg、尺寸
26 × 24 × 16 cm、双侧孔 12 × 5.5 cm，箱底初始 XY=(0.33,0) m，
桌面 Z=0.9609189697 m。**不叠加任何训练用等效外力**。
身体 50 Hz、手部 100 Hz、接触/力矩 1 kHz；无 IK 驱动、固定基座/手腕、
箱体动画或运行中 qpos 回放。

新版世界手腕目标来自 checkpoint 绑定的 payload-v2 路径，不复用旧的后拉 5 cm
高抬候选。闭指时间映射为 `CLOSE_SEAT`，后续完整执行
`PROBE_LIFT → HOLD_PROBE → LIFT_HIGHER → HOLD_HIGHER`，与原归档时序一致。
归档中的箱体期望位姿只作测量参考，不写入物理箱体；新增后缀未被当作已验证的
纯手抓取记录。二值闭指仍使用既有 75° 初始化与 curl 0.8 rad 配置。

目标参考保持原控制器的连续滤波，不因每步传入新目标而重置轨迹。
碰箱、碰桌、精度不足、抓空或滑移均记录并继续；跌倒、非脚部着地、数值异常、
意外外力或夹具仍停止。真实关节限位、惯量、碰撞、力矩限制不变。

`COMPLETE` 仅表示时序结束。真实拿起仍需双侧手指承重、离台 ≥8 mm、
没有桌/地/额外身体支撑、滑移和倾斜满足既有条件，且末态连续保持 ≥2 s。
严格验收还要求双腕 <5 mm/<3°、低速和全程约束检查通过。闭合命令不代表抓稳。

## 复现

从仓库根目录运行，输出目录必须为空或不存在：

```bash
.venv-r2v2/bin/python -u deploy_mujoco/r2v2_wrist_path_contact.py \
  --path-manifest /root/autodl-tmp/AMO_R2/logs/rsl_rl/r2v2_crate_wrist_payload_v2_28dof/2026-09-12_payload-v2-lift4-from4000/path_snapshot/manifest.json \
  --reach-config /root/autodl-tmp/Postman_Deploy/wrist_payload_contact_20260913/policy/reach_config.yaml \
  --parity-report /root/autodl-tmp/Postman_Deploy/wrist_payload_contact_20260913/parity/report.json \
  --output /root/autodl-tmp/Postman_Deploy/wrist_payload_contact_20260913/full_contact_trial
```

入口核对真实 task 身份、payload path contract、checkpoint/ONNX/路径哈希，
需要新的原生与部署观测/动作对照通过，不能只把旧报告改名使用。
产物为 `wrist_path_contact.mp4`、阶段截图、`report.json`、100 Hz `trace.json`、
`transitions.json` 与 `targets.json`。镜头保持正侧面全身和双手固定近景；末尾额外
2 s 明确标注为冻结展示，不属于物理稳定时间。

## 本次结果

2026-09-13 本次完整时序运行 64.21 s，无跌倒或数值异常；视频 1280×800、30 fps，
共 1988 帧、约 66.27 s（含明确标注的 2 s 终态冻结），完整解码通过。

末态箱子确已离开桌面：桌面与地面接触均为 false，桌面向上力为零，
双手实际净向上合力 3.92411 N，约为实际箱重 3.924 N。
100 Hz 记录中 50.58–64.21 s 连续 13.63 s 净空为正且无桌、地、非手部箱体支撑，
双侧手指各自实际承重、总手部承重保持在箱重 80% 以上。这段是真实物理时间，
不包含视频末尾冻结帧。
最低箱底相对桌面的净空仅 **3.844 mm**，箱体倾斜 **4.684°**；全程最大净空
约 6.370 mm，仍未达到原来的 8 mm 判据。因此“箱子已离台”和“严格拿起验收未过”
并不矛盾，不能将 `COMPLETE` 或看起来抬起改记为成功。

末态左腕 **7.47 mm / 2.47°**，右腕 **5.98 mm / 2.39°**。本次进入试提前未形成
持续双侧接触基线（`baseline_contact_verified=false`）；相对试提入口的腕—箱
关系变化约 19.84 mm，超过原判据，但不能把包含受力就位过程的该变化直接解释为
“已抓稳之后的滑移”。物理拿起和严格精度标志均为 false。
高位保持段自身的腕—箱关系变化约左 2.89 mm、右 3.03 mm，最低净空由约
6.30 mm 缓慢降至 3.84 mm。这是分段诊断，不是重设抓稳基线或更改原成功判据。

开场原 HOME 布局仍存在手—箱重叠及短暂手指限位软件残差，均保留在原始视频和
`constraint_events` 中，没有删除这段记录、放宽物理约束或重新摆放箱体。
后续需区分初始碰撞、手指受力就位、离台净空及末端精度，不能只凭本次结果归因为策略 OOD。

本次运行完整保存 `source_snapshot/`。接触/新路径回归 72 项、策略对照相关回归
24 项通过；测试通过不替代物理抓握验收。确切数值以本次 `report.json` 和 `trace.json` 为准。
