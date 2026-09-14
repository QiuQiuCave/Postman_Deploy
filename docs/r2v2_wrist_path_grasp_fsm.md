# 抓握验证后切换更高世界手腕目标

这是独立的真实接触仿真 FSM，保留旧的完整定时碰箱实验，不更改训练权重或训练路径。
身体仍由 28 维 Reach 策略控制，手指仍走独立 0/1 控制器；没有 IK 驱动、基座固定、
手腕焊接、辅助力或箱体位姿回放。

## 状态与目标切换

准备、外展、转腕、插入沿用冻结的 WristPath-v1 归档。之后：

1. `CLOSE_SEAT`：闭合并小幅向上就位，当前探索方案为手腕目标向上 5 mm，持续 4.8 s。
2. `PROBE_LIFT`：最新探索方案相对插入目标后移 5 cm、上移 3 cm；最多观察 10 s。
   双侧手指—孔梁接触持续 0.3 s 后，记录实测手腕—箱体关系作为滑移基线。
3. `GRASP_VERIFIED`：真实离台、双方承重、无额外身体/桌面支撑且相对位姿稳定，
   持续 0.3 s 后才能进入；再确认 0.2 s，不以 `CLOSED` 或命令 `1` 为抓稳证据。
4. `LIFT_HIGHER`：只在进入状态时调用双手 `set_target_world`，改为更高的绝对世界目标。
   保持原训练过的速度/加速度滤波参数，不逐帧重置目标轨迹；运动阶段为 4 s。
5. `HOLD_HIGHER`：箱底必须比抓稳时再高至少 8 mm，且真实稳定保持至少 2 s。
   检查最后一个时刻，不能把先抬起后掉落记为最终成功。

抓空超时、抓稳后滑移/失去承重，进入 `OBSERVE_FAILED` 观察 2 s 再结束：手保持闭合，
不自动松手、不再推进新的高抬目标。碰桌、碰箱和末端误差本身不会让尝试立即退出。
跌倒、非脚部着地、数值异常仍终止物理积分；真实限位、碰撞、惯量和力矩上限不变。

`pickup_observed` / `grasp_verified_ever` 是历史事件；`physical_pickup_verified` 检查最终
仍保持的抓握；`higher_lift_verified` 检查最终更高处保持。`COMPLETE` 和退出码 0 都不
代表这些物理成功条件已通过。

## 当前高抬候选及限制

固定策略使用 `model_1099.pt` 对应的已校验 ONNX，避免与正在续训的权重混用。
候选文件：

`/root/autodl-tmp/Postman_Deploy/wrist_path_grasp_fsm_20260912/geometry/probe3_higher5_linear/candidate.json`

抓握整体的候选变换为：绕初始箱体中心共同世界 Y 旋转 −5°，再后移 5 cm、上移 5 cm。
因此两只手腕最终目标 Z 约为 **1.16557 m**，比插入目标高约 **4.62 cm**，
比最新 3 cm 试提目标高约 **1.62 cm**；不是把每只手腕简单平移 5 cm。

目标间采用逐腕位置线性插值、四元数 SLERP 做静态采样检查；实际参考由原 ReachPolicy
滤波器生成，不能据此声称实际机器人沿严格刚体路径移动。

必须区分两种检查结果：

- 名义/准备两种双脚位置、三段各 11 点，共 66/66 点通过腕姿态、关节余量、自碰和
  支撑几何代理检查；最大腕误差 0.515 mm / 0.480°。
- 这些精确姿态 IK 解仍存在最高约 108 mm 的腰—箱穿透；最新候选的相邻 IK 解最大
  关节变化约 0.473 rad（名义双脚）/ 0.039 rad（准备双脚）。首轮 2 cm 试提候选
  曾出现 1.827 rad 分支跳变。加入非手部避箱罚项的 12 个端点解消除了检测到的非手部穿透，但腕朝向
  误差 3.48–7.22°，未同时满足 3° 判据。

所以该候选仅是用户允许碰箱继续的 **探索目标**，不是连续、无碰撞的搬运路径，
更不是抓箱成功证明。IK 结果仅用于诊断，绝不写入全身控制。真实成功指标仍拒绝
非手部托箱或借助桌面支撑。该更高目标尚未并入正在进行的同任务训练。

计划加载会核对训练路径、模型源文件、目标和几何证据 SHA256；拒绝失败/不完整采样、
只有末点的证据或插值方式不一致的证据。旧归档及 checkpoint 的路径哈希不被改写。

## 运行

在 `/root/code/amo/Postman_Deploy` 中执行，输出目录须不存在或为空：

```bash
.venv-r2v2/bin/python -u deploy_mujoco/r2v2_wrist_path_contact.py \
  --path-manifest /root/autodl-tmp/AMO_R2/logs/rsl_rl/r2v2_crate_wrist_path_v1_28dof/2026-09-11_path-v1-lift2-rear3-from4502/path_snapshot/manifest.json \
  --reach-config /root/autodl-tmp/Postman_Deploy/wrist_path_contact_20260912/policy/reach_config.yaml \
  --parity-report /root/autodl-tmp/Postman_Deploy/wrist_path_contact_20260912/parity/report.json \
  --grasp-plan /root/autodl-tmp/Postman_Deploy/wrist_path_grasp_fsm_20260912/geometry/probe3_higher5_linear/candidate.json \
  --output /root/autodl-tmp/Postman_Deploy/wrist_path_grasp_fsm_new_trial
```

增加 `--no-video` 仅记录真实仿真和指标；省略 `--grasp-plan` 完全保留旧定时实验。
视频名为 `wrist_path_grasp_fsm.mp4`，附带 `report.json`、`trace.json`、`targets.json`、
`transitions.json` 和阶段截图。FSM 事件、目标设置事件及接触观测均写入 report/trace。

实现：`common/r2v2_wrist_path_grasp_fsm.py`。
候选生成：`tools/check_r2v2_grasp_lift_geometry.py`。
CPU 回归覆盖真实抓稳后才发高目标、目标只设置一次、抓空/滑落不误报、末帧失去接触，
以及模型/路径/完整分段证据绑定；合成观测的成功分支不等于真实物理成功。

## 本轮真实接触验证结果

两次均使用冻结的 1099 ONNX，仿真运行 58.22 s，无跌倒或数值异常，没有因为碰箱
提前中断。两次 `physical_pickup_verified`、`higher_lift_commanded` 和
`higher_lift_verified` 都为 false：不能称为完成高位抬箱，也不能把视频的 `COMPLETE`
当成成功。高位分支目前只有合成观测回归覆盖，尚无真实抓稳后执行的物理验证。

- 首轮：`geometry/final_linear_higher5_rear5_pitchm5/candidate.json`，试提目标相对
  插入位后移 3 cm、上移 2 cm。未捕获持续双侧孔梁接触，最终桌面承重 3.865 N。
  产物目录：`/root/autodl-tmp/Postman_Deploy/wrist_path_grasp_fsm_20260912/full_contact_fsm_trial`。
- 第二轮：当前 `geometry/probe3_higher5_linear/candidate.json`，试提相对插入位
  后移 5 cm、上移 3 cm。47.290 s 捕获持续双侧手指—孔梁接触基线，形成部分承重，
  但未达到真实试提条件，56.22 s 超时，保持闭合再观察 2 s 后结束。
  产物目录：`/root/autodl-tmp/Postman_Deploy/wrist_path_grasp_fsm_20260912/probe3_full_contact_fsm_trial`。

第二轮的直接测量：

- 最终桌面向上承重 **2.562 N（箱重的 65.3%）**；左/右手指—孔梁向上力
  0.597 / 0.749 N，双手净向上合力 1.362 N（34.7% 箱重）。
- 试提阶段目标相对闭合末点再上移 25 mm、后移 50 mm；实际双腕上移
  **13.50 / 15.67 mm**、后移 51.25 / 49.89 mm。
- 最终左腕误差 **23.34 mm / 15.07°**，右腕 **17.68 mm / 12.82°**。
- 试提阶段箱底最高瞬时净空仅 **0.364 mm**，远低于 8 mm 判据；有瞬时桌面力为零，
  不能声称从未短暂脱离接触，但真实稳定试提持续时间为 **0 s**。

这说明第二轮已经产生双侧承重，仍卡在有效试提；腕目标上移不等于实际腕或箱体等量
上移。还不能仅凭本次测试断言是力矩不足、姿态 OOD 或几何限制，需要后续分离验证。
没有通过放宽判据强制进入高位段，训练任务也没有因此被改成接触/负载训练。

两段视频均为 1280×800、30 fps、约 60.27 s（含末帧标注的 2 s 静止展示），完整解码
通过。部署相关六组回归测试共 **104 passed**。
