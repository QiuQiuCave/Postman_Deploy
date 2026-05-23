# Selected Deploy Reference Motions

这里保存已经人工筛选过、适合后续 sim2sim / deploy 实验的 reference motion
片段。部署时策略不会自动读取本目录;需要手动把选中的 `.npz` 复制到对应
policy 的 `motion/` 运行时文件名。

## 当前已选

| 文件 | 来源 | 片段 | 长度 | SHA256 | 读取策略 | 运行时目标路径 | 触发键位 |
|---|---|---|---|---|---|---|---|
| `walk4_subject1_1m17_1m40_23s__DualAgentTracking__walk_tracking_ref.npz` | `/home/qiuziyu/code/postman/upper_lower/data/demo/lafan/walk4_subject1.npz` | `1:17-1:40`, frame `[3850, 5000)` | `1150` frames, `23.00s @ 50Hz` | `8c970e646fa59af71dd9e95959d0098496b4af81b08a6bd4ee82028b08dcfd55` | `policy/dual_agent_tracking/DualAgentTracking.py` / `DualAgentTracking` | `policy/dual_agent_tracking/motion/walk_tracking_ref.npz` | sim2sim `a+l1`; deploy_real `A+L1`; current runtime |
| `run1_subject2_2m15_2m22_7s__DualAgentRunTracking__run_tracking_ref.npz` | `/home/qiuziyu/datasets/gae_mimic_dataset/extend_datasets/lafan1_dataset/g1/train/run1_subject2.npz` | `2:15-2:22`, frame `[6750, 7100)` | `350` frames, `7.00s @ 50Hz` | `d1a6c74a8c1ea640882dd6b71fde8b79e2c4ebb8556620aa65956ef93955720d` | `policy/dual_agent_run_tracking/DualAgentRunTracking.py` / `DualAgentRunTracking` | `policy/dual_agent_run_tracking/motion/run_tracking_ref.npz` | sim2sim `b+l1`; deploy_real `B+L1`; current runtime |
| `sprint1_subject2_0m05_0m12_7s__DualAgentDanceTracking__dance_tracking_ref.npz` | `/home/qiuziyu/datasets/gae_mimic_dataset/extend_datasets/lafan1_dataset/g1/train/sprint1_subject2.npz` | `0:05-0:12`, frame `[250, 600)` | `350` frames, `7.00s @ 50Hz` | `1f2f10098c2fe0aea93b09afcf2211815e4378f84993ab0feb6a8ae190a6e2db` | `policy/dual_agent_dance_tracking/DualAgentDanceTracking.py` / `DualAgentSprintTracking` | `policy/dual_agent_dance_tracking/motion/dance_tracking_ref.npz` | sim2sim `y+l1`; current runtime |
| `jumps1_subject2_first75_0m23_0m30_7s__DualAgentJumpTracking__jump_tracking_ref.npz` | `/home/qiuziyu/datasets/gae_mimic_dataset/extend_datasets/lafan1_dataset/g1/train/jumps1_subject2_first75.npz` | `0:23-0:30`, frame `[1150, 1500)` | `350` frames, `7.00s @ 50Hz` | `9df2a6e567626e4939a95a2609f6d22e61bab868fe2d300ed45f5d0209199488` | `policy/dual_agent_jump_tracking/DualAgentJumpTracking.py` / `DualAgentJumpTracking` | `policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz` | sim2sim `x+l1`; deploy_real `X+L1` if `enable_jump_tracking: true`; current runtime |
| `jumps1_subject2_first75_2m10_2m27_17s__DualAgentJumpTracking__jump_tracking_ref.npz` | `/home/qiuziyu/datasets/gae_mimic_dataset/extend_datasets/lafan1_dataset/g1/train/jumps1_subject2_first75.npz` | `2:10-2:27`, frame `[6500, 7350)` | `850` frames, `17.00s @ 50Hz` | `4e6d2069ccb7c62e8178e36fe67dca275bb17cbadfdb0b9a3c94243fd1f6ff08` | `policy/dual_agent_jump_tracking/DualAgentJumpTracking.py` / `DualAgentJumpTracking` | `policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz` | sim2sim `x+l1`; deploy_real `X+L1` if `enable_jump_tracking: true` |
| `jumps1_subject2_first75_1m45_2m08_23s__DualAgentJumpTracking__jump_tracking_ref.npz` | `/home/qiuziyu/datasets/gae_mimic_dataset/extend_datasets/lafan1_dataset/g1/train/jumps1_subject2_first75.npz` | `1:45-2:08`, frame `[5250, 6400)` | `1150` frames, `23.00s @ 50Hz` | `df8c06f2178a938a2ef82399c417c7da0a8f643e7d49e07eec3ea98e574a4e2b` | `policy/dual_agent_jump_tracking/DualAgentJumpTracking.py` / `DualAgentJumpTracking` | `policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz` | sim2sim `x+l1`; deploy_real `X+L1` if `enable_jump_tracking: true` |

## 使用方式

把当前选中的 run 片段切到部署运行时:

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1
cp reference_motion_bank/selected/run1_subject2_2m15_2m22_7s__DualAgentRunTracking__run_tracking_ref.npz \
   policy/dual_agent_run_tracking/motion/run_tracking_ref.npz
```

把当前选中的 walk 片段切到部署运行时:

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1
cp reference_motion_bank/selected/walk4_subject1_1m17_1m40_23s__DualAgentTracking__walk_tracking_ref.npz \
   policy/dual_agent_tracking/motion/walk_tracking_ref.npz
```

把当前选中的 jump 片段切到部署运行时:

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1
cp reference_motion_bank/selected/jumps1_subject2_first75_0m23_0m30_7s__DualAgentJumpTracking__jump_tracking_ref.npz \
   policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz
```

把当前选中的 sprint 片段切到部署运行时:

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1
cp reference_motion_bank/selected/sprint1_subject2_0m05_0m12_7s__DualAgentDanceTracking__dance_tracking_ref.npz \
   policy/dual_agent_dance_tracking/motion/dance_tracking_ref.npz
```

把 `2:10-2:27` jump 备选片段切到部署运行时:

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1
cp reference_motion_bank/selected/jumps1_subject2_first75_2m10_2m27_17s__DualAgentJumpTracking__jump_tracking_ref.npz \
   policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz
```

恢复完整 run reference:

```bash
cd /home/qiuziyu/code/postman/FSMDeploy_G1
cp policy/dual_agent_run_tracking/motion/run_tracking_ref.full_11890_before_crop_2m15_2m22_2026-05-23.npz \
   policy/dual_agent_run_tracking/motion/run_tracking_ref.npz
```

## 策略读取关系

`DualAgentTracking.yaml` 中的:

```yaml
motion_file: "walk_tracking_ref.npz"
```

决定了 `DualAgentTracking` 只会读取:

```text
policy/dual_agent_tracking/motion/walk_tracking_ref.npz
```

`DualAgentRunTracking.yaml` 中的:

```yaml
motion_file: "run_tracking_ref.npz"
```

决定了 `DualAgentRunTracking` 只会读取:

```text
policy/dual_agent_run_tracking/motion/run_tracking_ref.npz
```

`DualAgentJumpTracking.yaml` 中的:

```yaml
motion_file: "jump_tracking_ref.npz"
```

决定了 `DualAgentJumpTracking` 只会读取:

```text
policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz
```

`DualAgentDanceTracking.yaml` 中的:

```yaml
motion_file: "dance_tracking_ref.npz"
```

决定了当前 `y+l1` sprint tracking 入口只会读取:

```text
policy/dual_agent_dance_tracking/motion/dance_tracking_ref.npz
```

所以本目录是候选库,不是 runtime 路径。后续新增 walk / jump 片段时也保持同样
规则:文件名写清楚来源、时间窗、目标策略和运行时文件名,再在上表追加一行。
