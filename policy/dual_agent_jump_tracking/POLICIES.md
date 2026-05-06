# dual_agent_jump_tracking - policy artifact ledger

`DualAgentJumpTracking` is the LAFAN jump demo counterpart to
`DualAgentTracking` and `DualAgentRunTracking`. It is bound to `x+l1` for MuJoCo
sim2sim testing.

## ONNX checkpoints (`model/`)

| File | Origin | Date | Status |
|---|---|---|---|
| `dual_agent_combined.onnx` | upper/lower iter 7000 from `upper_lower/logs/rsl_rl/g1_dual_agent/2026-05-06_17-38-04_jumps1_subject2_first75_stage2_from_stage1_15000_semantic` | 2026-05-06 | **active** - picked up by `DualAgentJumpTracking.py`. |

## Motion references (`motion/`)

| File | Source motion | Date | Status |
|---|---|---|---|
| `jump_tracking_ref.npz` | `/home/qiuziyu/datasets/gae_mimic_dataset/extend_datasets/lafan1_dataset/g1/train/jumps1_subject2_first75.npz` (50 fps, 9166 frames, ~183.3 s) | 2026-05-06 | **active** - preprocessed by `upper_lower/scripts/factoryIsaac/dual_agent_tracking_preprocess_motion.py`. |

MuJoCo box spawn uses the policy-specific pelvis-frame offset
`(0.32, 0.0, 0.22)`, matching the current run-tracking sim2sim offset.

This checkpoint uses the semantic lower-body action layout from training:
`[6, 3, 0, 9, 13, 17, 7, 4, 1, 10, 14, 18, 2, 5, 8]`. The ONNX export and
runtime `last_action` feedback must both use this layout.
