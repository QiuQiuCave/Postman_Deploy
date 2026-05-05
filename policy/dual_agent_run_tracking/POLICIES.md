# dual_agent_run_tracking — policy artifact ledger

`DualAgentRunTracking` is the run demo counterpart to `DualAgentTracking`.
It keeps walk on `a+l1` and binds this run policy to `b+l1` for MuJoCo
sim2sim A/B testing.

## ONNX checkpoints (`model/`)

| File | Origin | Date | Status |
|---|---|---|---|
| `dual_agent_combined.onnx` | upper/lower iter 13000 from `upper_lower/logs/rsl_rl/g1_dual_agent/2026-05-05_02-02-39_run1_subject2_resume2500` | 2026-05-05 | **active** — picked up by `DualAgentRunTracking.py`; verified in Isaac play before sim2sim integration. |

## Motion references (`motion/`)

| File | Source motion | Date | Status |
|---|---|---|---|
| `run_tracking_ref.npz` | `/home/qiuziyu/datasets/gae_mimic_dataset/extend_datasets/lafan1_dataset/g1/train/run1_subject2.npz` (50 fps, 11890 frames, ~237.8 s) | 2026-05-05 | **active** — preprocessed by `upper_lower/scripts/factoryIsaac/dual_agent_tracking_preprocess_motion.py`. |

The run policy is currently sim2sim-only. Do not bind it in `deploy_real.py`
without a separate hardware validation ladder.

MuJoCo box spawn uses the policy-specific pelvis-frame offset
`(0.32, 0.0, 0.22)`, which is 8 cm higher than the default walk/box offset.
