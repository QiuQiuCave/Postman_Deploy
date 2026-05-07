# dual_agent_run_tracking — policy artifact ledger

`DualAgentRunTracking` is the run demo counterpart to `DualAgentTracking`.
It keeps walk on `a+l1` and binds this run policy to `b+l1` for MuJoCo
sim2sim A/B testing and to `B+L1` in `deploy_real.py`.

## ONNX checkpoints (`model/`)

| File | Origin | Date | Status |
|---|---|---|---|
| `dual_agent_combined.onnx` | upper/lower iter 10000 from `upper_lower/logs/rsl_rl/g1_dual_agent/2026-05-07_00-04-09_run1_subject2_stage2_semantic_from_jump7000` | 2026-05-07 | **active** — semantic action-layout export for `DualAgentRunTracking.py`; exported after the later training run numerically diverged. |
| `dual_agent_combined.legacy_iter13000_2026-05-05_before_semantic_run10000_2026-05-07.onnx` | upper/lower iter 13000 from `upper_lower/logs/rsl_rl/g1_dual_agent/2026-05-05_02-02-39_run1_subject2_resume2500` | 2026-05-05 | **backup** — previous active run policy; pair with `config/DualAgentRunTracking.legacy_iter13000_2026-05-05.yaml` because it uses the legacy lower action layout. |

## Motion references (`motion/`)

| File | Source motion | Date | Status |
|---|---|---|---|
| `run_tracking_ref.npz` | `/home/qiuziyu/datasets/gae_mimic_dataset/extend_datasets/lafan1_dataset/g1/train/run1_subject2.npz` (50 fps, 11890 frames, ~237.8 s) | 2026-05-05 | **active** — preprocessed by `upper_lower/scripts/factoryIsaac/dual_agent_tracking_preprocess_motion.py`. |

The run policy is wired into both MuJoCo sim2sim (`b+l1`) and deploy_real
(`B+L1`). Treat the real-robot path as an experimental entry until it clears a
separate hardware validation ladder.

MuJoCo box spawn uses the policy-specific pelvis-frame offset
`(0.32, 0.0, 0.22)`, which is 8 cm higher than the default walk/box offset.

The current active checkpoint uses the same semantic lower-body action layout as
`DualAgentJumpTracking`:
`[6, 3, 0, 9, 13, 17, 7, 4, 1, 10, 14, 18, 2, 5, 8]`. The ONNX export and
runtime `last_action` feedback must both keep this layout.
