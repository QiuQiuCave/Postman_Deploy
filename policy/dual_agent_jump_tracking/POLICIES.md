# dual_agent_jump_tracking - policy artifact ledger

`DualAgentJumpTracking` is the LAFAN jump demo counterpart to
`DualAgentTracking` and `DualAgentRunTracking`. It is bound to `x+l1` for MuJoCo
sim2sim testing and to `X+L1` in `deploy_real.py` behind
`enable_jump_tracking`.

## ONNX checkpoints (`model/`)

| File | Origin | Date | Status |
|---|---|---|---|
| `dual_agent_combined.onnx` | upper/lower iter 7000 from `upper_lower/logs/rsl_rl/g1_dual_agent/2026-05-06_17-38-04_jumps1_subject2_first75_stage2_from_stage1_15000_semantic` | 2026-05-06 | **active** - picked up by `DualAgentJumpTracking.py`. |

## Motion references (`motion/`)

| File | Source motion | Date | Status |
|---|---|---|---|
| `jump_tracking_ref.npz` | `jumps1_subject2_first75`, `0:23-0:30`, frame `[1150, 1500)`, 350 frames / 7.00 s @ 50 Hz, SHA256 `9df2a6e567626e4939a95a2609f6d22e61bab868fe2d300ed45f5d0209199488` | 2026-05-23 | **active sim2real short clip** — mirrored from `reference_motion_bank/selected/jumps1_subject2_first75_0m23_0m30_7s__DualAgentJumpTracking__jump_tracking_ref.npz`. |
| `jump_tracking_ref.full_9166_before_crop_1m45_2m08_2026-05-23.npz` | full `jumps1_subject2_first75` runtime reference before the deploy crops, 9166 frames / ~183.3 s @ 50 Hz | 2026-05-23 | **backup** — restore this if the full jump loop is needed again. |

Earlier selected but inactive jump candidates remain in
`reference_motion_bank/selected/`:

- `1:45-2:08`, 1150 frames / 23.00 s
- `2:10-2:27`, 850 frames / 17.00 s

MuJoCo box spawn uses the policy-specific pelvis-frame offset
`(0.32, 0.0, 0.22)`, matching the current run-tracking sim2sim offset.

This checkpoint uses the semantic lower-body action layout from training:
`[6, 3, 0, 9, 13, 17, 7, 4, 1, 10, 14, 18, 2, 5, 8]`. The ONNX export and
runtime `last_action` feedback must both use this layout.
