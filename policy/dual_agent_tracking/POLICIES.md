# dual_agent_tracking — policy artifact ledger

This skill (FSM state `DualAgentTracking`) consumes a single combined ONNX
plus a preprocessed motion reference. To support side-by-side evaluation of
new candidates, multiple checkpoints are kept here with descriptive names.
Only `dual_agent_combined.onnx` and `walk_tracking_ref.npz` (no version
suffix) are loaded at runtime — everything else is an archive / candidate.

## ONNX checkpoints (`model/`)

| File | Origin | Date | Status |
|---|---|---|---|
| `dual_agent_combined.onnx` | upper/lower iter 3500 from `upper_lower/logs/rsl_rl/g1_dual_agent/2026-05-07_12-04-19_walk1_subject2_stage2_semantic_upperjump7000_lowerrun10000` | 2026-05-08 | **active** — semantic action-layout export for `DualAgentTracking.py`; selected because the later walk training run diverged after this stable window. |
| `dual_agent_combined.legacy_iter15000_2026-04-22_before_semantic_walk3500_2026-05-08.onnx` | previous active walk policy, production-era iter 15000 lineage | 2026-04-22 | **backup** — pair with `config/DualAgentTracking.legacy_iter15000_2026-04-22.yaml` because it uses the legacy lower action layout. |
| `dual_agent_combined.f00418b_iter15000_2026-04-22.onnx` | upper_lower commit `f00418b`, run `2026-04-21_20-01-44_joint_train`, iter 15000 | 2026-04-22 | **production baseline** — verified sim2sim (deploy_mujoco) AND sim2real (G1 hardware). Slim obs (upper 96-dim, lower 109-dim, both single-frame). Do not delete. |
| `dual_agent_combined.c63a653_iter15000_2026-04-26.onnx` | upper_lower commit `c63a653` (per-task config separation cleanup), run `2026-04-26_01-33-28_joint_train`, iter 15000 | 2026-04-26 | **byte-identical to f00418b** (md5 `63e442aef1644bf3bfb509e0a0447bec` matches; underlying `.pt` checkpoints also match across all sampled iters: 500, 15000). Confirms cleanup is a true functional no-op AND that PPO training is fully deterministic under fixed seed. No re-verification needed — physically the same bytes as production. |
| `dual_agent_combined.old_121_480.onnx` | Pre-obs-reduction era (before commit `e33e437`). Lower 121-dim single, upper 96×5=480-dim with history. | 2026-04-21 | **legacy** — superseded by the slim-obs production version above. Kept for reference only. |

## Motion references (`motion/`)

The motion reference is a preprocessed extract of the LAFAN walk clip:
15 lower-body joint pos/vel in the semantic lower-body order + torso_link
world pose kept for audit/preview. The current runtime file is a manually
selected short deploy segment from `data/demo/lafan/walk4_subject1.npz`.

| File | Source motion | Date | Status |
|---|---|---|---|
| `walk_tracking_ref.npz` | `walk4_subject1`, `1:17-1:40`, frame `[3850, 5000)`, 1150 frames / 23.00 s @ 50 Hz, SHA256 `8c970e646fa59af71dd9e95959d0098496b4af81b08a6bd4ee82028b08dcfd55` | 2026-05-23 | **active sim2real short clip** — mirrored from `reference_motion_bank/selected/walk4_subject1_1m17_1m40_23s__DualAgentTracking__walk_tracking_ref.npz`. |
| `walk_tracking_ref.full_8195_before_crop_1m17_1m40_2026-05-23.npz` | full LAFAN walk4_subject1 runtime reference before the deploy crop | 2026-05-23 | **backup** — restore this if the full 163.9 s loop is needed again. |
| `walk_tracking_ref.f00418b_2026-04-21.npz` | LAFAN walk4_subject1 (50 fps, 8195 frames, ~163.9s loop) | 2026-04-21 | **production** — paired with the f00418b ONNX. Source motion file unchanged since, so this byte-equivalent backup serves as the canonical reference. |

The current active checkpoint uses the same semantic lower-body action layout as
the semantic run/jump tracking policies:
`[6, 3, 0, 9, 13, 17, 7, 4, 1, 10, 14, 18, 2, 5, 8]`. The ONNX export and
runtime `last_action` feedback must both keep this layout.

## Promotion procedure (candidate → active)

When a new candidate ONNX clears sim2sim verification:

1. Confirm the candidate is in this directory with a descriptive
   `.commit_iterN_YYYY-MM-DD.onnx` name.
2. Replace the active file: `cp <candidate>.onnx dual_agent_combined.onnx`
3. Update the table above: move the previous active line to "production
   baseline" status (or "superseded" if there's already a production), and
   mark the new active.
4. Commit the change to git so the FSMDeploy_G1 repo tracks the rotation.
5. If sim2real verification passes after sim2sim, update successful_runs.md
   in the trainer repo with the new run path and commit hash.

## Why we keep multiple ONNX files here

The runtime FSM only needs one. We keep older versions because:
- **Rollback safety**: if a newly-promoted policy misbehaves on real
  hardware, we can `cp` the previous one back to active without re-exporting.
- **A/B comparison**: when iterating on training changes, comparing the new
  candidate against the previous production in identical sim2sim conditions
  isolates whether the training change actually helped.
- **Provenance audit**: filenames carry commit hash + iter + date so we can
  always trace back to the exact training run that produced any deployed
  policy. Without this we'd be one accidental overwrite away from losing the
  link.
