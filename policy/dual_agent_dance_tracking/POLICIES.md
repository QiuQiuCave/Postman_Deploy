# DualAgent Dance/Sprint Tracking Policy

Active sim2sim policy:

- Training run: `upper_lower/logs/rsl_rl/g1_dual_agent/2026-05-23_08-41-53_sprint1_subject2_stage2_shaped_from_stage1_5000_upperjump7000`
- Checkpoint: `{upper,lower}_model_5000.pt`
- Motion source: `/home/qiuziyu/datasets/gae_mimic_dataset/extend_datasets/lafan1_dataset/g1/train/sprint1_subject2.npz`
- Active deploy segment: `0:05-0:12`, frame `[250, 600)`, 350 frames / 7.00 s @ 50 Hz
- Exported ONNX: `model/dual_agent_combined.onnx`
- Deploy motion reference: `motion/dance_tracking_ref.npz`
- Selected-bank mirror: `reference_motion_bank/selected/sprint1_subject2_0m05_0m12_7s__DualAgentDanceTracking__dance_tracking_ref.npz`
- Motion SHA256: `1f2f10098c2fe0aea93b09afcf2211815e4378f84993ab0feb6a8ae190a6e2db`
- MuJoCo sim2sim binding: `y+l1`
- Export scratch dir: `upper_lower/logs/sim2sim_exports/sprint1_subject2_stage2_5000_y_l1_2026-05-23`

The policy directory and FSM state still use the historical
`dual_agent_dance_tracking` names so the existing `y+l1` binding can be reused.
The active artifact content is currently sprint tracking, not dance.

Backups kept in this policy directory:

- `model/dual_agent_combined.dance2_subject5_iter18500_backup_2026-05-23.onnx`
- `motion/dance_tracking_ref.dance2_subject5_backup_2026-05-23.npz`
- `motion/dance_tracking_ref.full_13655_before_crop_0m05_0m12_2026-05-23.npz`

The policy uses the same slim dual-agent actor contract as walk/run/jump:
upper obs `96`, lower obs `109`, and semantic lower-body action slots:

```text
[6, 3, 0, 9, 13, 17, 7, 4, 1, 10, 14, 18, 2, 5, 8]
```

This entry is currently wired for sim2sim only. Do not bind it into
`deploy_real.py` before a separate hardware safety ladder.
