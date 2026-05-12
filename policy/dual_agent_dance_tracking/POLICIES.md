# DualAgent Dance Tracking Policy

Active sim2sim policy:

- Training run: `upper_lower/logs/rsl_rl/g1_dual_agent/2026-05-10_01-16-54_dance2_subject5_stage2_upperjump7000_lowerdance13500`
- Checkpoint: `{upper,lower}_model_18500.pt`
- Motion: `/home/qiuziyu/datasets/gae_mimic_dataset/extend_datasets/lafan1_dataset/g1/train/dance2_subject5.npz`
- Exported ONNX: `model/dual_agent_combined.onnx`
- Deploy motion reference: `motion/dance_tracking_ref.npz`
- MuJoCo sim2sim binding: `y+l1`

The policy uses the same slim dual-agent actor contract as walk/run/jump:
upper obs `96`, lower obs `109`, and semantic lower-body action slots:

```text
[6, 3, 0, 9, 13, 17, 7, 4, 1, 10, 14, 18, 2, 5, 8]
```

This entry is currently wired for sim2sim only. Do not bind it into
`deploy_real.py` before a separate hardware safety ladder.
