# DualAgentJumpTracking Sim2Sim Integration

Date: 2026-05-06

This note records the jump tracking deployment path and the action-index issue
that was fixed before the MuJoCo validation succeeded.

## Summary

`DualAgentJumpTracking` is a MuJoCo-only tracking demo bound to `x+l1`.

It mirrors the existing walk/run tracking runtime:

- upper ONNX input: `upper_obs`, 96D
- lower ONNX input: `lower_obs`, 109D
- ONNX output: `actions`, 29D in MuJoCo motor order
- motion reference: lower-body joint pos/vel plus torso anchor, 50 Hz
- box handling: shared `transport_box`, spawned by the deploy loop for policies
  with `needs_transport_box = True`

Active artifacts:

```text
policy/dual_agent_jump_tracking/model/dual_agent_combined.onnx
policy/dual_agent_jump_tracking/motion/jump_tracking_ref.npz
```

Source checkpoint:

```text
upper_lower/logs/rsl_rl/g1_dual_agent/
2026-05-06_17-38-04_jumps1_subject2_first75_stage2_from_stage1_15000_semantic/
upper_model_7000.pt
lower_model_7000.pt
```

Source motion:

```text
/home/qiuziyu/datasets/gae_mimic_dataset/extend_datasets/
lafan1_dataset/g1/train/jumps1_subject2_first75.npz
```

## FSM Routing

New enum values:

```python
FSMStateName.DUAL_AGENT_JUMP_TRACK = 19
FSMCommand.DUAL_AGENT_JUMP_TRACK = 19
```

New runtime class:

```text
policy/dual_agent_jump_tracking/DualAgentJumpTracking.py
```

`DualAgentJumpTracking` subclasses `DualAgentRunTracking` and only overrides
the policy directory, YAML filename, FSM state name, and display name. The
shared runtime handles ONNX loading, motion stepping, obs construction, ramp-in,
action reorder, and transitions between walk/run/jump.

Keyboard / joystick routing:

- `a+l1`: walk tracking
- `b+l1`: run tracking
- `x+l1`: jump tracking
- `y+l1`: still reserved

`FixedPose` can now transition directly into walk/run/jump. This matters for
the keyboard flow `start -> fixed_pose -> x+l1`.

## Semantic Lower Action Layout

The validated jump checkpoint was trained with semantic lower-body action
indices. These are recorded in:

```yaml
policy/dual_agent_jump_tracking/config/DualAgentJumpTracking.yaml
lower_action_indices: [6, 3, 0, 9, 13, 17, 7, 4, 1, 10, 14, 18, 2, 5, 8]
```

This differs from older walk/run artifacts:

```yaml
lower_action_indices: [0, 1, 2, ..., 14]
```

The deploy runtime uses `lower_action_indices` for `last_action_lower`:

```python
last_action_lower = self.action_isaac[self.lower_action_indices]
```

The first jump ONNX export failed in MuJoCo because the exporter still wrote the
15-D lower action into Isaac slots `0..14`. That was correct for old walk/run
but wrong for semantic jump. After the exporter was fixed and the ONNX was
re-exported, MuJoCo no longer showed the full-body action scramble.

Future deployment rule:

1. Check `params/env.yaml` in the training run for
   `observations.lower_body_policy.actions.params.action_indices`.
2. Export with `dual_agent_export_onnx.py --action_layout auto`.
3. Copy the same lower action indices into the matching deploy YAML.
4. Re-run policy smoke and MuJoCo keyboard validation.

## Box Mass

The shared MuJoCo `transport_box` is currently:

```xml
<geom type="box" size="0.15 0.15 0.15" mass="0.5" .../>
```

This is deploy-side only. Isaac training mass/randomization is controlled in
`upper_lower`, not by `g1_description/scene.xml`.

## Validation

Checks performed before this commit:

```bash
python -m py_compile \
  common/utils.py FSM/FSM.py \
  policy/dual_agent_run_tracking/DualAgentRunTracking.py \
  policy/dual_agent_tracking/DualAgentTracking.py \
  policy/dual_agent_jump_tracking/DualAgentJumpTracking.py \
  deploy_mujoco/deploy_mujoco.py \
  deploy_mujoco/deploy_mujoco_keyboard_input.py

git diff --check
```

Runtime smoke:

```bash
uv run python - <<'PY'
import numpy as np
from common.ctrlcomp import StateAndCmd, PolicyOutput
from policy.dual_agent_jump_tracking.DualAgentJumpTracking import DualAgentJumpTracking

state = StateAndCmd(29)
out = PolicyOutput(29)
policy = DualAgentJumpTracking(state, out)
assert policy.lower_action_indices.tolist() == [6, 3, 0, 9, 13, 17, 7, 4, 1, 10, 14, 18, 2, 5, 8]
print("jump runtime smoke ok")
PY
```

MuJoCo keyboard validation:

```bash
uv run python deploy_mujoco/deploy_mujoco_keyboard_input.py
# start
# x+l1
```

The verified log showed:

```text
Switched to dual_agent_jump_tracking_mode
DualAgentJumpTracking: ramp complete, starting policy inference.
BoxTransport: spawned box, pinned for 1.0s.
BoxTransport: released gravity hold.
```
