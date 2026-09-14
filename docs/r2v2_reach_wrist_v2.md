# R2V2 wrist-world v2 deployment contract (simulation only)

This interface belongs to AMO_R2 task `R2V2-Reach-CrateWrist-v2-28DoF`.
It does **not** replace the legacy TCP policy or authorize a real-robot run.
The ONNX shape is still `obs[1,1460] -> actions[1,28]`, but endpoint position,
orientation, reference velocity and history now mean the true wrist frame.
An old actor must not simply be renamed or have v2 metadata attached to it.

## Export metadata and model sites

Required v2 ONNX metadata, in addition to the existing observation/action/PD
metadata checked by `ReachPolicy`:

```text
endpoint_contract = wrist_world_v2
quaternion_order = wxyz
speed_reference_point = wrist
endpoint_body_names = left_hand_roll_link,right_hand_roll_link
joint_names = <the existing ordered 28 BODY_JOINTS, comma-separated>
```

`left_wrist` and `right_wrist` sites must be attached directly to the matching
`*_hand_roll_link`, with exactly zero local position and identity rotation.
The adapter rejects missing sites, shifted axes/origins, unknown versions,
incomplete v2 metadata, and mismatched configured/ONNX versions. Old exports
without version metadata keep `legacy_tcp_v1` and the original `*_tcp` sites.
The v2 scene contains only wrist endpoint sites, so old exports cannot silently
use it. It still contains the complete new robot with independent hand servos;
the 28-body action bus never writes the 12 independent hand actuator channels.

## World goal API

```python
policy = ReachPolicy(model, data, onnx_path,
                     expected_endpoint_contract="wrist_world_v2")
policy.set_target_world("left", wrist_position_world, wrist_quaternion_wxyz)
```

Calls change the final goal without resetting the ongoing smooth trajectory.
For v2, no 17.35 cm legacy TCP offset is added. `set_wrist_target_world` is an
explicit cross-version helper: v2 is the identity conversion; legacy applies
its calibrated wrist-to-TCP transform once. Do not pre-add the legacy offset
before calling either v2 setter.

`wrist_pose(data, side)` reads the link origin and axes. `wrist_twist` uses the
body-origin Jacobian, not COM velocity. Debug RPY uses extrinsic XYZ
`Rz(yaw) @ Ry(pitch) @ Rx(roll)` via `rotation_from_rpy_deg`; internal rotations
are normalized WXYZ quaternions. Policy runs at 50 Hz, hand trajectories at
100 Hz, physics/PD at 1 kHz. Smooth-reference bounds remain 0.25 m/s and
0.8 rad/s (0.8 m/s², 2 rad/s² acceleration); these are not hard bounds on
measured physical wrist motion.

## Native training/deployment parity

Use an explicitly exported v2 checkpoint/ONNX pair, separate output directory,
and the training task's own native scene:

```bash
.venv-r2v2/bin/python tools/verify_r2v2_reach_policy.py \
  --task R2V2-Reach-CrateWrist-v2-28DoF \
  --checkpoint /absolute/path/model_N.pt \
  --onnx /absolute/path/policy_N.onnx \
  --output /root/autodl-tmp/AMO_R2/parity/wrist_v2_N \
  --frames 120
```

The native collector uses AMO_R2's interpreter and may use its GPU; add
`--device cpu` only if that training environment supports CPU simulation.
It preserves v2 safety terminations and fails closed if the rollout terminates.
It tests same-observation actor output, per-term history assembly and end-to-end
output from identical qpos/qvel. It uses the exact exported frozen-open-hand
training scene with new-model meshes, not the old training body or a reconstructed
deployment scene with a different joint count. Numerical parity is not proof of
20-second standing, table collision clearance, grasp or loaded transport success.

`deploy_mujoco/config/r2v2_reach_wrist_v2.yaml` is an unwired validation template.
Set its checkpoint/ONNX paths only after a matching parity report passes. The
existing validation entrypoint checks report hashes and endpoint contract. Its
air-path probe is separate from the new crate curriculum, not crate acceptance.
Old tabletop/grasp/crate configurations continue to use their legacy policies.

Changing the shared adapter invalidates earlier adapter SHA-256 parity reports;
re-run verification rather than bypassing the freshness check.
