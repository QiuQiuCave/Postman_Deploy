# Five-height screening, frozen model_3500 (2026-09-11)

This is **empty-hand path screening**, not successful grasping or training.
All five trials keep the same pinned checkpoint/ONNX, robot state, observation
history, continuous wrist references, open-finger state, crate XY and dimensions.
Only the table/crate height changes: 0, -5, -10, -15, -20 cm.

The new full articulated-hand R2V2 remains free-base, with real inertias, joint
limits and collisions. Body inference 50Hz, fingers 100Hz, physics 1kHz. Critical
contact and body-limit events are checked each substep; detailed measurements
are recorded at 100Hz. No IK, mocap, body/crate weld, external force or runtime
pose reset drives the policy video. Existing training is left running unchanged.

## Outputs

All runtime data are on the data disk:
`/root/autodl-tmp/Postman_Deploy/crate_height_sweep_20260911/`.

- `comparison_body/height_comparison.mp4`: five full-body side views, main video.
- `comparison_hands/height_comparison.mp4`: five hand/crate close-ups.
- `videos/down_00cm` through `videos/down_20cm`: each original two-camera video,
  report, full trace/targets/transitions, initial/final and phase screenshots.
- `planned_paths/down_XXcm`: complete 100Hz **desired** trajectories, including
  the unexecuted suffix, NPZ + manifest. Not successful demonstrations.
- `preparation/prepared_state.json` and `.npz`: common policy-generated start.
- `kinematics_soft90`, `kinematics_approach_soft90`,
  `kinematics_down20_path_soft90`: separate offline static diagnostics.

Comparison videos are 1920×1280, 30fps, 14.60s. All panels use a common elapsed
simulation clock. A safety-stopped trial freezes explicitly; it is never sped
up, extended with new physics, or shown as a successful completion.

Checkpoint SHA256:
`c2f91c3d12acaf0d18004f858a9731a12412159bbfba54dd09d46ea0061be4a8`.
Policy and parity are pinned in
`/root/autodl-tmp/Postman_Deploy/simple_dual_reach_20260911/`.

## Preparation and planned path

In an **empty scene**, the same frozen policy runs HOME for 5s, then 8s toward
world wrists `[.18, +/-.27, 1.13]`, identity quaternion. This is not an approach
among obstacles. Actual final wrists are approximately left
`[.18393,.25414,1.13903]`, right `[.18044,-.22900,1.09794]`: the right retains
about 5.2cm goal error. It is accepted only as a stopped, unassisted starting
state; it is not claimed to be an accurate reach. All five newly initialized
table/crate scenes have no initial robot/prop contacts.

Each new scene starts from this identical physical/controller state, holds 2s
for crate settling, then uses:

`OUTSIDE → TURN_WRISTS → PREALIGN → source READY → INSERT → INSERT_SETTLE → source CLOSE → PROBE_LIFT → HOLD`

Outside wrists are `[.18, +/-.37, 1.21051+delta_z]`, initially identity. The
wrists rotate there to the recorded down20/yaw15 orientation before translating
to source READY. The same measured relative trajectory is relocated through the
settled crate anchor, retaining source crate motion for the lifted wrist goals.
Each segment gets the same extra 2s settling window.

The goal path starts from the **existing smooth reference**, not the biased
actual wrist position. Replacing a goal with the actual pose at this boundary
would suddenly remove 5cm of commanded correction. An initial implementation
with this discontinuity was corrected; those diagnostic outputs remain under
`headless/`, while `continuous_headless/` and all final videos use continuous
reference handoff. No precision or collision threshold was relaxed.

Target-error gates are scoring criteria, not switches, in this empty-hand
screen: inaccurate but safe segments may continue, allowing phase comparison.
Hard safety stops terminate immediately. Fingers remain command **0** throughout;
the recorded CLOSE event is archived only. The upward wrist path therefore
tests unloaded reaching, not closed-hand grasping or carrying the crate.

Continuous conservative open-hand mesh sweeps certify only the added desired
approach geometry (palm, thumb, four fingers), not actual policy tracking or
forearm/torso clearance. Desired turn clearance is about 22.9mm from the box;
several observed wrist errors exceed this margin, explaining real collisions
despite a geometrically clear hand-goal path.

## Observed policy results

| Table offset | Tabletop Z | Stop time | Actual stop |
| --- | ---: | ---: | --- |
| 0 cm | 1.010919 m | 9.695 s | Right middle finger contacts crate during turn |
| -5 cm | 0.960919 m | 9.870 s | Right middle finger contacts crate during turn |
| -10 cm | 0.910919 m | 10.050 s | Right middle finger contacts crate during turn |
| -15 cm | 0.860919 m | 12.550 s | Left ankle roll exceeds hard-limit tolerance |
| -20 cm | 0.810919 m | 12.298 s | Left ankle roll exceeds hard-limit tolerance |

**0/5 complete paths.** Every trial stopped in TURN_WRISTS, before PREALIGN or
insertion. Source-relative insertion/seating/lift paths are generated and
archived but were not physically executed. No grasp was attempted.

Terminal errors are at different stopping times; they are not fair standalone
rankings. At the same still-alive time **t=9s**:

| Offset | Left wrist cm / deg | Right wrist cm / deg | Base tilt |
| --- | ---: | ---: | ---: |
| 0 cm | 3.21 / 23.44 | 5.61 / 8.33 | 6.32deg |
| -5 cm | 3.88 / 19.81 | 5.15 / 7.57 | 7.59deg |
| -10 cm | 4.44 / 16.33 | 4.70 / 12.53 | 8.71deg |
| -15 cm | 4.66 / 14.31 | 3.91 / 19.86 | 9.33deg |
| -20 cm | 5.01 / 14.79 | 4.33 / 29.27 | 9.05deg |

These are moving-path errors, not settled endpoint accuracy. Lowering does not
monotonically improve the current policy. During subsequent instability, -15cm
reaches a peak base tilt of 20.36deg and -20cm 28.43deg; foot-origin excursions
reach 18.4/20.5cm respectively. Neither is a stable planted-foot controller.

## Separate geometry evidence and candidate selection

Offline static IK uses the actual new model and fixed nominal foot frames, with
central 90% joint search bounds while preserving the model's true hard limits.
Four source keyframes (READY, insert, seated, lifted) pass at rates 0/4, 1/4,
3/4, 3/4, 4/4 from high to low. The -15cm lifted pose retains 6.5deg orientation
error. For -20cm, 63 sampled poses over the complete path pass the static screen,
with maximum wrist error 0.112mm/0.246deg and minimum hard-limit margin 6deg.

This only identifies plausible geometry: IK is not the policy driver, its
nominal foot anchor differs from the prepared dynamic trial, contact is checked
after solving rather than optimized away, and sampled clearance does not prove
continuous avoidance or dynamic balance. Even the -20cm lifted solution uses
substantial waist/base yaw, so it should not be called a verified natural motion.

Keep **-15cm and -20cm** as review candidates: -20cm offers stronger geometric
coverage, while this frozen policy becomes less violently unstable at -15cm.
Do not select a final training centre solely from terminal errors or static IK.
No post-training was started; the user reviews the comparison first.

## Reproduce a new trial

```bash
.venv-r2v2/bin/python deploy_mujoco/r2v2_crate_height_sweep.py \
  --reach-config /root/autodl-tmp/Postman_Deploy/simple_dual_reach_20260911/policy/reach_wrist_v2.yaml \
  --parity-report /root/autodl-tmp/Postman_Deploy/simple_dual_reach_20260911/parity/report.json \
  --prepared-state /root/autodl-tmp/Postman_Deploy/crate_height_sweep_20260911/preparation/prepared_state.json \
  --height-offset -0.20 \
  --output /root/autodl-tmp/Postman_Deploy/crate_height_sweep_20260911/new_trial
```

`tools/prepare_r2v2_crate_height_sweep.py` regenerates a new common start;
`tools/export_r2v2_crate_height_paths.py` archives complete desired paths;
`tools/compose_r2v2_crate_height_comparison.py --view body|hands` builds mosaics.
All output tools reject nonempty destinations.

Verification: 158 focused and existing regression tests passed. For all five
heights, rendered trace, targets and transitions match continuous-headless
results byte-for-byte. Checkpoint, ONNX and prepared-state hashes are identical;
all logged finger commands are zero. Videos were probed and inspected.
