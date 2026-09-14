# Recorded oblique crate motion → full-body wrist policy

This experiment records the successful **bilateral** hand-only fixture run:
downward tilt 20°, inward yaw 15°, 60 mm insertion, 0.8 rad four-finger curl,
30 mm closure seating and 100 mm lift. Crate: 26 × 24 × 16 cm, openings
12 × 5.5 cm, mass 0.4 kg. The fixture baseline remains unchanged.

## Portable motion bank

`reference_motion_bank/r2v2_crate/down20_yaw15_60mm/` contains:

- `trajectory.npz`: 1,285 unique 100 Hz samples spanning 12.84 s (~1.1 MiB).
- `manifest.json`: units, conventions, exact phase/command events, hand and
  crate configurations, success evidence and source/trajectory SHA256 hashes.

The recording includes actual crate/world/wrist transforms, both wrist paths
relative to the moving crate and to the frozen initial crate, fixture target
poses separately, binary commands, six driven finger references, measurements
and commanded torques per hand. Missing finger velocities and mimic-state
signals are explicitly identified rather than synthesized. The terminal
duplicate timestamp retains its last COMPLETE row; original event times remain.

```python
from common.r2v2_crate_motion_recording import load_crate_motion

motion = load_crate_motion("reference_motion_bank/r2v2_crate/down20_yaw15_60mm")
sample = motion.sample(t)  # linear position + shortest WXYZ quaternion SLERP
world_wrists = motion.world_wrist_targets(t, world_anchor, use_fixture_targets=False)
```

`T_A_B` maps column coordinates from B to A. Crate origin is its outside bottom
centre, X front/back, Y left/right, Z up. Wrist means the actual
`left_hand_roll_link` / `right_hand_roll_link` origin and axes, not an offset TCP.

## Reconstruction and control

```text
T_world_crate_desired(t) = T_world_anchor × T_anchor_crate_recorded(t)
T_world_wrist_goal(t) = T_world_crate_desired(t) × T_crate_wrist_recorded(t)
```

The new scene anchor is calibrated once against the actual resting crate and
the source's settled crate pose. The full-body replay uses **measured** source
wrist trajectories, not fixture mocap targets. Keeping the recorded crate
motion is essential: after grasp, the relative wrist pose is nearly constant;
multiplying only the live crate pose by that pose would erase the lift command.

Targets stream at 50 Hz through the policy's existing world-target interface.
Updating goals does not reset its smooth reference or observation history.
The training speed conditions remain unchanged. Finger control is independent:
the recorded binary close event starts the same 100 Hz Ruckig trajectory, with
1 kHz torque/contact integration. Recorded qpos or torques are never forced
onto the physical robot. Stage gates can stretch time; this is not a guarantee
of matching the fixture's timing or physical tracking precision.

## Full-body experiment

Entry point: `deploy_mujoco/r2v2_crate_motion_replay.py`.

```bash
.venv-r2v2/bin/python deploy_mujoco/r2v2_crate_motion_replay.py \
  --motion reference_motion_bank/r2v2_crate/down20_yaw15_60mm \
  --reach-config /root/autodl-tmp/Postman_Deploy/crate_motion_replay_20260911/policy/reach_wrist_v2.yaml \
  --parity-report /root/autodl-tmp/Postman_Deploy/crate_motion_replay_20260911/parity/report.json \
  --output /root/autodl-tmp/Postman_Deploy/crate_motion_replay_20260911/new_trial
```

Use a new/empty output directory. `--no-video` saves measurements only;
`--top-view` changes only the right camera. Default video has a fixed exact
side overview and a fixed close-up, with target markers clearly distinguished
from real wrists. Two final frozen seconds are explicitly not physical time.

The pinned wrist-v2 checkpoint is `model_3249.pt`. Native/deploy parity passed
240 samples and two resets: maximum same-observation action difference
2.86e-6, full observation/history difference 1.67e-6, end-to-end action
difference 1.12e-5. Iteration 3249 has just entered stage 4 / 70%; the last
completed evaluation was stage 3 / 50%, not a passed 70% evaluation.

The scene preserves the new articulated-hand robot, real limits, inertia,
collision and 10 finger mimic constraints: nq=64, nv=62, nu=40. No wrist
fixtures, mocap, body/crate welds, IK, external forces or rollout pose resets.
Native initial default body q is clamped to the same 90% soft range at time
zero only; physical limits are unchanged.

Stages: RESET_SETTLE → STAND → PREALIGN → recorded READY → INSERT →
INSERT_SETTLE → CLOSE → PROBE_LIFT → HOLD → COMPLETE. Preparation is a short
2 s stability probe, **not** the earlier 20 s deployment certification.
Prealign requires 2 cm/10°; insertion requires 5 mm/3°, wrist speed <2 cm/s,
continuous stability 0.3 s. Each stage times out at 10 s. Safety is checked at
100 Hz, with immediate nonfinite/warning termination; real body limits use
0.01 rad violation and self penetration 2 mm. No relaxation to force a video.
Close requires actual insertion, lift requires bilateral contact, and final
success requires real clearance, load, low slip, level crate and settled wrists
for 2 s. A failed loaded grasp never automatically opens the hands.

## Observed first attempt (2026-09-11)

The direct default-pose start **failed at 0.10 s**, before trajectory playback:
the forward open fingers overlap the near crate/table workspace and the left
pinky makes table contact. This is an initialization/path issue, not proof
that the oblique grasp itself fails. The safety stop was retained, with all
fingers still commanded open; no grasp/lift success is claimed.

Evidence: data disk folder `crate_motion_replay_20260911/full_scene_collision/`
contains the actual video, initial/final and phase screenshots, report, trace,
transitions and emitted targets. Free-space policy prealignment is investigated
separately; it has no physical crate/table and cannot establish grasp success.
If prealignment succeeds, a subsequent scene must be initialized with its
complete continuous body/controller state, then checked for collisions. Such
a two-scene test is not a demonstrated obstacle-avoiding approach from HOME.

Training, the old fixture baseline and the shared Reach adapter are not changed
by this experiment. Results and videos are stored on the data disk.

### Free-space prealignment diagnosis

`deploy_mujoco/r2v2_crate_motion_prepare.py` accepts the same required CLI
arguments (`--motion`, `--reach-config`, `--parity-report`, `--output`). It runs
the real free-base policy in an empty scene, with a **render-only** crate outline.
It retains the 1 kHz safety checks of the Reach compatibility runner. The
2 s preparatory standing interval is a short survival check, not certification.
Only a successful prealignment would export a prepared state for a new trial.

The native-soft-limit-aligned run also **failed**, at 5.209 s (2.809 s after
starting the source READY goal). Left ankle roll reached 0.35911372 rad against
the unchanged physical upper limit 0.349066 rad: excess 0.01004772 rad crossed
the 0.01 rad safety tolerance. At the stop:

| Metric | Left | Right |
| --- | ---: | ---: |
| Wrist position error | 121.56 mm | 149.33 mm |
| Wrist orientation error | 41.31° | 27.96° |

Maximum base tilt: 13.897°. No deep self penetration or finger-limit failure.
No prepared state was exported; insertion, closure and lifting were **not
executed**. The initial uncorrected reset gave the same failure pattern at
5.171 s; both runs remain on disk. This is not a successful full-body replay.

Latest diagnostic artifacts:
`/root/autodl-tmp/Postman_Deploy/crate_motion_replay_20260911/free_space_prepare_softlimit/`.
Video: `free_space_prepare.mp4`, 1280 × 800, 30 fps, 7.267 s including the final
2 s frozen state. The yellow crate outline provides spatial context only;
it cannot provide contact forces or establish manipulation success.

The recorded READY targets are approximately left/right
`[0.31355, ±0.37210, 1.21049] m`. Compared with the current 70% course's nominal
PREALIGN targets, these are about 10.2 cm higher, 4.85 cm farther out per side,
2.63 cm closer in X, and 23.97° different in orientation. These differences
support an out-of-distribution hypothesis, but do not by themselves separate
training coverage from articulated-hand sim2sim dynamics. Any future training
change should include the full outside approach, insertion, seating and lifted
poses, not just the final insertion target. No new training was started here.

Verification: 150 focused recording/replay/scene and existing Reach-policy/
parity regression tests passed. Video files were decoded/probed and their
terminal screenshots inspected. Neither a failed prealignment nor unit tests
validate the unexecuted grasp/lift phases.
