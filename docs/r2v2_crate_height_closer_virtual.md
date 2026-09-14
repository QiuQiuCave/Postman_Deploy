# Closer virtual-prop height comparison (2026-09-11)

This is a **no-contact, empty-hand policy path preview**, not a grasp or loaded
lift experiment. It follows the physical five-height screen documented in
`r2v2_crate_height_sweep.md` without changing its pinned model_3500, ONNX,
policy-prepared robot state, observation history, hand profile, timing or cameras.
No new training is started and the existing training is not modified.

## Requested scene change

- Table and crate move together by world X **-0.05 m**, toward the robot.
- Crate bottom-frame XY changes from `[.38, 0]` to `[.33, 0]` metres; table
  centre XY changes from `[.48, 0]` to `[.43, 0]` metres.
- Heights remain the same five offsets: 0, -5, -10, -15, -20 cm relative to
  tabletop Z=1.0109189696536514 m.
- Crate dimensions remain 26 cm left/right × 24 cm front/back × 16 cm high;
  handle openings remain 12 × 5.5 cm.
- The original starting wrist references are unchanged. The outside/turn
  waypoints and subsequent crate-relative path move 5 cm toward the body.
  OUTSIDE therefore smoothly transitions to X=.13 m instead of .18 m, with
  Y=+/-.37 m. No initial reference jump or runtime robot pose reset is used.

## Meaning of virtual props

Both table (including legs) and crate are static, invisible model geometry with
**contype=0 and conaffinity=0**. The crate has no free joint and cannot fall,
move with the hands, support the robot or impart contact forces. Render-only
`mjGEOM_LINE` outlines show the table in amber and the crate/handle components
in cyan. Rounded beam outlines use mesh-local bounding boxes; these lines are
placeholders, not exact rounded-surface contact visualizations.

Robot model inertias, true joint limits, self-collisions, hand mimic joints,
actuators and real foot/floor physics are unchanged. Model dimensions become
`nq=57, nv=56, nu=40, neq=10, nmocap=0`, solely because the free crate is removed.
There are no support forces, IK controls, wrist fixtures or added welds.

Passing through the table/crate no longer creates a force or a stop. The desired
hand sweep geometry is still measured for context, but **does not gate playback**
in this mode. Existing robot-limit, self-contact, fall/ground-contact and
numerical-safety stops remain active. Tracking errors remain measurements, not
phase gates; each segment retains its original duration plus 2 s settling.

All fingers remain command **0**. Source CLOSE is a phase label and archived
source command only. Even if the upward wrist goals execute, the cyan crate
stays at its initial location: it is not animated to imply a successful lift.
Contact/load/slip/lift are not physically evaluated; report flags distinguish
this from the original physical experiment. Zero prop contact is by construction,
not evidence of collision-free reach or successful grasping.

## Outputs and reproduction

Output root on the data disk:
`/root/autodl-tmp/Postman_Deploy/crate_height_sweep_closer5_virtual_20260911/`.

- `videos/down_XXcm/`: original exact-side + hand-close-up video, trace,
  world targets, transitions and report for each height.
- `comparison_body/height_comparison.mp4`: five-way full-body side comparison.
- `comparison_hands/height_comparison.mp4`: five-way hand/crate close-up.
- `planned_paths/down_XXcm/`: complete desired paths, not executed-success data.

Each independent scene restores the same robot/controller state at t=0. All
five videos use a common elapsed simulation clock. Failed scenes freeze with
an explicit label; no extra physics or retiming is used to prolong them.

```bash
.venv-r2v2/bin/python deploy_mujoco/r2v2_crate_height_sweep.py \
  --reach-config /root/autodl-tmp/Postman_Deploy/simple_dual_reach_20260911/policy/reach_wrist_v2.yaml \
  --parity-report /root/autodl-tmp/Postman_Deploy/simple_dual_reach_20260911/parity/report.json \
  --prepared-state /root/autodl-tmp/Postman_Deploy/crate_height_sweep_20260911/preparation/prepared_state.json \
  --height-offset -0.15 --x-offset -0.05 --virtual-props \
  --output /root/autodl-tmp/Postman_Deploy/closer_virtual_new_trial
```

Omitting `--virtual-props` preserves the physical scene. Its previous near-edge
clearance validation remains enforced; a closer physical scene is not silently
enabled by this preview feature. All output tools reject nonempty destinations.

## Actual result

| Table height offset | Tabletop world Z | Simulated time | Outcome |
| --- | ---: | ---: | --- |
| 0 cm | 1.010919 m | 35.781 s | Right arm yaw limit during upward wrist path |
| -5 cm | 0.960919 m | 42.840 s | Entire scheduled sequence completed; accuracy criteria NOT passed |
| -10 cm | 0.910919 m | 13.167 s | Left ankle roll limit at PREALIGN entry |
| -15 cm | 0.860919 m | 12.187 s | Left ankle roll limit during wrist turn |
| -20 cm | 0.810919 m | 11.836 s | Left ankle roll limit during wrist turn |

One of five completes playback, **zero of five passes the original coarse reach
accuracy criteria**. None attempts grasping. The -5 cm case is useful for
reviewing the complete motion, not a physically verified insertion/lift.

For that -5 cm case, mean errors over the INSERT_SETTLE settling interval are
left **2.09 cm / 20.78 deg**, right **1.85 cm / 16.06 deg**. Final HOLD settling
means are left **2.89 cm / 25.62 deg**, right **5.69 cm / 22.97 deg**. Its maximum
foot-origin excursion is 6.62 mm; this does not erase the large wrist orientation
bias. The final instantaneous errors are 2.88/5.74 cm and 25.61/22.99 deg.

The other cases stop at their actual unmodified hard-limit tolerance, not from
contact with a virtual prop. The 0 cm case exceeds the right arm yaw upper
limit by 0.010003 rad. The -10/-15/-20 cm cases exceed the left ankle roll upper
limit by 0.010050/0.010140/0.014540 rad, respectively.

Do not attribute differences from the earlier physical comparison solely to
the 5 cm distance change: **both horizontal placement and prop contact physics
changed**. In particular, this preview allows the hands to pass through the
crate while revealing how the current policy behaves later in the path.
The earlier physical experiment's low-height candidate preference is therefore
not a definitive choice of post-training centre. Review the complete -5 cm
preview before choosing; actual contact feasibility still needs separate work.

## Verification

245 focused and existing regression tests passed. Tests check unchanged robot
dynamics/controllers, actual overlapping placeholders with no contact response,
static crate geometry, real foot/floor contact, retained body-limit and non-foot
ground-contact stops, exact target translation and continuous reference handoff.

All five reports have matching checkpoint/ONNX/prepared-state hashes. All five
traces have zero prop contacts, zero crate pose drift and open-finger commands.
The 0 cm and -15 cm cases were also independently rerun without rendering:
their traces, targets and transitions match the video runs byte-for-byte.
There are no runtime exceptions or cleanup errors. Complete desired paths for
all five heights contain 4085 samples each at 100 Hz (40.84 s after START_HOLD),
including any unexecuted suffix; these are not successful demonstrations.
