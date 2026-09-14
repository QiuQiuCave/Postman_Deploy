# Simple dual-wrist Reach probe (2026-09-11)

Empty scene, true new free-base R2V2 with articulated fingers held open by the
independent controller. Body policy alone drives the 28 body joints. No table,
crate, IK, auxiliary forces, welds or runtime pose resets. World targets refer
to `left_hand_roll_link` / `right_hand_roll_link`, without legacy TCP offsets.

## Reproduce

```bash
.venv-r2v2/bin/python deploy_mujoco/r2v2_simple_dual_reach.py \
  --reach-config /root/autodl-tmp/Postman_Deploy/simple_dual_reach_20260911/policy/reach_wrist_v2.yaml \
  --parity-report /root/autodl-tmp/Postman_Deploy/simple_dual_reach_20260911/parity/report.json \
  --output /root/autodl-tmp/Postman_Deploy/simple_dual_reach_20260911/new_run
```

Use a new output directory. `--no-video` retains the complete trace and report;
`--cases HOME_HOLD,SYMMETRIC_PREALIGN_30,RETURN_HOME` selects a shorter sequence.
Default sequence is HOME (5s), paired 30% prealign, paired 30% insert, paired
50% prealign, left-high probe, right-high probe, HOME return (6s each). There is
one 0.4s native-style reset-settling phase. Goals are set once per segment;
policy reference/history is not restarted. A timing transition does not mean
the preceding target passed. Physical safety remains active at 1kHz.

The asymmetric probes start from the 30% paired insert pose: one wrist moves
2cm forward and 3cm up, with a 5deg local-yaw change; the other moves 1cm back.
These are independent-arm **generalization probes**, not the correlated box
pose randomization used by the current training course.

Pinned checkpoint: `model_3500.pt`; matching ONNX SHA and checkpoint SHA are
recorded in the parity report. 240 native samples/two resets passed complete
inference parity: observation/history max error 3.30e-7, same-observation action
2.86e-6, end-to-end action 1.05e-5. Original training and shared adapter unchanged.

## Observed nominal run

`/root/autodl-tmp/Postman_Deploy/simple_dual_reach_20260911/video_sequence/`
contains `simple_dual_reach.mp4`, report/trace/transitions JSON and screenshots.
Video: 1280×800, 30fps, 43.467s (41.4s simulation plus 2s frozen terminal view and
frame sampling). The headless run reproduced the same measurements.

Each cell below reports mean position/orientation error over the **last 2s**
of that segment; these are not transient maxima or arbitrary-workspace claims.

| Target | Left: mm / deg | Right: mm / deg | Simple wrist gate |
| --- | ---: | ---: | --- |
| HOME | 7.66 / 3.01 | 12.12 / 4.80 | pass |
| Paired prealign 30% | 11.63 / 3.91 | 16.22 / 5.55 | pass |
| Paired insert 30% | 5.32 / 4.33 | 11.09 / 5.06 | pass |
| Paired prealign 50% | 16.39 / 6.16 | 15.24 / 5.44 | pass |
| Asymmetric left high | 7.31 / 5.02 | 18.13 / 3.70 | pass |
| Asymmetric right high | 6.08 / 4.82 | 22.95 / 5.77 | fail |
| HOME return | 7.89 / 3.45 | 10.90 / 5.37 | pass |

Simple wrist gate: both wrists <20mm, <10deg, <20mm/s simultaneously for at least
0.3s, with the scheduled segment completed. Every passing segment also had 100%
of its final-2s samples within these thresholds. It does not imply the entire
transition met endpoint tolerances. No segment passed the finer 5mm/3deg gate.

The sequence completed without falls, joint-limit violations or deep self
contacts. Peak base tilt was 16.12deg. Tail wrist position jitter RMS about its
own mean was 0.01–0.26mm: steady bias and jitter are different measurements.
`foot_drift_m` is the maximum displacement of any foot collision-sphere centre
from the post-settle anchor (peak 26.92mm); it includes foot rotation and is
**not** a pure foot-slip measure or the training foot-origin-site metric.

Conclusion: this nominal run supports coarse, small near-body target screening.
It does not validate precise insertion, obstacle avoidance, arbitrary independent
arm goals, loaded manipulation, or robustness across random initial states.
The separate training 70% evaluation remained unsuccessful despite survival;
this easier demonstration is not a replacement for that evaluation.

30 focused tests cover target/SLERP calibration, bounded asymmetric perturbations,
one-shot target setting, continuous bilateral gates and tail-window statistics.
