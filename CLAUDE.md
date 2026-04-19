# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

Multi-policy deployment framework for the Unitree G1 (29-DoF) humanoid robot, based on a finite-state machine that switches between motion policies. Fork of [ccrpRepo/RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy) with added **GAE_Mimic** motion tracking.

Deployment targets: **MuJoCo simulation** (`deploy_mujoco/`) and **real robot** via `unitree_sdk2_python` (`deploy_real/`).

## Architecture

- **FSM** (`FSM/FSM.py`): owns one instance of every policy, holds a `cur_policy`, and each tick calls `cur_policy.run()` then `checkChange()`. Transitions go through a two-phase `CHANGE`→`NORMAL` cycle (`exit()` on the old, `enter()` on the new). To add a policy: instantiate it in `FSM.__init__`, add a branch to `get_next_policy`, and route an `FSMCommand` to it from whichever policies should transition into it (usually `LocoMode.checkChange`).
- **FSMState** (`FSM/FSMState.py`): base class. Subclasses implement `enter / run / exit / checkChange` and set `self.name` (from `FSMStateName`) and `self.name_str`.
- **Shared enums** live in `common/utils.py`: `FSMStateName` (policy IDs) and `FSMCommand` (joystick/keyboard-issued commands). Keep these two in sync when adding policies.
- **State bus** (`common/ctrlcomp.py`): `StateAndCmd` carries robot state + `vel_cmd` + `skill_cmd` *into* policies; `PolicyOutput` carries `actions / kps / kds` *out*. Both are constructed once in the deploy entrypoint and passed into `FSM`.
- **Deploy entrypoints** (`deploy_mujoco/deploy_mujoco.py`, `deploy_mujoco/deploy_mujoco_keyboard_input.py`, `deploy_real/deploy_real.py`): load sim/real config, build `StateAndCmd`/`PolicyOutput`, build `FSM`, then in the loop: read joystick/keyboard → write `state_cmd.skill_cmd` + `vel_cmd` → `FSM.run()` → apply `policy_output` via PD control (`pd_control`) to the robot.
- **Path convention**: `common/path_config.py` inserts the repo root into `sys.path` so every module does `from common.path_config import PROJECT_ROOT` as its first import and uses absolute imports (`from policy.loco_mode.LocoMode import LocoMode`). Preserve this pattern in new files.

## Policy layout

Each policy under `policy/<name>/` typically contains:
- `<Name>.py` — the `FSMState` subclass
- `config/<Name>.yaml` — kps/kds, default angles, `joint2motor_idx` remap, obs/action scales, cmd ranges
- `model/` — `.pt` (TorchScript, e.g. `LocoMode`) or `.onnx` (e.g. `GAE_Mimic`) — **not checked in; user-provided**
- `motion/` — reference trajectories for mimic policies (npz), also user-provided

`LocoMode` is the canonical reference — observation assembly, joint reordering via `joint2motor_idx`, and the action-scale/default-angle pattern in `run()` are reused by most policies.

## Joint ordering gotcha

Policies are trained in their own joint order, but the robot/MuJoCo model has a fixed motor order. Every policy that loads a config uses `joint2motor_idx` to remap obs (motor → policy order) on input and actions (policy → motor order) on output. When adding a new policy or debugging NaN/unstable behavior, check this remap first.

## Control flow for a new skill

1. Add an entry to `FSMStateName` and (if needed) `FSMCommand` in `common/utils.py`.
2. Create `policy/<name>/<Name>.py` subclassing `FSMState`; mirror `LocoMode` structure.
3. Add config yaml under `policy/<name>/config/`.
4. Instantiate in `FSM.__init__` and wire up `get_next_policy`.
5. In the source state's `checkChange` (usually `LocoMode`), map the `FSMCommand` to the new `FSMStateName`.
6. In deploy entrypoints, bind a joystick button combo / keyboard string to set `state_cmd.skill_cmd`.

## Running

```bash
# MuJoCo with Xbox controller
python deploy_mujoco/deploy_mujoco.py

# MuJoCo with keyboard (type "b+l1", "vel 0.5 0 0", "exit", ...)
python deploy_mujoco/deploy_mujoco_keyboard_input.py

# Real robot (requires unitree_sdk2_python)
python deploy_real/deploy_real.py
```

Command bindings are documented in `README.md`. `F1`/`Select` → passive/E-stop. `L3` → passive.

## Environment

Python 3.8, PyTorch 2.3.1, numpy 1.20.0, `onnx` + `onnxruntime`, `mujoco`, `unitree_sdk2_python` (installed editable from source). See `README.md` for the exact install recipe.

## Safety

Test every change in MuJoCo before touching a real robot. Only `LocoMode`, `FixedPose`, `PassiveMode`, and `Dance` are considered stable on hardware; other skills are sim-only or experimental.
