"""Verify the deployment actor against real, history-bearing training observations.

Run with the Postman simulation interpreter. A subprocess uses the existing,
read-only AMO_R2 environment to collect native ManagerBasedRlEnv snapshots. This
keeps PyTorch/MJWarp and the deployment runtime independent. No training occurs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TASK = "R2V2-Reach-DualArm-SpeedLimited-28DoF"
RUN = (
    "logs/rsl_rl/r2v2_reach_dual_arm_speed_limited_28dof/"
    "2026-08-30_00-45-04_dual-arm-speed-limited-v1-from-13250"
)
DEFAULT_OUTPUT = ROOT / "artifacts/r2v2_reach/parity_12000"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare_weight_arrays(initializers: dict, state: dict, epsilon: float) -> dict:
    """Compare every actor tensor and the actual ONNX normalization divisor."""
    expected = {
        name: np.asarray(value)
        for name, value in state.items()
        if name.startswith("mlp.") or name == "obs_normalizer._mean"
    }
    denominator = np.asarray(state["obs_normalizer._std"]) + epsilon
    divisors = {
        key: value
        for key, value in initializers.items()
        if key not in expected and np.asarray(value).shape == denominator.shape
    }
    differences = {}
    for name, expected_value in expected.items():
        actual = initializers.get(name)
        differences[name] = (
            None
            if actual is None or np.asarray(actual).shape != expected_value.shape
            else float(np.max(np.abs(actual - expected_value)))
        )
    divisor_matches = {
        name: float(np.max(np.abs(value - denominator)))
        for name, value in divisors.items()
    }
    divisor_match = min(divisor_matches, key=divisor_matches.get) if divisor_matches else None
    exact = all(error == 0.0 for error in differences.values())
    exact_divisor = divisor_match is not None and divisor_matches[divisor_match] == 0.0
    return {
        "passed": bool(exact and exact_divisor),
        "tensor_max_abs_error": differences,
        "normalization_epsilon": epsilon,
        "normalization_divisor_initializer": divisor_match,
        "normalization_divisor_max_abs_error": (
            divisor_matches[divisor_match] if divisor_match is not None else None
        ),
        "comparison": "All eight MLP tensors, normalization mean and std + epsilon; exact equality",
    }


def collect_native(args: argparse.Namespace) -> None:
    """Only called in AMO_R2's Python; all outputs go into Postman artifacts."""
    from dataclasses import asdict

    import mujoco
    import onnx
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    from rsl_rl.modules.normalization import EmpiricalNormalization

    import r2v2_loco.tasks  # noqa: F401

    configure_torch_backends()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor_state = {
        key: value.detach().cpu().numpy()
        for key, value in checkpoint["actor_state_dict"].items()
    }
    epsilon = float(EmpiricalNormalization(1).eps)
    initializers = {
        item.name: onnx.numpy_helper.to_array(item)
        for item in onnx.load(args.onnx).graph.initializer
    }
    weights = compare_weight_arrays(initializers, actor_state, epsilon)
    cfg = load_env_cfg(TASK, play=True)
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.terminations = {}
    cfg.episode_length_s = 1000.0
    cfg.observations["actor"].enable_corruption = False
    for command_name in ("reach", "reach_right"):
        cfg.commands[command_name].resampling_time_range = (1.0e9, 1.0e9)
    agent_cfg = load_rl_cfg(TASK)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
        runner.load(str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location=device)
        native_policy = runner.get_inference_policy(device=device)
        if not weights["passed"]:
            old_metadata = onnx.load(args.onnx).metadata_props
            new_path = args.output / f"policy_{args.checkpoint.stem}.onnx"
            runner.export_policy_to_onnx(str(args.output), new_path.name)
            exported = onnx.load(new_path)
            exported.metadata_props.extend(old_metadata)
            onnx.save(exported, new_path)
            args.onnx = new_path
            weights = compare_weight_arrays(
                {item.name: onnx.numpy_helper.to_array(item) for item in exported.graph.initializer},
                actor_state,
                epsilon,
            )
            weights["reexported_for_checkpoint"] = True
        if not weights["passed"]:
            raise RuntimeError(f"ONNX parameter mismatch even after export: {weights}")
        env.scene.write(args.output / "training_scene")
        records: dict[str, list] = {}

        def record(key, tensor):
            value = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else tensor
            records.setdefault(key, []).append(np.asarray(value).copy())

        metadata = {
            "task": TASK,
            "frames": args.frames,
            "seed": args.seed,
            "episodes": 2,
            "step_dt": env.step_dt,
            "physics_dt": env.physics_dt,
            "torch_version": torch.__version__,
            "native_mujoco_version": mujoco.__version__,
            "checkpoint_iter": int(checkpoint["iter"]),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "weights": weights,
            "onnx_path": str(args.onnx.resolve()),
            "onnx_sha256": sha256(args.onnx),
            "training_scene": str((args.output / "training_scene/scene.xml").resolve()),
            "observation_terms": {},
        }
        offset = 0
        manager = env.observation_manager
        names = manager._group_obs_term_names["actor"]
        terms = manager._group_obs_term_cfgs["actor"]
        for name, term in zip(names, terms, strict=True):
            sample = term.func(env, **term.params)
            width = sample.shape[-1]
            length = int(term.history_length)
            metadata["observation_terms"][name] = {
                "offset": offset,
                "width": width,
                "history_length": length,
                "scale": None if term.scale is None else term.scale.detach().cpu().tolist(),
            }
            offset += width * max(length, 1)
        ids = torch.zeros(1, dtype=torch.long, device=device)
        targets = (
            ((0.38, 0.22, 0.42, 15.0, -10.0, 20.0), (0.38, -0.22, 0.42, -15.0, -10.0, -20.0)),
            ((0.42, 0.17, 0.56, -20.0, 15.0, 25.0), (0.33, -0.25, 0.32, 15.0, -20.0, -15.0)),
            ((0.29, 0.28, 0.40, 5.0, -25.0, 30.0), (0.45, -0.12, 0.48, -10.0, 20.0, -30.0)),
        )
        for episode in range(2):
            observations, _ = env.reset(seed=args.seed + episode)
            for index in range(args.frames):
                with torch.no_grad():
                    observations = wrapped.get_observations()
                    action = native_policy(observations)
                    record("qpos", env.sim.data.qpos[0])
                    record("qvel", env.sim.data.qvel[0])
                    record("native_observation", observations["actor"][0])
                    record("native_action", action[0])
                    record("previous_action", env.action_manager.action[0])
                    record("episode_start", index == 0)
                    for side, command_name in (("left", "reach"), ("right", "reach_right")):
                        command = env.command_manager.get_term(command_name)
                        for field in (
                            "target_pos_w", "target_quat_w", "goal_pos_w", "goal_quat_w",
                            "reference_lin_vel_w", "reference_ang_vel_w", "linear_speed_limit",
                            "angular_speed_limit", "arm_mask",
                        ):
                            record(f"{side}_{field}", getattr(command, field)[0])
                    record("twist_command", env.command_manager.get_command("twist")[0])
                    record("base_pose_command", env.command_manager.get_command("base_pose")[0])
                    if index in (0, args.frames // 3, 2 * args.frames // 3):
                        target_index = min(index * 3 // args.frames, 2)
                        for side, command_name, target in zip(
                            ("left", "right"), ("reach", "reach_right"), targets[target_index], strict=True
                        ):
                            env.command_manager.get_term(command_name).set_target_xyz_rpy(
                                ids, arm=side, xyz_rpy=torch.tensor(target), degrees=True
                            )
                    env.step(action)
        np.savez_compressed(args.output / "native_snapshots.npz", **records)
        _dump(args.output / "native_metadata.json", metadata)
    finally:
        env.close()


def _report_errors(errors: np.ndarray, tolerance: float) -> dict:
    return {
        "passed": bool(np.isfinite(errors).all() and np.max(np.abs(errors)) <= tolerance),
        "max_abs_error": float(np.max(np.abs(errors))),
        "tolerance": tolerance,
    }


def verify_deployment(args: argparse.Namespace) -> dict:
    import mujoco
    import onnxruntime as ort

    from common.r2v2_reach_policy import ReachPolicy

    metadata = json.loads((args.output / "native_metadata.json").read_text())
    args.onnx = Path(metadata["onnx_path"])
    if metadata["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Native snapshots belong to a different checkpoint; recollect them")
    if metadata["onnx_sha256"] != sha256(args.onnx):
        raise ValueError("ONNX changed after native collection; recollect snapshots")
    # mjlab exports in-memory meshes only; this robot's meshes are disk-backed.
    # Resolve those assets read-only at their original location, not by copying
    # a robot/model from deployment or silently substituting simplified geometry.
    xml_root = ET.fromstring(Path(metadata["training_scene"]).read_text())
    xml_root.find("compiler").set(
        "meshdir", str((args.training_root / "src/r2v2_loco/assets/r2v2").resolve())
    )
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml_root, encoding="unicode"))
    data = mujoco.MjData(model)
    samples = np.load(args.output / "native_snapshots.npz")
    adapter = ReachPolicy(model, data, args.onnx)
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    observed, actions_onnx, actions_deploy = [], [], []
    fields = {
        "position": "target_pos_w", "quaternion": "target_quat_w",
        "goal_position": "goal_pos_w", "goal_quaternion": "goal_quat_w",
        "linear_velocity": "reference_lin_vel_w", "angular_velocity": "reference_ang_vel_w",
    }
    for index in range(len(samples["qpos"])):
        data.qpos[:] = samples["qpos"][index]
        data.qvel[:] = samples["qvel"][index]
        mujoco.mj_forward(model, data)
        if samples["episode_start"][index]:
            adapter.reset(data)
        adapter.last_action[:] = samples["previous_action"][index]
        for side in ("left", "right"):
            reference = adapter.references[side]
            for destination, source in fields.items():
                getattr(reference, destination)[:] = samples[f"{side}_{source}"][index]
        observation = np.asarray(
            adapter.observe(data, advance_reference=False, update_history=True), dtype=np.float32
        ).reshape(1, -1)
        native = samples["native_observation"][index].astype(np.float32)[None]
        observed.append(observation[0])
        actions_onnx.append(session.run(None, {input_name: native})[0][0])
        actions_deploy.append(session.run(None, {input_name: observation})[0][0])
    observation_errors = np.asarray(observed) - samples["native_observation"]
    term_reports = {}
    latest_errors = []
    for name, details in metadata["observation_terms"].items():
        start = details["offset"]
        end = start + details["width"] * details["history_length"]
        term_reports[name] = _report_errors(observation_errors[:, start:end], 5.0e-5)
        latest_errors.append(observation_errors[:, end - details["width"]:end])
    single_frame = _report_errors(np.concatenate(latest_errors, axis=-1), 5.0e-5)
    history = _report_errors(observation_errors, 5.0e-5)
    action_report = _report_errors(np.asarray(actions_onnx) - samples["native_action"], 1.0e-4)
    end_to_end = _report_errors(np.asarray(actions_deploy) - samples["native_action"], 2.0e-4)
    adapter_path = ROOT / "common/r2v2_reach_policy.py"
    checks = {
        "weights": metadata["weights"],
        "observations": single_frame,
        "history": history,
        "actions_same_native_observation": action_report,
        "end_to_end_actions": end_to_end,
    }
    report = {
        "passed": all(item["passed"] for item in checks.values()),
        "task": TASK,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_iter": metadata["checkpoint_iter"],
        "onnx_path": str(args.onnx.resolve()),
        "onnx_sha256": sha256(args.onnx),
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256(adapter_path),
        "snapshot_sha256": sha256(args.output / "native_snapshots.npz"),
        "sample_count": len(samples["qpos"]),
        "episodes": metadata["episodes"],
        "checks": checks,
        "observation_terms": term_reports,
        "coverage": (
            "Actual native ManagerBasedRlEnv play rollouts: two resets, symmetric/asymmetric 6D goals, "
            "changing joint states/velocities, previous actions, world-frame reference pose/twist, "
            "and all 24 term-major ten-frame histories. CPU MuJoCo forward kinematics from identical qpos/qvel."
        ),
        "native_mujoco_version": metadata["native_mujoco_version"],
        "deployment_mujoco_version": mujoco.__version__,
        "limitations": "This is numerical interface parity on the OLD training robot, not new-model standing or grasp validation.",
    }
    np.savez_compressed(args.output / "deployment_comparison.npz", observation=np.asarray(observed),
                        action_onnx_native=np.asarray(actions_onnx), action_onnx_deployment=np.asarray(actions_deploy))
    _dump(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=ROOT.parent / "AMO_R2")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=120, help="Frames per episode; two episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-snapshots", action="store_true")
    parser.add_argument("--collect-native", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run_path = args.training_root / RUN
    args.checkpoint = (args.checkpoint or run_path / "model_12000.pt").resolve()
    args.onnx = (args.onnx or run_path / f"{run_path.name}.onnx").resolve()
    args.output = args.output.resolve()
    if args.frames < 30:
        parser.error("At least 30 frames per episode are needed to exercise history rollover")
    # Never write snapshots into AMO_R2 or overwrite its exported policy.
    if args.output.is_relative_to(args.training_root.resolve()):
        parser.error("Output must be outside the read-only training repository")
    for path in (args.checkpoint, args.onnx):
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    return args


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.collect_native:
        collect_native(args)
        return 0
    # A failed rerun must not leave a previous successful report as an apparent
    # fresh gate. The collector itself does not publish a deployment pass.
    _dump(args.output / "report.json", {"passed": False, "status": "verification_in_progress"})
    try:
        if not args.reuse_snapshots:
            command = [
                str(args.training_root / ".venv/bin/python"), str(Path(__file__).resolve()),
                "--collect-native", "--training-root", str(args.training_root.resolve()),
                "--checkpoint", str(args.checkpoint), "--onnx", str(args.onnx),
                "--output", str(args.output), "--frames", str(args.frames), "--seed", str(args.seed),
            ]
            subprocess.run(command, check=True, cwd=ROOT)
        report = verify_deployment(args)
    except Exception as exc:
        _dump(args.output / "report.json", {
            "passed": False, "status": "verification_error",
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
