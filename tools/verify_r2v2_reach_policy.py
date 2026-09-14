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
WRIST_TASK = "R2V2-Reach-CrateWrist-v2-28DoF"
PATH_TASK = "R2V2-Reach-CrateWristPath-v1-28DoF"
PAYLOAD_TASK = "R2V2-Reach-CrateWristPayload-v2-28DoF"
PAYLOAD_PATH_CONTRACT = "wrist_payload_path_v2"
RUN = (
    "logs/rsl_rl/r2v2_reach_dual_arm_speed_limited_28dof/"
    "2026-08-30_00-45-04_dual-arm-speed-limited-v1-from-13250"
)
DEFAULT_OUTPUT = ROOT / "artifacts/r2v2_reach/parity_12000"


def endpoint_contract_for_task(task: str) -> str:
    if task == TASK:
        return "legacy_tcp_v1"
    if task in (WRIST_TASK, PATH_TASK, PAYLOAD_TASK):
        return "wrist_world_v2"
    raise ValueError(f"Unsupported parity task: {task}")


def resolve_training_scene(xml_path: Path, training_root: Path, task: str) -> str:
    """Resolve disk meshes for the selected training asset, never swap its body.

    V2's exported scene includes the locked hand bodies/joints exactly as trained;
    it is intentionally not reconstructed with deployment's 40 actuators.
    """
    contract = endpoint_contract_for_task(task)
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    asset = (training_root / "src/r2v2_loco/assets/r2v2" if contract == "legacy_tcp_v1"
             else ROOT / "r2v2_description/source/r2v2_with_hand/meshes")
    compiler.set("meshdir", str(asset.resolve()))
    for mesh in root.findall("./asset/mesh"):
        if mesh.get("file") and not (asset / mesh.get("file")).is_file():
            raise FileNotFoundError(f"Missing {task} native mesh: {asset / mesh.get('file')}")
    return ET.tostring(root, encoding="unicode")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_payload_bindings(checkpoint, onnx_metadata, checkpoint_path, path_manifest,
                              *, asset_manifest_sha256=None):
    """Bind the new task, exact checkpoint, path and real robot assets; fail closed.

    This is not a path-v1 alias even though both actors have 1460 inputs. The
    optional asset digest is provided by the actual native training runtime.
    """
    infos = checkpoint.get("infos") or {}
    run = infos.get("wrist_payload_run") or {}
    expected = dict(task_id=PAYLOAD_TASK, path_contract=PAYLOAD_PATH_CONTRACT,
                    endpoint_contract="wrist_world_v2", quaternion_order="wxyz",
                    speed_reference_point="wrist",
                    endpoint_body_names="left_hand_roll_link,right_hand_roll_link")
    for label, values in (("checkpoint", infos), ("run", run), ("ONNX", onnx_metadata)):
        for key, value in expected.items():
            if values.get(key) != value:
                raise ValueError(f"Payload {label} requires {key}={value}")
    checkpoint_sha = sha256(Path(checkpoint_path))
    if onnx_metadata.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Payload ONNX is not bound to the selected checkpoint hash")
    path_manifest = Path(path_manifest).resolve()
    manifest = json.loads(path_manifest.read_text())
    if (manifest.get("payload_training") or {}).get("contract") != PAYLOAD_PATH_CONTRACT:
        raise ValueError("Payload parity requires its versioned path manifest")
    archive = (path_manifest.parent / manifest["trajectory_file"]).resolve()
    if archive.parent != path_manifest.parent:
        raise ValueError("Payload trajectory must be alongside its manifest")
    trajectory_sha = sha256(archive)
    for label, value in (("manifest", manifest.get("trajectory_sha256")),
                         ("checkpoint", run.get("trajectory_sha256")),
                         ("ONNX", onnx_metadata.get("path_trajectory_sha256"))):
        if value != trajectory_sha:
            raise ValueError(f"Payload {label} trajectory hash mismatch")
    manifest_sha = sha256(path_manifest)
    if run.get("path_manifest_sha256") != manifest_sha:
        raise ValueError("Payload checkpoint path manifest hash mismatch")
    if asset_manifest_sha256 is not None and run.get("asset_manifest_sha256") != asset_manifest_sha256:
        raise ValueError("Payload checkpoint real robot asset hash differs from native runtime")
    return dict(passed=True, **expected, checkpoint_sha256=checkpoint_sha,
                path_manifest=str(path_manifest), path_manifest_sha256=manifest_sha,
                path_trajectory_sha256=trajectory_sha,
                asset_manifest_sha256=run.get("asset_manifest_sha256"),
                native_asset_verified=asset_manifest_sha256 is not None)


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
    from common.r2v2_reach_policy import validate_endpoint_metadata

    # Numerical parity must not depend on reduced-precision TF32 matmuls.
    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor_state = {
        key: value.detach().cpu().numpy()
        for key, value in checkpoint["actor_state_dict"].items()
    }
    epsilon = float(EmpiricalNormalization(1).eps)
    onnx_model = onnx.load(args.onnx)
    onnx_metadata = {entry.key: entry.value for entry in onnx_model.metadata_props}
    endpoint_contract = validate_endpoint_metadata(
        onnx_metadata,
        endpoint_contract_for_task(args.task),
    )
    initializers = {
        item.name: onnx.numpy_helper.to_array(item)
        for item in onnx_model.graph.initializer
    }
    weights = compare_weight_arrays(initializers, actor_state, epsilon)
    bindings = None
    if args.task == PAYLOAD_TASK:
        from r2v2_loco.robots.r2v2_wrist_constants import get_r2v2_wrist_source_manifest
        asset_digest = hashlib.sha256(json.dumps(get_r2v2_wrist_source_manifest(),
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        bindings = validate_payload_bindings(checkpoint, onnx_metadata, args.checkpoint,
            args.path_manifest, asset_manifest_sha256=asset_digest)
        if not weights["passed"]:
            raise ValueError("Payload ONNX actor mismatch; no metadata-carrying silent re-export permitted")
    cfg = load_env_cfg(args.task, play=True)
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    # Preserve v2's physical safety terminations. Legacy parity snapshots remain
    # reproducible; neither branch constitutes a standing or task success gate.
    if endpoint_contract == "legacy_tcp_v1":
        cfg.terminations = {}
    cfg.episode_length_s = 1000.0
    cfg.observations["actor"].enable_corruption = False
    for command_name in ("reach", "reach_right"):
        cfg.commands[command_name].resampling_time_range = (1.0e9, 1.0e9)
        if args.task == PAYLOAD_TASK:
            cfg.commands[command_name].path_manifest = str(args.path_manifest)
    agent_cfg = load_rl_cfg(args.task)
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        if args.task in (PATH_TASK, PAYLOAD_TASK):
            # Keep the path task's actual observations, reset and safety gates,
            # while this probe supplies its own held symmetric/asymmetric goals.
            # Otherwise its automatic path clock would overwrite those targets.
            env._paired_wrist_targets.set_external_control(True)
        if args.task == PAYLOAD_TASK:
            # Random training load is not part of numerical observer parity.
            # enabled=False means random training mode, NOT zero payload.
            env._wrist_payload.set_evaluation(enabled=True, mass_kg=0., seed=args.seed)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
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
            "task": args.task,
            "endpoint_contract": endpoint_contract,
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
            "payload_bindings": bindings,
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
                            command = env.command_manager.get_term(command_name)
                            if endpoint_contract == "wrist_world_v2":
                                # Exercise modest world-frame goals near the actual
                                # reset wrists; the older targets above are TCP/base-yaw.
                                robot = env.scene["robot"]
                                site_id = robot.find_sites(f"{side}_wrist")[0][0]
                                if index == 0:
                                    command._parity_start_position = robot.data.site_pos_w[:, site_id].clone()
                                    command._parity_start_quaternion = robot.data.site_quat_w[:, site_id].clone()
                                offsets = ((0.025, 0.0, 0.02), (0.04, 0.015, 0.035), (0.015, -0.01, 0.01))
                                offset = torch.tensor(offsets[target_index], device=device)
                                if side == "right":
                                    offset[1] *= -1
                                    if target_index == 1:
                                        offset[0] *= 0.75  # One asymmetric pose.
                                from mjlab.utils.lab_api.math import quat_mul, quat_from_euler_xyz
                                angles = torch.tensor(target[3:], device=device) * (np.pi / 180) * 0.2
                                delta = quat_from_euler_xyz(*angles.unbind())
                                q = quat_mul(command._parity_start_quaternion, delta.expand(1, 4))
                                command.set_target_world(ids, command._parity_start_position + offset, q)
                            else:
                                command.set_target_xyz_rpy(ids, arm=side, xyz_rpy=torch.tensor(target), degrees=True)
                    result = env.step(action)
                    if endpoint_contract == "wrist_world_v2" and bool((result[2] | result[3]).any()):
                        raise RuntimeError("Native wrist parity rollout hit a safety termination; no parity pass published")
        if args.task == PAYLOAD_TASK:
            payload = env._wrist_payload.summarize()
            if any(v["max_total_downward_force_N"] != 0 for v in payload["phase_metrics"].values()):
                raise RuntimeError("Unloaded payload parity unexpectedly applied external force")
            metadata["payload_runtime"] = payload
            # A modified input during collection must not inherit a prior pass.
            validate_payload_bindings(checkpoint, onnx_metadata, args.checkpoint,
                args.path_manifest, asset_manifest_sha256=asset_digest)
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


def seed_reset_history_from_current_state(adapter, data) -> None:
    """Match native reset padding after restoring the SAME reference state.

    No native observation/history values are copied. Every term is rebuilt by
    the deployment adapter from MuJoCo state and the restored goal/reference.
    Path v1 already moves its reference toward HOME during command-manager reset,
    unlike legacy's follow-current reset, so seeding before restoration is wrong.
    """
    terms = adapter.observation_terms(data)
    for name, value in terms.items():
        adapter.histories[name][:] = value


def verify_deployment(args: argparse.Namespace) -> dict:
    import mujoco
    import onnxruntime as ort

    from common.r2v2_reach_policy import ReachPolicy

    metadata = json.loads((args.output / "native_metadata.json").read_text())
    if metadata["task"] != args.task:
        raise ValueError("Native snapshots belong to a different task/endpoint contract")
    expected_contract = endpoint_contract_for_task(args.task)
    if metadata.get("endpoint_contract", "legacy_tcp_v1") != expected_contract:
        raise ValueError("Native snapshot endpoint contract mismatch")
    args.onnx = Path(metadata["onnx_path"])
    if metadata["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Native snapshots belong to a different checkpoint; recollect them")
    if metadata["onnx_sha256"] != sha256(args.onnx):
        raise ValueError("ONNX changed after native collection; recollect snapshots")
    if args.task == PAYLOAD_TASK:
        bindings = metadata.get("payload_bindings") or {}
        if not bindings.get("passed") or not bindings.get("native_asset_verified"):
            raise ValueError("Payload snapshots lack native asset/task/path binding evidence")
        if args.path_manifest is None or str(args.path_manifest.resolve()) != bindings.get("path_manifest"):
            raise ValueError("Payload snapshots belong to a different path manifest")
        if sha256(args.path_manifest) != bindings.get("path_manifest_sha256"):
            raise ValueError("Payload manifest changed after native collection")
        manifest = json.loads(args.path_manifest.read_text())
        if sha256(args.path_manifest.parent / manifest["trajectory_file"]) != bindings.get("path_trajectory_sha256"):
            raise ValueError("Payload trajectory changed after native collection")
    # mjlab exports in-memory meshes only; this robot's meshes are disk-backed.
    # Resolve those assets read-only at their original location, not by copying
    # a robot/model from deployment or silently substituting simplified geometry.
    model = mujoco.MjModel.from_xml_string(resolve_training_scene(
        Path(metadata["training_scene"]), args.training_root, args.task,
    ))
    data = mujoco.MjData(model)
    samples = np.load(args.output / "native_snapshots.npz")
    adapter = ReachPolicy(model, data, args.onnx, expected_endpoint_contract=expected_contract)
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
        if samples["episode_start"][index]:
            seed_reset_history_from_current_state(adapter, data)
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
    if args.task == PAYLOAD_TASK:
        checks["payload_bindings"] = metadata["payload_bindings"]
    report = {
        "passed": all(item["passed"] for item in checks.values()),
        "task": args.task,
        "endpoint_contract": adapter.endpoint_contract,
        "quaternion_order": "wxyz",
        "speed_reference_point": "wrist" if expected_contract == "wrist_world_v2" else "tcp",
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
            "and all 24 term-major ten-frame histories. CPU MuJoCo forward kinematics from identical qpos/qvel. "
            "Reset padding is independently rebuilt by deployment after synchronizing the same reset reference; "
            "no native observation/history vectors are injected into the deployment observer."
        ),
        "native_mujoco_version": metadata["native_mujoco_version"],
        "deployment_mujoco_version": mujoco.__version__,
        "limitations": (
            "Numerical interface parity on the selected task's exact training scene only. "
            "For wrist_world_v2 the hands are frozen open as trained; deployment has independent finger actuators. "
            "This is not free-standing, table collision, grasp, or load-carrying validation."
        ),
    }
    if args.task == PAYLOAD_TASK:
        report.update(path_contract=PAYLOAD_PATH_CONTRACT,
            path_trajectory_sha256=metadata["payload_bindings"]["path_trajectory_sha256"],
            path_manifest_sha256=metadata["payload_bindings"]["path_manifest_sha256"],
            payload_runtime=metadata["payload_runtime"])
    np.savez_compressed(args.output / "deployment_comparison.npz", observation=np.asarray(observed),
                        action_onnx_native=np.asarray(actions_onnx), action_onnx_deployment=np.asarray(actions_deploy))
    _dump(args.output / "report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, default=ROOT.parent / "AMO_R2")
    parser.add_argument("--task", choices=(TASK, WRIST_TASK, PATH_TASK, PAYLOAD_TASK), default=TASK)
    parser.add_argument("--device", help="Native training runtime device, e.g. cpu or cuda:0")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--path-manifest", type=Path, help="Required exact training path for payload-v2 parity")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=120, help="Frames per episode; two episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-snapshots", action="store_true")
    parser.add_argument("--collect-native", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.task in (WRIST_TASK, PATH_TASK, PAYLOAD_TASK) and (args.checkpoint is None or args.onnx is None or args.output == DEFAULT_OUTPUT):
        parser.error("wrist_world_v2 requires explicit --checkpoint, --onnx, and a separate --output")
    if args.task == PAYLOAD_TASK and args.path_manifest is None:
        parser.error("payload-v2 requires explicit --path-manifest")
    if args.path_manifest is not None:
        args.path_manifest = args.path_manifest.resolve()
        if not args.path_manifest.is_file():
            parser.error(f"Missing path manifest: {args.path_manifest}")
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
                "--task", args.task,
                "--checkpoint", str(args.checkpoint), "--onnx", str(args.onnx),
                "--output", str(args.output), "--frames", str(args.frames), "--seed", str(args.seed),
            ]
            if args.device:
                command += ["--device", args.device]
            if args.path_manifest:
                command += ["--path-manifest", str(args.path_manifest)]
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
