"""Independent native-observation/deployment parity for the frontal drop actor.

No training, silent re-export, task alias, observation injection or contact
success claim. Native samples use the actual frontal task and rigid-open asset;
deployment reconstructs all term-major histories from identical physical state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.verify_r2v2_reach_policy import (
    compare_weight_arrays, seed_reset_history_from_current_state, sha256, _report_errors,
)

TASK = "R2V2-Reach-FrontManipulation-v1-28DoF"
CONTRACT = dict(task_id=TASK, endpoint_contract="wrist_world_v2",
    path_contract="front_manipulation_path_v1", task_scope="tabletop_to_crate_drop_v1",
    quaternion_order="wxyz", speed_reference_point="wrist",
    endpoint_body_names="left_hand_roll_link,right_hand_roll_link")
REFERENCE_FIELDS = dict(position="target_pos_w", quaternion="target_quat_w",
    goal_position="goal_pos_w", goal_quaternion="goal_quat_w",
    linear_velocity="reference_lin_vel_w", angular_velocity="reference_ang_vel_w")


def canonical_digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def validate_front_bindings(checkpoint, onnx_metadata, checkpoint_path, path_manifest,
                            *, asset_manifest_sha256):
    infos = checkpoint.get("infos") or {}
    run = infos.get("front_manipulation_run") or {}
    for label, values in (("checkpoint", infos), ("run", run), ("ONNX", onnx_metadata)):
        for key, expected in CONTRACT.items():
            if values.get(key) != expected:
                raise ValueError(f"Front {label} requires {key}={expected}")
    checkpoint_sha = sha256(Path(checkpoint_path))
    if onnx_metadata.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Front ONNX checkpoint hash mismatch")
    manifest = json.loads(Path(path_manifest).read_text())
    path_sha = canonical_digest({k: v for k, v in manifest.items() if k != "content_sha256"})
    for label, value in (("manifest", manifest.get("content_sha256")),
                         ("run", run.get("path_sha256")), ("ONNX", onnx_metadata.get("path_sha256"))):
        if value != path_sha:
            raise ValueError(f"Front {label} path hash mismatch")
    for key in ("endpoint_contract", "path_contract", "task_scope"):
        if manifest.get(key) != CONTRACT[key]:
            raise ValueError(f"Front manifest {key} mismatch")
    state = infos.get("front_manipulation_curriculum") or {}
    if (state.get("schema_version") != 4 or state.get("path_contract") != CONTRACT["path_contract"]
            or state.get("contract", {}).get("path_sha256") != path_sha
            or state.get("contract", {}).get("task_scope") != CONTRACT["task_scope"]):
        raise ValueError("Front curriculum path/scope/schema mismatch")
    if not asset_manifest_sha256 or run.get("asset_manifest_sha256") != asset_manifest_sha256:
        raise ValueError("Front actual native robot asset hash mismatch")
    return dict(passed=True, **CONTRACT, checkpoint_sha256=checkpoint_sha,
        path_sha256=path_sha, path_manifest_sha256=sha256(Path(path_manifest)),
        asset_manifest_sha256=asset_manifest_sha256, native_asset_verified=True)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def streamed_target(home, side_index, index, frames):
    """Continuous cubic world target increments; never resets policy history."""
    segment = frames//3
    if index < segment:
        fraction = 0.
    else:
        stage = min(index//segment-1, 1)
        u = min(max((index-(stage+1)*segment)/segment, 0.), 1.)
        fraction = stage+u*u*(3.-2.*u)
    position = np.asarray(home, dtype=float)+fraction*np.array([
        .015, .012 if side_index == 0 else -.007, .01])
    angles = fraction*np.deg2rad([1., -1.5, 2. if side_index == 0 else -1.])
    return position, angles


def collect_native(args):
    from dataclasses import asdict
    import mujoco
    import onnx
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls, load_rl_cfg
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.utils.lab_api.math import quat_from_euler_xyz
    from rsl_rl.modules.normalization import EmpiricalNormalization
    import r2v2_loco.tasks  # noqa: F401
    from r2v2_loco.robots.r2v2_wrist_constants import get_r2v2_wrist_source_manifest
    from r2v2_loco.tasks.reach.front_manipulation_env_cfgs import front_manipulation_env_cfg
    from common.r2v2_reach_policy import validate_endpoint_metadata

    configure_torch_backends(allow_tf32=False)
    torch.set_num_threads(1)
    configure_seed = args.seed
    torch.manual_seed(configure_seed)
    device = args.device
    source = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    exported = onnx.load(args.onnx)
    metadata = {entry.key: entry.value for entry in exported.metadata_props}
    validate_endpoint_metadata(metadata, "wrist_world_v2")
    bindings = validate_front_bindings(source, metadata, args.checkpoint, args.path_manifest,
        asset_manifest_sha256=canonical_digest(get_r2v2_wrist_source_manifest()))
    weights = compare_weight_arrays(
        {x.name: onnx.numpy_helper.to_array(x) for x in exported.graph.initializer},
        {k: v.detach().cpu().numpy() for k, v in source["actor_state_dict"].items()},
        float(EmpiricalNormalization(1).eps))
    if not weights["passed"]:
        raise ValueError("Exact actor/normalizer mismatch; silent ONNX re-export forbidden")
    cfg = front_manipulation_env_cfg(play=True, path_manifest=args.path_manifest,
        path_sha256=bindings["path_sha256"])
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.auto_reset = False
    cfg.episode_length_s = 1000.
    cfg.observations["actor"].enable_corruption = False
    for name in ("push_robot", "base_mass", "base_com", "encoder_bias", "foot_friction"):
        cfg.events.pop(name, None)
    cfg.events["reset_robot_joints"].params["position_range"] = (-.005, .005)
    cfg.events["reset_base"].params["pose_range"] = {"x": (-.005, .005), "y": (-.005, .005), "yaw": (-.01, .01)}
    agent = load_rl_cfg(TASK)
    agent.warm_start_checkpoint = None
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    try:
        env._paired_wrist_targets.load_state_dict(source["infos"]["front_manipulation_curriculum"])
        env._paired_wrist_targets.set_external_control(True)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = load_runner_cls(TASK)(wrapped, asdict(agent), device=device)
        runner.alg.load(source, load_cfg={"actor": True}, strict=True)
        policy = runner.get_inference_policy()
        frozen = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
        env.scene.write(args.output/"training_scene")
        details, offset = {}, 0
        manager = env.observation_manager
        for name, term in zip(manager._group_obs_term_names["actor"], manager._group_obs_term_cfgs["actor"], strict=True):
            width = term.func(env, **term.params).shape[-1]
            details[name] = dict(offset=offset, width=width, history_length=int(term.history_length))
            offset += width*max(int(term.history_length), 1)
        if offset != 1460:
            raise ValueError(f"Native front observation contract changed: {offset}")
        records = {}
        def record(key, value):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            records.setdefault(key, []).append(np.asarray(value).copy())
        commands = [env.command_manager.get_term(n) for n in ("reach", "reach_right")]
        ids = torch.zeros(1, device=device, dtype=torch.long)
        home = env._paired_wrist_targets.path.manifest["phases"][1]
        for episode in range(2):
            wrapped.seed(args.seed+episode)
            with torch.inference_mode():
                wrapped.reset()
            for index in range(args.frames):
                with torch.inference_mode():
                    obs = wrapped.get_observations()
                    action = policy(obs)
                    record("qpos", env.sim.data.qpos[0]); record("qvel", env.sim.data.qvel[0])
                    record("native_observation", obs["actor"][0]); record("native_action", action[0])
                    record("previous_action", env.action_manager.action[0]); record("episode_start", index == 0)
                    for side, command in zip(("left", "right"), commands):
                        for field in REFERENCE_FIELDS.values():
                            record(f"{side}_{field}", getattr(command, field)[0])
                    if index >= 20:
                        for side_index, command in enumerate(commands):
                            p, angles = streamed_target(home["position_m"][side_index], side_index, index, args.frames)
                            p = torch.tensor(p, dtype=torch.float32, device=device)
                            angles = torch.tensor(angles, dtype=torch.float32, device=device)
                            q = quat_from_euler_xyz(*angles.unbind())
                            command.set_target_world(ids, p[None]+env.scene.env_origins, q[None])
                    result = env.step(action)
                    if bool((result[2] | result[3]).any()):
                        raise RuntimeError("Native front parity safety termination; no pass published")
        if any(not torch.equal(v, runner.alg.actor.state_dict()[k]) for k, v in frozen.items()):
            raise RuntimeError("Parity rollout changed actor or normalization")
        validate_front_bindings(source, metadata, args.checkpoint, args.path_manifest,
            asset_manifest_sha256=canonical_digest(get_r2v2_wrist_source_manifest()))
        np.savez_compressed(args.output/"native_snapshots.npz", **records)
        dump(args.output/"native_metadata.json", dict(task=TASK, **CONTRACT,
            checkpoint_iter=int(source["iter"]), checkpoint_sha256=sha256(args.checkpoint),
            onnx_sha256=sha256(args.onnx), bindings=bindings, weights=weights,
            episodes=2, frames=args.frames, seed=args.seed, observation_terms=details,
            native_mujoco_version=mujoco.__version__, step_dt=env.step_dt, physics_dt=env.physics_dt))
    finally:
        env.close()


def verify_deployment(args):
    import mujoco
    import onnxruntime as ort
    from common.r2v2_reach_policy import ReachPolicy
    meta = json.loads((args.output/"native_metadata.json").read_text())
    for key, expected in {**CONTRACT, "task": TASK, "checkpoint_sha256": sha256(args.checkpoint),
                          "onnx_sha256": sha256(args.onnx)}.items():
        if meta.get(key) != expected:
            raise ValueError(f"Stale native front snapshots: {key}")
    if meta["bindings"]["path_manifest_sha256"] != sha256(args.path_manifest):
        raise ValueError("Path changed after native snapshots")
    root = ET.parse(args.output/"training_scene/scene.xml").getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    meshdir = ROOT/"r2v2_description/source/r2v2_with_hand/meshes"
    compiler.set("meshdir", str(meshdir))
    for mesh in root.findall("./asset/mesh"):
        if mesh.get("file") and not (meshdir/mesh.get("file")).is_file():
            raise FileNotFoundError(meshdir/mesh.get("file"))
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    adapter = ReachPolicy(model, data, args.onnx, expected_endpoint_contract="wrist_world_v2")
    options = ort.SessionOptions(); options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"])
    samples = np.load(args.output/"native_snapshots.npz")
    observed, native_actions, deployed_actions = [], [], []
    for i in range(len(samples["qpos"])):
        data.qpos[:] = samples["qpos"][i]; data.qvel[:] = samples["qvel"][i]
        mujoco.mj_forward(model, data)
        if samples["episode_start"][i]:
            adapter.reset(data)
        adapter.last_action[:] = samples["previous_action"][i]
        for side in ("left", "right"):
            for target, source in REFERENCE_FIELDS.items():
                getattr(adapter.references[side], target)[:] = samples[f"{side}_{source}"][i]
        if samples["episode_start"][i]:
            seed_reset_history_from_current_state(adapter, data)
        obs = adapter.observe(data, advance_reference=False, update_history=True)[None]
        observed.append(obs[0])
        native_actions.append(session.run(None, {"obs": samples["native_observation"][i:i+1]})[0][0])
        deployed_actions.append(session.run(None, {"obs": obs})[0][0])
    errors = np.asarray(observed)-samples["native_observation"]
    term_reports = {}
    for name, detail in meta["observation_terms"].items():
        start = detail["offset"]; end = start+detail["width"]*detail["history_length"]
        term_reports[name] = _report_errors(errors[:, start:end], 5e-5)
    checks = dict(weights=meta["weights"], bindings=meta["bindings"],
        history=_report_errors(errors, 5e-5),
        actions_same_native_observation=_report_errors(np.asarray(native_actions)-samples["native_action"], 1e-4),
        end_to_end_actions=_report_errors(np.asarray(deployed_actions)-samples["native_action"], 2e-4))
    report = dict(passed=all(item["passed"] for item in checks.values()), task=TASK, **CONTRACT,
        checkpoint_iter=meta["checkpoint_iter"], checkpoint_path=str(args.checkpoint),
        checkpoint_sha256=sha256(args.checkpoint), onnx_path=str(args.onnx), onnx_sha256=sha256(args.onnx),
        adapter_sha256=sha256(ROOT/"common/r2v2_reach_policy.py"),
        verifier_sha256=sha256(Path(__file__)), snapshot_sha256=sha256(args.output/"native_snapshots.npz"),
        sample_count=len(samples["qpos"]), episodes=meta["episodes"], checks=checks, observation_terms=term_reports,
        native_mujoco_version=meta["native_mujoco_version"], deployment_mujoco_version=mujoco.__version__,
        coverage="Two actual front-task resets; continuously streamed cubic asymmetric 6-D wrist goals; all 24 term-major ten-frame histories independently reconstructed from qpos/qvel and reference state",
        limitations="Numerical interface parity only. Native rigid-open fingers, no object contacts. Not grasp, payload, balance, or deployment dynamics validation.")
    np.savez_compressed(args.output/"deployment_comparison.npz", observation=observed,
        action_onnx_native=native_actions, action_onnx_deployment=deployed_actions)
    dump(args.output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "onnx", "path-manifest", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--training-root", type=Path, default=ROOT.parent/"AMO_R2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--collect-native", action="store_true")
    parser.add_argument("--reuse-snapshots", action="store_true")
    args = parser.parse_args()
    if args.frames < 90:
        parser.error("At least 90 frames needed for reset and goal/history coverage")
    for key in ("checkpoint", "onnx", "path_manifest", "output", "training_root"):
        setattr(args, key, getattr(args, key).resolve())
    if args.output.is_relative_to(args.training_root):
        parser.error("Output must be outside the read-only training repository")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.collect_native:
        collect_native(args); return 0
    dump(args.output/"report.json", dict(passed=False, status="verification_in_progress"))
    try:
        if not args.reuse_snapshots:
            cmd = [str(args.training_root/".venv/bin/python"), str(Path(__file__).resolve()), "--collect-native"]
            for key in ("checkpoint", "onnx", "path_manifest", "output", "training_root", "device", "frames", "seed"):
                cmd.extend(["--"+key.replace("_", "-"), str(getattr(args, key))])
            subprocess.run(cmd, check=True, cwd=ROOT)
        report = verify_deployment(args)
    except Exception as exc:
        dump(args.output/"report.json", dict(passed=False, status="verification_error", error=f"{type(exc).__name__}: {exc}"))
        raise
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
