"""Validate or run the released single-network CONTACT skill.

Assets are provided separately from source. This entry never selects models
by training return and never changes archived model/report bytes.
"""

import argparse
import json
import os
from pathlib import Path
import time


def expand(value):
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, str):
        result = os.path.expandvars(value)
        if "${" in result:
            raise ValueError("Unresolved asset environment variable: " + value)
        return result
    return value


def load_assets(path):
    document = expand(json.loads(Path(path).read_text()))
    if document.get("schema") != "contact_reach_delivery_assets_v2":
        raise ValueError("Explicit delivery asset manifest required")
    for key, value in document.get("environment", {}).items():
        if not key.startswith("R2V2_"):
            raise ValueError("Only R2V2 resource/path settings are permitted")
        os.environ[key] = value
    from r2v2_deploy.configuration import digest

    for row in document["assets"]:
        if digest(Path(row["path"])) != row["sha256"]:
            raise ValueError("Asset hash mismatch: " + row["id"])
    return document


def run(assets, output=None, *, check_only=False, record=False):
    document = load_assets(assets)
    from r2v2_deploy.compatibility.release import validated_candidate
    from r2v2_deploy.tasks.exit_placement import ExitPlacementProgram
    from r2v2_deploy.configuration import digest

    cfg = document["configuration"]
    candidate = document["candidate"]
    program = ExitPlacementProgram(cfg["program"])
    # A path-only alias is independent of, and never overwrites, the archive.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="r2v2-delivery-validate-") as temporary:
        alias = Path(temporary) / "candidate.json"
        alias.write_text(json.dumps(candidate))
        validated_candidate(
            alias,
            program,
            reach_finger_condition_protocol=cfg["reach_finger_condition_protocol"],
        )
    if check_only:
        return dict(
            validated=True,
            asset_id=document["model_asset_id"],
            checkpoint_sha256=digest(candidate["checkpoint"]),
            physical_contact_run=False,
        )
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    alias = root / "candidate.json"
    alias.write_text(json.dumps(candidate, indent=2))
    (root / "resolved_config.json").write_text(json.dumps(cfg, indent=2))
    from r2v2_deploy.evaluation.contact_reach import create_student

    exp = create_student(cfg, alias, root / "physical_assets")
    frames = []

    def capture():
        if not record or (
            frames and abs(frames[-1]["time_s"] - float(exp.data.time)) < 1e-9
        ):
            return
        import numpy as np
        from r2v2_deploy.data.grasp_recording import body_transform

        observation = exp.policy.last_observation
        if observation.shape == (1460,):
            # Before the first actor call only: append the real command
            # condition for recording, without advancing control history.
            observation = np.r_[observation, np.float32(exp.policy.closure_provider())]
        frames.append(
            dict(
                time_s=float(exp.data.time),
                phase=exp.phase,
                qpos=exp.data.qpos.copy(),
                qvel=exp.data.qvel.copy(),
                action=exp.policy.last_action.copy(),
                observation=observation.copy(),
                wrist_world=np.array(
                    [
                        body_transform(exp.scratch, exp.wrist_bodies[s])
                        for s in ("left", "right")
                    ]
                ),
                object_position=exp.scratch.xpos[exp.object_body].copy(),
            )
        )

    capture()
    started = time.monotonic()
    while not exp.done and exp.data.time < 150:
        if time.monotonic() - started > 600:
            raise TimeoutError("CONTACT exceeded the 600s wall-clock guard")
        exp.step()
        if exp.steps % 20 == 0 or exp.done:
            capture()
    raw = exp.report()
    outcome = exp.demonstration_reach_program.report()
    raw.update(
        demonstration_reach=outcome,
        evaluation_execution_protocol=cfg["execution_protocol"],
    )
    report = root / "full_contact.json"
    report.write_text(json.dumps(raw, indent=2, allow_nan=False))
    summary = dict(
        model_asset_id=document["model_asset_id"],
        full_success=bool(
            exp.completed and exp.placement_completed and not exp.failure
        ),
        failure=exp.failure,
        failure_phase=exp.failure_phase,
        duration_s=float(exp.data.time),
        full_report_sha256=digest(report),
        **outcome,
    )
    (root / "report.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    if record:
        import numpy as np

        recording = root / "recording.npz"
        np.savez_compressed(
            recording,
            **{key: np.asarray([f[key] for f in frames]) for key in frames[0]},
        )
        (root / "recording.json").write_text(
            json.dumps(
                dict(
                    schema="native_contact_recording_v1",
                    recording_sha256=digest(recording),
                    report_sha256=digest(report),
                    summary_sha256=digest(root / "report.json"),
                    program_sha256=program.sha256,
                    program_path=str(Path(cfg["program"]).resolve()),
                    frames=len(frames),
                    scene_settings_sha256=digest(cfg["scene_configuration"]),
                    physics_fingerprint=raw["physics_fingerprint"],
                    physical_state_replay_for_rendering_only=True,
                    normal_standing_start=True,
                    single_student_body_policy=True,
                    runtime_teacher=False,
                    runtime_ik=False,
                    runtime_state_restore=False,
                ),
                indent=2,
            )
        )
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Save actual physical states for offline video rendering",
    )
    args = parser.parse_args(argv)
    if not args.check_only and args.output is None:
        parser.error("--output is required for a physical CONTACT run")
    print(
        json.dumps(
            run(
                args.assets, args.output, check_only=args.check_only, record=args.record
            ),
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
