"""Native reference, execution and compatibility checks without robot rollout."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from r2v2_deploy.configuration import digest
from r2v2_deploy.tasks.contact_reach import GoalTube, safety_conditions
from r2v2_deploy.tasks.contact_reach_program import reference_closure
from r2v2_deploy.tasks.physical_validity import stable_table_contact


def make_reference(tmp_path):
    arrays = dict(
        time_s=np.arange(4) * 0.02,
        wrist_target=np.zeros((4, 2, 7)),
        phase=np.array(["HOME"] * 4),
        phase_elapsed_s=np.arange(4) * 0.02,
        hand_commands=np.zeros((4, 2), dtype=int),
    )
    arrays["wrist_target"][..., 3] = 1
    data = tmp_path / "reference.npz"
    np.savez_compressed(data, **arrays)
    manifest = dict(
        schema="contact_reach_reference_v1",
        program_sha256="program",
        derivation="lossless_native_target_projection_v1",
        data_file=data.name,
        data_sha256=digest(data),
        source_data_sha256="source",
        source_group="nominal",
        array_sha256={
            k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
            for k, v in arrays.items()
        },
    )
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_reference_projection_checks_all_arrays_and_program(tmp_path):
    path, m = make_reference(tmp_path)
    tube = GoalTube(path, "program")
    assert tube.targets.shape == (4, 2, 7) and tube.data_sha256 == "source"
    assert tube.provenance["expert_action_labels_used"] is False
    with pytest.raises(ValueError):
        GoalTube(path, "different")
    m["array_sha256"]["hand_commands"] = "0" * 64
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        GoalTube(path, "program")


def test_hand_condition_is_continuous_command_reference_not_grasp_truth():
    class Controller:
        poses = [np.zeros(6), np.ones(6)]
        reference = type("Reference", (), {"position": np.ones(6) * 0.4})()

    assert reference_closure(Controller()) == pytest.approx(0.4)


def test_stable_table_support_is_observation_not_required_hand_command():
    row = dict(
        delivery_table_contact=True,
        delivery_table_xy_margin_m=0.03,
        object_tilt_deg=2.0,
        object_linear_speed_mps=0.001,
        object_angular_speed_radps=0.001,
        crate_object_contact=False,
        floor_contact=False,
    )
    assert stable_table_contact(row)
    row["floor_contact"] = True
    assert not stable_table_contact(row)


def test_source_closure_does_not_import_teacher_or_old_task_entrypoints():
    import ast
    import r2v2_deploy

    root = Path(r2v2_deploy.__file__).parent
    forbidden = (
        "r2v2_deploy.teachers",
        "r2v2_deploy.tasks.front_drop",
        "r2v2_deploy.tasks.top_grasp",
        "r2v2_deploy.data.topbox_terminal_recovery",
        "scripts.",
    )
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.startswith(forbidden) for n in names), (path, names)


def test_runtime_rejects_unrecognized_profile(tmp_path, monkeypatch):
    from r2v2_deploy.compatibility import release

    directory = tmp_path / "compatibility"
    directory.mkdir()
    (directory / "runtime_profile.json").write_text('{"schema":"unknown"}')
    monkeypatch.setattr(release, "PACKAGE", tmp_path)
    with pytest.raises(ValueError, match="Unknown"):
        release.validate_runtime()


def test_runtime_rejects_changed_implementation(tmp_path, monkeypatch):
    from r2v2_deploy.compatibility import release

    directory = tmp_path / "compatibility"
    directory.mkdir()
    (tmp_path / "source.py").write_text("changed")
    (directory / "runtime_profile.json").write_text(
        json.dumps(
            dict(
                schema="native_contact_reach_runtime_v1",
                source_files={"source.py": "0" * 64},
                resource_files={},
                runtime_packages={},
            )
        )
    )
    monkeypatch.setattr(release, "PACKAGE", tmp_path)
    with pytest.raises(ValueError, match="hash mismatch"):
        release.validate_runtime()
