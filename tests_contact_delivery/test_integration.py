"""Checks that require the isolated CONTACT environment and local robot assets."""

from pathlib import Path
import subprocess
import sys

import mujoco

from r2v2_deploy.compatibility.release import validate_runtime
from r2v2_deploy.simulation.robot import (
    BODY_JOINTS, JointMap, build_model_xml, hand_names, load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_preserved_upstream_runtime_and_physical_resource_identity():
    profile = validate_runtime()
    assert profile["actor_input_dim"] == 1461
    assert profile["body_action_dim"] == 28


def test_full_robot_compiles_with_disjoint_body_and_hand_channels():
    model = mujoco.MjModel.from_xml_string(build_model_xml(load_config()))
    mapping = JointMap.create(model, BODY_JOINTS)
    hands = JointMap.create(model, hand_names("left") + hand_names("right"))
    assert (model.nq, model.nv, model.nu, model.neq) == (57, 56, 40, 10)
    assert len(set(mapping.actuators)) == 28
    assert len(set(hands.actuators)) == 12
    assert set(mapping.actuators).isdisjoint(hands.actuators)


def test_checkout_entry_works_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/r2v2_contact_delivery.py"), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    assert "--assets" in result.stdout and "--check-only" in result.stdout
