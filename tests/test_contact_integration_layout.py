"""Dependency-free merge guards; do not import CONTACT into the legacy runtime."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/r2v2_deploy"


def test_native_source_and_shared_resources_keep_upstream_hashes():
    profile = json.loads((PACKAGE / "compatibility/runtime_profile.json").read_text())
    for key, root in (("source_files", PACKAGE), ("resource_files", ROOT)):
        for relative, expected in profile[key].items():
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, relative


def test_native_runtime_does_not_import_legacy_modules():
    for path in PACKAGE.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(
                name.split(".")[0] in {"common", "FSM", "deploy_mujoco", "r2v2_description"}
                for name in names
            ), (path, names)


def test_baseline_entry_and_legacy_support_files_are_retained():
    for relative in (
        "common/path_config.py", "FSM/FSM.py", "FSM/FSMState.py", "FSM/R2V2FSM.py",
        "common/r2v2_loco_cola.py", "common/r2v2_loco_policy.py",
        "common/r2v2_reach_policy.py", "deploy_mujoco/r2v2_loco_cola.py",
        "deploy_mujoco/config/r2v2_loco_cola_straight.json",
        "pyproject.toml", "uv.lock", "requirements-r2v2-reach-sim.txt",
    ):
        assert (ROOT / relative).is_file(), relative
