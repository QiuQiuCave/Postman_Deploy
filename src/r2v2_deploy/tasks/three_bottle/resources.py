"""Repository-contained resources for the three-bottle workflow."""

from __future__ import annotations
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
ASSETS = ROOT / "assets/three_bottle"
CONFIG = ROOT / "configs/three_bottle"
SKILLS = ROOT / "policy/r2v2_skills/20261026"


def resource(name: str) -> Path:
    """Resolve a named task resource; all paths stay within this checkout."""
    if name == "@skills":
        return SKILLS
    if name.startswith("@skills/"):
        return SKILLS / name.removeprefix("@skills/")
    if name.endswith(".json") and (CONFIG / name).is_file():
        return CONFIG / name
    path = (ASSETS / name).resolve()
    if not path.is_relative_to(ASSETS.resolve()):
        raise ValueError(f"Invalid task resource: {name}")
    return path


def verify_runtime() -> dict:
    """Validate the numerical runtime and immutable resources before stepping."""
    expected = {
        "numpy": "1.26.4",
        "mujoco": "3.3.7",
        "onnxruntime": "1.29.0",
        "ruckig": "0.12.2",
    }
    actual = {name: importlib.metadata.version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(
            f"Install requirements-three-bottle.txt; expected {expected}, found {actual}"
        )
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    for relative, identity in manifest["files"].items():
        path = ROOT / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != identity["sha256"]:
            raise ValueError(f"Resource checksum mismatch: {relative}")
    return actual
