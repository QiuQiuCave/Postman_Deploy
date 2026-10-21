"""Build the native scene from explicit final settings, never training reports."""

import copy
import json
from pathlib import Path
import yaml


def create_scene(cfg, candidate, output):
    from r2v2_deploy.tasks.exit_placement import ExitPlacementProgram
    from r2v2_deploy.simulation.contact import NativeContactScene
    from r2v2_deploy.data.state_io import record_hand_generator

    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    settings = json.loads(Path(cfg["scene_configuration"]).read_text())
    if settings.get("schema") != "native_contact_scene_settings_v1":
        raise ValueError("Final native scene settings required")
    base = Path(cfg["scene_configuration"]).resolve().parent
    physical = copy.deepcopy(settings["physical_config"])
    for key in ("reach_config", "path_manifest"):
        path = Path(physical[key])
        physical[key] = str(path if path.is_absolute() else base / path)
    reach = yaml.safe_load(Path(physical["reach_config"]).read_text())
    reach.update(policy_path=candidate["onnx"], checkpoint_path=candidate["checkpoint"])
    reach_path = root / "reach.yaml"
    reach_path.write_text(yaml.safe_dump(reach))
    physical.update(
        reach_config=str(reach_path),
        parity_report=candidate["parity"],
        air_evidence=None,
    )
    exp = NativeContactScene(physical, program=ExitPlacementProgram(cfg["program"]))
    for controller in exp.hands.controllers.values():
        record_hand_generator(controller)
    return exp
