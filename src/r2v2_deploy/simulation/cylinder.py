"""Cylinder physical parameters and content-validated profile loading."""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml
from r2v2_deploy.resources import PROJECT_ROOT

DEFAULT_CYLINDER_PROFILE = "baseline_40mm_100g"
CYLINDER_PROFILE_DIRECTORY = PROJECT_ROOT / "deploy_mujoco/config/cylinder_profiles"


def load_cylinder_profile(source=None):
    """Load a named profile, repo-relative YAML path, or resolved mapping.

    The returned plain mapping records both physical values and their
    provenance so a report cannot present an assumed mass as a measurement.
    Missing profiles/fields and unknown fields fail closed; the old baseline
    is selected only when the caller deliberately omits ``source``.
    """
    if source is None:
        source = DEFAULT_CYLINDER_PROFILE
    if isinstance(source, Mapping):
        profile = copy.deepcopy(dict(source))
    else:
        if not isinstance(source, (str, Path)):
            raise ValueError(
                "Cylinder profile source must be a name, YAML path or mapping"
            )
        path = Path(source)
        if not path.suffix and path.parent == Path("."):
            path = CYLINDER_PROFILE_DIRECTORY / (path.name + ".yaml")
        elif not path.is_absolute():
            path = PROJECT_ROOT / path
        profile = yaml.safe_load(path.read_text())
    fields = {
        "schema_version",
        "profile_id",
        "radius_m",
        "height_m",
        "mass_kg",
        "nominal_capacity_ml",
        "geometry_source",
        "mass_provenance",
        "mass_notes",
        "grasp_calibration_status",
        "grasp_calibration_notes",
        "simulation_only",
    }
    if not isinstance(profile, dict) or set(profile) != fields:
        raise ValueError(
            "Cylinder profile must contain exactly the documented profile fields"
        )
    if (
        type(profile["schema_version"]) is not int
        or profile["schema_version"] != 1
        or profile["simulation_only"] is not True
    ):
        raise ValueError(
            "Cylinder profiles require schema_version 1 and simulation_only: true"
        )
    for key in ("radius_m", "height_m", "mass_kg", "nominal_capacity_ml"):
        value = profile[key]
        if key == "nominal_capacity_ml" and value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or (not np.isfinite(value))
            or (value <= 0)
        ):
            raise ValueError(f"Cylinder profile {key} must be a positive finite number")
        profile[key] = float(value)
    for key in (
        "profile_id",
        "geometry_source",
        "mass_provenance",
        "mass_notes",
        "grasp_calibration_status",
        "grasp_calibration_notes",
    ):
        if not isinstance(profile[key], str) or not profile[key].strip():
            raise ValueError(f"Cylinder profile {key} must be a nonempty string")
    if profile["grasp_calibration_status"] not in ("baseline_verified", "unvalidated"):
        raise ValueError("Unknown cylinder grasp_calibration_status")
    if profile["grasp_calibration_status"] == "baseline_verified":
        baseline = CylinderParameters()
        if (profile["radius_m"], profile["height_m"], profile["mass_kg"]) != (
            baseline.radius,
            baseline.height,
            baseline.mass,
        ):
            raise ValueError(
                "baseline_verified applies only to the original 40 mm / 120 mm / 100 g profile"
            )
    return profile


@dataclass
class CylinderParameters:
    side: str = "left"
    x: float = 0.015
    palm_offset: float = 0.035
    z: float = 0.45
    radius: float = 0.02
    height: float = 0.12
    mass: float = 0.1
    close_time: float = 0.5
    withdraw_start: float = 5.0
    withdraw_end: float = 7.0
    release_time: float = 12.0
    duration: float = 15.0
    grasp: bool = True

    @classmethod
    def from_profile(cls, source=None, **experiment_parameters):
        """Use profile dimensions/mass with optional probe-side/timeline settings.

        Physics overrides must be made in an explicit profile rather than
        silently changing the physical interpretation of a named profile.
        The old direct ``CylinderParameters(...)`` interface remains valid.
        """
        if {"radius", "height", "mass"} & experiment_parameters.keys():
            raise ValueError("Put radius, height and mass in the cylinder profile")
        profile = load_cylinder_profile(source)
        return cls(
            radius=profile["radius_m"],
            height=profile["height_m"],
            mass=profile["mass_kg"],
            **experiment_parameters,
        )

    @property
    def half_height(self):
        return self.height / 2

    def upright_center_height(self, surface_height, clearance=0.0):
        """World center height of an upright cylinder above a horizontal surface."""
        if not np.all(np.isfinite([surface_height, clearance])):
            raise ValueError("Surface height and clearance must be finite")
        return float(surface_height + self.half_height + clearance)

    @property
    def center(self):
        return np.array(
            [
                self.x,
                (0.15 - self.palm_offset) * (1 if self.side == "left" else -1),
                self.z,
            ]
        )
