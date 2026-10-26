"""Independent open-top crate with external vertical cylindrical side grips.

Same 36 x 24 x 16 cm shell as the beverage scene, but solid side walls instead
of through holes. One rigid body; two cylinders and four connecting rods are
real collision geoms. This does not modify or enable any existing crate task.
"""
from dataclasses import dataclass, fields
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from common.path_config import PROJECT_ROOT
from common.r2v2_crate import _finite_scalar, _text, _vector


DEFAULT_CONFIG = PROJECT_ROOT / "deploy_mujoco/config/r2v2_crate_vertical_grips.yaml"


@dataclass(frozen=True)
class VerticalGripCrateParameters:
    depth: float = .24
    width: float = .36
    height: float = .16
    bottom_thickness: float = .005
    wall_thickness: float = .004
    grip_diameter: float = .04
    grip_length: float = .12
    wall_grip_clearance: float = .04
    connector_diameter: float = .012
    mass: float = .4
    grip_assembly_mass_fraction: float = .20
    rgba: tuple = (.12, .28, .45, 1.)
    grip_rgba: tuple = (.95, .43, .10, 1.)
    connector_rgba: tuple = (.65, .23, .055, 1.)

    def __post_init__(self):
        for name in ("depth", "width", "height", "bottom_thickness", "wall_thickness",
                     "grip_diameter", "grip_length", "wall_grip_clearance", "connector_diameter", "mass"):
            value = _finite_scalar(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        if 2*self.wall_thickness >= min(self.depth, self.width) or self.bottom_thickness >= self.height:
            raise ValueError("Walls/bottom leave no cavity")
        if self.grip_length >= self.height or self.grip_diameter >= self.depth:
            raise ValueError("Grips must fit the side wall height/depth")
        if self.connector_diameter >= min(self.grip_diameter, self.grip_length/2):
            raise ValueError("Connector rods leave no free grasp span")
        fraction = _finite_scalar(self.grip_assembly_mass_fraction, "grip_assembly_mass_fraction")
        if not 0 < fraction < 1:
            raise ValueError("Grip mass fraction must be between 0 and 1")
        object.__setattr__(self, "grip_assembly_mass_fraction", fraction)
        for name in ("rgba", "grip_rgba", "connector_rgba"):
            value = _vector(getattr(self, name), 4, name)
            if min(value) < 0 or max(value) > 1:
                raise ValueError("Color channels must be within [0, 1]")
            object.__setattr__(self, name, value)

    @property
    def grip_center_abs_y(self):
        return self.width/2+self.wall_grip_clearance+self.grip_diameter/2

    @property
    def overall_width(self):
        return 2*self.grip_center_abs_y+self.grip_diameter

    @property
    def unobstructed_grip_span(self):
        return self.grip_length-2*self.connector_diameter


def load_vertical_grip_crate_config(path=DEFAULT_CONFIG):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT/path
    values = yaml.safe_load(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Expected a flat crate config mapping")
    unknown = set(values)-{f.name for f in fields(VerticalGripCrateParameters)}
    if unknown:
        raise ValueError(f"Unknown fields: {sorted(unknown)}")
    return VerticalGripCrateParameters(**values)


def add_vertical_grip_crate(root, params=None, position=(0, 0, 0),
                           quaternion=(1, 0, 0, 0), free=True):
    """Append reusable geometry; same crate/body/site names, explicit opt-in.

    Total mass defaults to 0.4 kg for controlled asset comparisons. Shell is
    assigned 80%, the two grip assemblies 20%. Rods penetrate the shell/grips
    to model their rigid attachment; explicit component mass distribution is
    an approximation, not a claim of uniform density over a Boolean union.
    """
    p = load_vertical_grip_crate_config() if params is None else params
    if not isinstance(p, VerticalGripCrateParameters) or root.tag != "mujoco":
        raise ValueError("Expected MJCF root and VerticalGripCrateParameters")
    if root.find('.//body[@name="cargo_crate"]') is not None:
        raise ValueError("cargo_crate already exists")
    if not isinstance(free, bool):
        raise ValueError("free must be bool")
    pos = _vector(position, 3, "position")
    quat = _vector(quaternion, 4, "quaternion")
    scale = max(abs(q) for q in quat)
    if scale == 0:
        raise ValueError("Quaternion must be nonzero")
    normalized = [q/scale for q in quat]
    norm = math.hypot(*normalized)
    normalized = [q/norm for q in normalized]
    shell, grips = [], []

    def box(name, center, size):
        shell.append((name, dict(type="box", pos=_text(center), size=_text(v/2 for v in size)),
                      math.prod(size), p.rgba))

    def cylinder(name, start, end, radius, color):
        length = math.dist(start, end)
        grips.append((name, dict(type="cylinder", fromto=_text((*start, *end)), size=str(radius)),
                      math.pi*radius*radius*length, color))

    b, t = p.bottom_thickness, p.wall_thickness
    box("crate_bottom", (0, 0, b/2), (p.depth, p.width, b))
    for label, sign in (("front", 1), ("back", -1)):
        box(f"crate_{label}_wall", (sign*(p.depth-t)/2, 0, (p.height+b)/2),
            (t, p.width, p.height-b))
    for label, sign in (("left", 1), ("right", -1)):
        box(f"crate_{label}_wall", (0, sign*(p.width-t)/2, (p.height+b)/2),
            (p.depth-2*t, t, p.height-b))
        y = sign*p.grip_center_abs_y
        low, high = (p.height-p.grip_length)/2, (p.height+p.grip_length)/2
        cylinder(f"crate_{label}_vertical_grip", (0, y, low), (0, y, high),
                 p.grip_diameter/2, p.grip_rgba)
        for end, z in (("lower", low+p.connector_diameter/2),
                       ("upper", high-p.connector_diameter/2)):
            cylinder(f"crate_{label}_{end}_connector", (0, sign*(p.width/2-t/2), z),
                     (0, y, z), p.connector_diameter/2, p.connector_rgba)

    body = ET.Element("body", name="cargo_crate", pos=_text(pos), quat=_text(normalized))
    if free:
        ET.SubElement(body, "freejoint", name="crate_free")
    for components, total_mass in ((shell, p.mass*(1-p.grip_assembly_mass_fraction)),
                                   (grips, p.mass*p.grip_assembly_mass_fraction)):
        volume = sum(part[2] for part in components)
        for name, attrs, part_volume, color in components:
            ET.SubElement(body, "geom", name=name, mass=str(total_mass*part_volume/volume),
                contype="1", conaffinity="1", group="0", condim="3", priority="1",
                friction="1 .005 .0001", solref=".008 1", solimp=".95 .99 .001",
                margin="0", gap="0", rgba=_text(color), **attrs)
    for side, sign in (("left", 1), ("right", -1)):
        ET.SubElement(body, "site", name=f"crate_{side}_handle", size=".003",
            pos=_text((0, sign*p.grip_center_abs_y, p.height/2)), rgba="0 1 0 0")
    for name in ("crate_floor", "crate_placement"):
        ET.SubElement(body, "site", name=name, pos=_text((0, 0, b)), size=".002", rgba="0 1 0 0")
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    world.append(body)
    return body
