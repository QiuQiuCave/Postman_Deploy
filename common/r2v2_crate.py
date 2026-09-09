"""A free, open-top cargo crate assembled from non-overlapping solids.

The body frame is centred on the *outside bottom* of the crate.  X measures
front/back depth, Y left/right width, and Z height; +Y is the left handle.
There are no welds, mocap bodies, support forces, or robot/controller edits.
"""

from common.path_config import PROJECT_ROOT

from dataclasses import dataclass, fields
import math
from numbers import Real
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "deploy_mujoco/config/r2v2_crate.yaml"


def _finite_scalar(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _vector(value, count, name):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain {count} finite numbers")
    try:
        result = tuple(value)
    except TypeError as error:
        raise ValueError(f"{name} must contain {count} finite numbers") from error
    if len(result) != count:
        raise ValueError(f"{name} must contain {count} finite numbers")
    return tuple(_finite_scalar(item, name) for item in result)


@dataclass(frozen=True)
class CrateParameters:
    """SI-unit dimensions and contact parameters for a single rigid crate.

    The opening is centred along X and lies directly below the top beam.
    Beam thickening extends inward only, preserving the outside dimensions.
    Mass is distributed with uniform density over the actual component
    volumes, including the rounded-beam mesh volume, without double counting.
    """

    depth: float = 0.24
    width: float = 0.36
    height: float = 0.16
    bottom_thickness: float = 0.005
    wall_thickness: float = 0.004
    handle_opening_width: float = 0.120
    handle_opening_height: float = 0.055
    handle_beam_height: float = 0.020
    handle_beam_thickness: float = 0.010
    handle_rounding_radius: float = 0.002
    handle_rounding_segments: int = 6
    mass: float = 0.4
    friction: tuple = (1.0, 0.005, 0.0001)
    condim: int = 3
    solref: tuple = (0.008, 1.0)
    solimp: tuple = (0.95, 0.99, 0.001)
    rgba: tuple = (0.12, 0.28, 0.45, 1.0)

    def __post_init__(self):
        positive_fields = ("depth", "width", "height", "bottom_thickness", "wall_thickness",
                           "handle_opening_width", "handle_opening_height", "handle_beam_height",
                           "handle_beam_thickness", "mass")
        for name in positive_fields:
            value = _finite_scalar(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        radius = _finite_scalar(self.handle_rounding_radius, "handle_rounding_radius")
        if radius < 0 or radius >= min(self.handle_beam_height, self.handle_beam_thickness) / 2:
            raise ValueError("handle_rounding_radius must be nonnegative and below half the beam cross-section")
        object.__setattr__(self, "handle_rounding_radius", radius)
        if (isinstance(self.handle_rounding_segments, bool)
                or not isinstance(self.handle_rounding_segments, int)
                or not 1 <= self.handle_rounding_segments <= 64):
            raise ValueError("handle_rounding_segments must be an integer in [1, 64]")
        if self.inner_depth <= 0 or self.inner_width <= 0:
            raise ValueError("wall_thickness leaves no interior cavity")
        if self.handle_opening_width >= self.inner_depth:
            raise ValueError("handle_opening_width must leave nonzero front/back posts")
        if self.handle_opening_bottom <= self.bottom_thickness:
            raise ValueError("handle opening must lie above the bottom and leave a nonzero lower wall")
        if not self.wall_thickness <= self.handle_beam_thickness < self.width / 2:
            raise ValueError("handle_beam_thickness must thicken inward and leave an interior gap")
        friction = _vector(self.friction, 3, "friction")
        if min(friction) < 0 or friction[0] <= 0:
            raise ValueError("friction must be nonnegative with positive sliding friction")
        object.__setattr__(self, "friction", friction)
        if isinstance(self.condim, bool) or self.condim not in (1, 3, 4, 6) or not isinstance(self.condim, int):
            raise ValueError("condim must be one of 1, 3, 4, 6")
        solref = _vector(self.solref, 2, "solref")
        if min(solref) <= 0:
            raise ValueError("solref time constant and damping ratio must be positive")
        object.__setattr__(self, "solref", solref)
        solimp = _vector(self.solimp, 3, "solimp")
        if not (0 < solimp[0] <= solimp[1] < 1 and solimp[2] > 0):
            raise ValueError("solimp requires 0 < dmin <= dmax < 1 and positive width")
        object.__setattr__(self, "solimp", solimp)
        rgba = _vector(self.rgba, 4, "rgba")
        if min(rgba) < 0 or max(rgba) > 1:
            raise ValueError("rgba channels must lie in [0, 1]")
        object.__setattr__(self, "rgba", rgba)

    @property
    def inner_depth(self):
        return self.depth - 2 * self.wall_thickness

    @property
    def inner_width(self):
        return self.width - 2 * self.wall_thickness

    @property
    def handle_opening_top(self):
        return self.height - self.handle_beam_height

    @property
    def handle_opening_bottom(self):
        return self.handle_opening_top - self.handle_opening_height


def load_crate_config(path=None) -> CrateParameters:
    """Load a flat YAML mapping, rejecting unknown keys and invalid values."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError("Crate config must be a mapping of CrateParameters fields")
    unknown = set(values) - {field.name for field in fields(CrateParameters)}
    if unknown:
        raise ValueError(f"Unknown crate configuration keys: {sorted(map(str, unknown))}")
    return CrateParameters(**values)


def _text(values):
    return " ".join(format(float(value), ".17g") for value in values)


def _rounded_beam_mesh(params):
    """Return a closed convex X extrusion, plus its actual polygon volume."""
    p = params
    half_y, half_z, r = p.handle_beam_thickness / 2, p.handle_beam_height / 2, p.handle_rounding_radius
    polygon = []
    # Counterclockwise in the YZ plane.  Every arc lies *inside* the nominal
    # rectangle, so rounding cannot shrink the through-hole clear dimensions.
    for cy, cz, start in ((half_y-r, half_z-r, 0), (-half_y+r, half_z-r, math.pi/2),
                          (-half_y+r, -half_z+r, math.pi), (half_y-r, -half_z+r, 3*math.pi/2)):
        for step in range(p.handle_rounding_segments + 1):
            angle = start + math.pi / 2 * step / p.handle_rounding_segments
            polygon.append((cy + r * math.cos(angle), cz + r * math.sin(angle)))
    count = len(polygon)
    vertices = [(x, y, z) for x in (-p.inner_depth/2, p.inner_depth/2) for y, z in polygon]
    vertices.extend(((-p.inner_depth/2, 0, 0), (p.inner_depth/2, 0, 0)))
    faces = []
    for index in range(count):
        following = (index + 1) % count
        faces.extend(((index, following, count + following),
                      (index, count + following, count + index),
                      (2*count, following, index),
                      (2*count + 1, count + index, count + following)))
    area = sum(polygon[i][0] * polygon[(i+1) % count][1]
               - polygon[(i+1) % count][0] * polygon[i][1] for i in range(count)) / 2
    mesh = ET.Element("mesh", name="crate_handle_beam_mesh", inertia="exact", smoothnormal="false",
                      vertex=_text(value for vertex in vertices for value in vertex),
                      face=" ".join(str(index) for face in faces for index in face))
    return mesh, area * p.inner_depth


def add_crate(root, params=None, position=(0, 0, 0), quaternion=(1, 0, 0, 0), free=True) -> ET.Element:
    """Append one crate body and return it, without changing existing assets.

    Default ``free=True`` adds exactly one 6-DoF ``crate_free`` joint; passing
    False creates a static fixture only for geometric/hand-insertion previews.
    The 11 collision geoms touch at their boundaries but have no overlapping
    volume. Their explicit masses sum to ``params.mass``; MuJoCo computes the
    assembled inertia from these solids (the convex beam mesh uses exact
    inertia). No extra body inertial or hidden massive visual is added.

    ``crate_left_handle`` / ``crate_right_handle`` sites are the hole centres
    at the side walls' midplanes. ``crate_floor`` and ``crate_placement`` mark
    the interior bottom surface, not the centre of a prospective payload.
    """
    p = load_crate_config() if params is None else params
    if not isinstance(p, CrateParameters):
        raise TypeError("params must be CrateParameters or None")
    pos = _vector(position, 3, "position")
    quat = _vector(quaternion, 4, "quaternion")
    scale = max(abs(value) for value in quat)
    if scale == 0:
        raise ValueError("quaternion must have nonzero norm (w, x, y, z)")
    scaled = tuple(value/scale for value in quat)
    norm = math.hypot(*scaled)
    quat = tuple(value/norm for value in scaled)
    if not isinstance(free, bool):
        raise ValueError("free must be a boolean")
    if root.tag != "mujoco":
        raise ValueError("root must be the MJCF mujoco element")
    if root.find('.//body[@name="cargo_crate"]') is not None:
        raise ValueError("cargo_crate already exists in this MJCF")
    if root.find('.//*[@name="crate_handle_beam_mesh"]') is not None:
        raise ValueError("crate_handle_beam_mesh already exists in this MJCF")

    pieces = []

    def box(name, center, size):
        pieces.append((name, center, {"type": "box", "size": _text(v/2 for v in size)}, math.prod(size)))

    # Front/back walls own all four outer corners; side panels stop at their
    # inner faces. The bottom owns z=[0,b], and every wall starts at z=b.
    b, t = p.bottom_thickness, p.wall_thickness
    box("crate_bottom", (0, 0, b/2), (p.depth, p.width, b))
    for name, sign in (("front", 1), ("back", -1)):
        box(f"crate_{name}_wall", (sign*(p.depth-t)/2, 0, (p.height+b)/2),
            (t, p.width, p.height-b))
    lower_height = p.handle_opening_bottom-b
    post_width = (p.inner_depth-p.handle_opening_width)/2
    post_x = (p.inner_depth+p.handle_opening_width)/4
    hole_z = (p.handle_opening_bottom+p.handle_opening_top)/2
    beam_mesh = None
    if p.handle_rounding_radius:
        beam_mesh, beam_volume = _rounded_beam_mesh(p)
    else:
        beam_volume = p.inner_depth*p.handle_beam_thickness*p.handle_beam_height
    for side, sign in (("left", 1), ("right", -1)):
        side_y = sign*(p.width-t)/2
        box(f"crate_{side}_lower_wall", (0, side_y, (b+p.handle_opening_bottom)/2),
            (p.inner_depth, t, lower_height))
        for end, sx in (("front", 1), ("back", -1)):
            box(f"crate_{side}_{end}_post", (sx*post_x, side_y, hole_z),
                (post_width, t, p.handle_opening_height))
        center = (0, sign*(p.width-p.handle_beam_thickness)/2, p.height-p.handle_beam_height/2)
        attributes = ({"type": "mesh", "mesh": "crate_handle_beam_mesh"} if beam_mesh is not None else
                      {"type": "box", "size": _text((p.inner_depth/2, p.handle_beam_thickness/2,
                                                       p.handle_beam_height/2))})
        pieces.append((f"crate_{side}_handle_beam", center, attributes, beam_volume))
    density = p.mass/sum(volume for _, _, _, volume in pieces)
    body = ET.Element("body", name="cargo_crate", pos=_text(pos), quat=_text(quat))
    if free:
        ET.SubElement(body, "freejoint", name="crate_free")
    for name, center, attributes, volume in pieces:
        ET.SubElement(body, "geom", name=name, pos=_text(center), mass=format(density*volume, ".17g"),
                      density="0", contype="1", conaffinity="1", group="0", condim=str(p.condim),
                      priority="1", friction=_text(p.friction), solref=_text(p.solref), solimp=_text(p.solimp),
                      margin="0", gap="0", rgba=_text(p.rgba), **attributes)
    for side, sign in (("left", 1), ("right", -1)):
        ET.SubElement(body, "site", name=f"crate_{side}_handle", size="0.002", rgba="0.2 0.8 0.3 0",
                      pos=_text((0, sign*(p.width-t)/2, hole_z)), quat="1 0 0 0")
    for name in ("crate_floor", "crate_placement"):
        ET.SubElement(body, "site", name=name, pos=_text((0, 0, b)), size="0.002", rgba="0.2 0.8 0.3 0")
    if beam_mesh is not None:
        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        asset.append(beam_mesh)
    world = root.find("worldbody")
    if world is None:
        world = ET.SubElement(root, "worldbody")
    world.append(body)
    return body


__all__ = ["CrateParameters", "load_crate_config", "add_crate"]
