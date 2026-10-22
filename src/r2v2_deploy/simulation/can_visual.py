"A textured aluminium-can skin for the existing free cylinder.\n\nOnly rendering assets and zero-mass, non-colliding geoms are appended.  The\ncaller owns visibility of the original collider; this helper never changes\nits attributes, the body's inertial properties, or any simulation settings.\n"

from r2v2_deploy.resources import PROJECT_ROOT

import math

from pathlib import Path

import xml.etree.ElementTree as ET

_PREFIX = "r2v2_cola_can"

_SEGMENTS = 96


def _numbers(values):
    return " ".join((format(float(value), ".12g") for value in values))


def _mesh(asset, name, vertices, faces, texcoords=None):
    attributes = {
        "name": name,
        "vertex": _numbers((value for vertex in vertices for value in vertex)),
        "face": " ".join((str(index) for face in faces for index in face)),
        "smoothnormal": "true",
    }
    if texcoords is not None:
        attributes["texcoord"] = _numbers((value for uv in texcoords for value in uv))
    ET.SubElement(asset, "mesh", **attributes)


def _skin_mesh(asset, radius, half_height):
    rings = ((-0.985, 0.96), (-0.945, 0.996), (0.925, 0.996), (0.985, 0.96))
    vertices, texcoords, faces = ([], [], [])
    stride = _SEGMENTS + 1
    for height_fraction, radius_fraction in rings:
        for index in range(stride):
            u = index / _SEGMENTS
            angle = 2 * math.pi * u - math.pi
            vertices.append(
                (
                    radius * radius_fraction * math.cos(angle),
                    radius * radius_fraction * math.sin(angle),
                    half_height * height_fraction,
                )
            )
            texcoords.append((u, 1 - (height_fraction + 0.985) / 1.97))
    for ring in range(len(rings) - 1):
        for index in range(_SEGMENTS):
            lower = ring * stride + index
            upper = lower + stride
            faces.extend(((lower, lower + 1, upper + 1), (lower, upper + 1, upper)))
    bottom, top = (len(vertices), len(vertices) + 1)
    vertices.extend(((0, 0, rings[0][0] * half_height), (0, 0, rings[-1][0] * half_height)))
    texcoords.extend(((0.5, 1), (0.5, 0)))
    for index in range(_SEGMENTS):
        faces.append((bottom, index + 1, index))
        upper = (len(rings) - 1) * stride + index
        faces.append((top, upper, upper + 1))
    name = f"{_PREFIX}_shell_mesh"
    _mesh(asset, name, vertices, faces, texcoords)
    return name


def _ring_mesh(asset, name, radius_x, radius_y, tube_radius):
    """Closed elliptical torus; a thin version also forms the pull-tab."""
    vertices, faces = ([], [])
    cross_segments = 8
    for index in range(_SEGMENTS):
        angle = 2 * math.pi * index / _SEGMENTS
        for cross in range(cross_segments):
            section = 2 * math.pi * cross / cross_segments
            radial = tube_radius * math.cos(section)
            vertices.append(
                (
                    (radius_x + radial) * math.cos(angle),
                    (radius_y + radial) * math.sin(angle),
                    tube_radius * math.sin(section),
                )
            )
    for index in range(_SEGMENTS):
        for cross in range(cross_segments):
            a = index * cross_segments + cross
            b = (index + 1) % _SEGMENTS * cross_segments + cross
            c = (index + 1) % _SEGMENTS * cross_segments + (cross + 1) % cross_segments
            d = index * cross_segments + (cross + 1) % cross_segments
            faces.extend(((a, b, c), (a, c, d)))
    _mesh(asset, name, vertices, faces)
    return name


def _visual_geom(cylinder, suffix, **attributes):
    ET.SubElement(
        cylinder,
        "geom",
        name=f"{_PREFIX}_{suffix}",
        group="0",
        mass="0",
        density="0",
        contype="0",
        conaffinity="0",
        **attributes,
    )


def add_can_visual(root, cylinder, texture_path) -> None:
    """Append a can skin, lids, rims and a pull-tab without changing physics.

    ``root`` is the MJCF root and ``cylinder`` its existing ``test_cylinder``
    body, whose centred, Z-aligned ``cylinder_geom`` supplies the dimensions.
    ``texture_path`` is an existing unfolded label image; relative paths are
    resolved from PROJECT_ROOT.  Explicit mesh UVs cover the complete image:
    u wraps round the circumference, v goes top-to-bottom, and the label
    logo centres at u=.25/.75 face local -Y/+Y. A square image is suitable;
    no generated UVs or
    texture repetition are used.  All visuals fit inside the original
    radius and half-height.  This function deliberately leaves the original
    collider's rgba unchanged, so the caller should hide that rendering.
    """
    collider = cylinder.find('./geom[@name="cylinder_geom"]')
    if cylinder.tag != "body" or collider is None or collider.get("type") != "cylinder":
        raise ValueError("Can visuals require a body with cylinder_geom of type cylinder")
    if any((float(value) != 0 for value in collider.get("pos", "0 0 0").split())):
        raise ValueError("Can collider must be centred in its body frame")
    if any(
        (
            key in collider.attrib
            for key in ("quat", "axisangle", "euler", "xyaxes", "zaxis", "fromto")
        )
    ):
        raise ValueError("Can collider must use the body's Z axis")
    dimensions = [float(value) for value in collider.get("size", "").split()]
    if len(dimensions) != 2 or not all((math.isfinite(v) and v > 0 for v in dimensions)):
        raise ValueError("Can collider size must be a positive radius and half-height")
    radius, half_height = dimensions
    path = Path(texture_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Can label texture is missing: {path}")
    if root.find(f'.//*[@name="{_PREFIX}_label_texture"]') is not None:
        raise ValueError("Can visuals have already been added to this MJCF")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name=f"{_PREFIX}_label_texture", type="2d", file=str(path))
    label_material, metal_material = (f"{_PREFIX}_label", f"{_PREFIX}_aluminium")
    ET.SubElement(
        asset,
        "material",
        name=label_material,
        texture=f"{_PREFIX}_label_texture",
        texuniform="false",
        texrepeat="1 1",
        rgba="1 1 1 1",
        specular="0.35",
        shininess="0.35",
        reflectance="0",
    )
    ET.SubElement(
        asset,
        "material",
        name=metal_material,
        rgba="0.72 0.75 0.78 1",
        specular="0.85",
        shininess="0.75",
        reflectance="0.05",
    )
    shell = _skin_mesh(asset, radius, half_height)
    rim = _ring_mesh(asset, f"{_PREFIX}_rim_mesh", radius * 0.9675, radius * 0.9675, radius * 0.025)
    tab = _ring_mesh(asset, f"{_PREFIX}_tab_mesh", radius * 0.145, radius * 0.225, radius * 0.007)
    _visual_geom(cylinder, "shell", type="mesh", mesh=shell, material=label_material)
    for side, direction in (("top", 1), ("bottom", -1)):
        _visual_geom(
            cylinder,
            f"{side}_lid",
            type="cylinder",
            material=metal_material,
            size=_numbers((radius * 0.961, half_height / 600)),
            pos=_numbers((0, 0, direction * half_height * 0.99)),
        )
        _visual_geom(
            cylinder,
            f"{side}_rim",
            type="mesh",
            mesh=rim,
            material=metal_material,
            pos=_numbers((0, 0, direction * half_height * 0.99)),
        )
    _visual_geom(
        cylinder,
        "aperture",
        type="ellipsoid",
        rgba="0.12 0.14 0.16 1",
        size=_numbers((radius * 0.17, radius * 0.26, half_height / 3000)),
        pos=_numbers((0, -radius * 0.3, half_height * 0.992)),
    )
    _visual_geom(
        cylinder,
        "pull_tab",
        type="mesh",
        mesh=tab,
        material=metal_material,
        pos=_numbers((0, radius * 0.115, half_height * 0.994)),
    )
    _visual_geom(
        cylinder,
        "tab_rivet",
        type="cylinder",
        material=metal_material,
        size=_numbers((radius * 0.0325, half_height / 600)),
        pos=_numbers((0, -radius * 0.09, half_height * 0.994)),
    )


__all__ = ["add_can_visual"]
