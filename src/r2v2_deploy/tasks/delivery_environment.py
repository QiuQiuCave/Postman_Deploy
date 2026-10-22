"Static delivery background and an optional tabletop-integrated pickup tray.\n\nNo robot state, policy, object pose or contact acceptance is changed here.\nThe tray floor is the existing physical tabletop, not an extra support force.\n"

import copy

import xml.etree.ElementTree as ET

import numpy as np

from r2v2_deploy.resources import PROJECT_ROOT


def add_delivery_environment(root, scene, *, pickup_tray_height_m=0.0):
    world = root.find("worldbody")
    table = ET.SubElement(world, "body", name="dropoff_table", pos=".65 -2.5 1.09")
    ET.SubElement(
        table,
        "geom",
        name="dropoff_table_top",
        type="box",
        size=".195 .26 .02",
        rgba=".48 .34 .22 1",
        friction="1 .005 .0001",
    )
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        ET.SubElement(
            table,
            "geom",
            name=f"dropoff_table_leg_{i}",
            type="box",
            pos=f"{sx * 0.183} {sy * 0.248} -.555",
            size=".012 .012 .535",
            rgba=".25 .28 .30 1",
            friction="1 .005 .0001",
        )
    spec = scene.get("delivery_dropoff_table")
    if spec is not None:
        center = np.asarray(spec["center_world_m"], dtype=float)
        half = np.asarray(spec["half_size_m"], dtype=float)
        if (
            center.shape != (3,)
            or half.shape != (3,)
            or (not np.isfinite(center).all())
            or (not np.isfinite(half).all())
            or np.any(half <= 0)
            or (center[2] <= half[2])
        ):
            raise ValueError("Invalid explicit static delivery table geometry")
        table.set("pos", " ".join(map(str, center)))
        table.find('./geom[@name="dropoff_table_top"]').set(
            "size", " ".join(map(str, half))
        )
        for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            leg = table.find(f'./geom[@name="dropoff_table_leg_{i}"]')
            leg.set(
                "pos",
                " ".join(
                    map(
                        str,
                        [
                            sx * (half[0] - 0.015),
                            sy * (half[1] - 0.015),
                            -center[2] / 2,
                        ],
                    )
                ),
            )
            leg.set("size", " ".join(map(str, [0.012, 0.012, (center[2] - 0.02) / 2])))
    source = ET.parse(PROJECT_ROOT / "assets/shelf/shelf.xml").getroot()
    root.find("asset").extend(copy.deepcopy(list(source.find("asset"))))
    shelf = copy.deepcopy(source.find('./worldbody/body[@name="shelf"]'))
    shelf.set("pos", ".65 -3.65 0")
    shelf.set("euler", "0 0 -1.5707963267948966")
    for geom in shelf.iter("geom"):
        geom.set("name", "shelf_" + geom.get("name"))
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
    world.append(shelf)
    layout = dict(
        background_static=True,
        walking_executed=False,
        shelf_asset="assets/shelf/shelf.xml",
        pickup_tray=None,
    )
    if pickup_tray_height_m:
        center = np.asarray(scene["can_initial_world_m"][:2]) + [0.075, 0.0]
        depth, width, wall = (0.25, 0.23, 0.004)
        top = scene["tabletop_height_m"]
        side_height = max(pickup_tray_height_m, 0.06)
        tray = ET.SubElement(
            world.find('./body[@name="tabletop"]'),
            "body",
            name="pickup_tray",
            pos=" ".join(map(str, np.r_[center, top] - scene["table_center_world_m"])),
        )
        for name, pos, size in (
            (
                "front",
                [-depth / 2 + wall / 2, 0, pickup_tray_height_m / 2],
                [wall / 2, width / 2, pickup_tray_height_m / 2],
            ),
            (
                "back",
                [depth / 2 - wall / 2, 0, side_height / 2],
                [wall / 2, width / 2, side_height / 2],
            ),
            (
                "left",
                [0, width / 2 - wall / 2, side_height / 2],
                [depth / 2 - wall, wall / 2, side_height / 2],
            ),
            (
                "right",
                [0, -width / 2 + wall / 2, side_height / 2],
                [depth / 2 - wall, wall / 2, side_height / 2],
            ),
        ):
            ET.SubElement(
                tray,
                "geom",
                name="pickup_tray_" + name,
                type="box",
                pos=" ".join(map(str, pos)),
                size=" ".join(map(str, size)),
                rgba=".15 .48 .38 1",
                friction="1 .005 .0001",
            )
        layout["pickup_tray"] = dict(
            center_xy_m=center.tolist(),
            depth_m=depth,
            width_m=width,
            wall_m=wall,
            height_m=side_height,
            front_height_m=pickup_tray_height_m,
            floor_height_m=top,
            fixed_to_table=True,
            floor_shared_with_table=True,
        )
    return layout
