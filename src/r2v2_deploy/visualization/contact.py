"""Render an actual native CONTACT recording as H.264; never synthesize physics."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")


def render(source, output, *, fps=15, view="overview"):
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from r2v2_deploy.configuration import digest
    from r2v2_deploy.data.delivery_state import physics_fingerprint
    from r2v2_deploy.data.grasp_recording import body_transform
    from r2v2_deploy.simulation.front_drop_scene import build_front_drop_model
    from r2v2_deploy.tasks.exit_placement import ExitPlacementProgram

    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not 1 <= fps <= 30 or view not in ("overview", "side"):
        raise ValueError("Use 1..30 fps and an explicit camera view")
    audit = json.loads((source / "recording.json").read_text())
    raw = json.loads((source / "full_contact.json").read_text())
    if (
        audit.get("schema") != "native_contact_recording_v1"
        or digest(source / "recording.npz") != audit["recording_sha256"]
        or digest(source / "full_contact.json") != audit["report_sha256"]
        or digest(source / "report.json") != audit["summary_sha256"]
        or audit.get("runtime_teacher") is not False
        or audit.get("runtime_ik") is not False
        or audit.get("runtime_state_restore") is not False
    ):
        raise ValueError("Native single-student recording identity mismatch")
    program = ExitPlacementProgram(audit["program_path"])
    if program.sha256 != audit["program_sha256"]:
        raise ValueError("Recording program changed")
    cfg = raw["config"]
    model, _, _ = build_front_drop_model(
        raw["reach_configuration"],
        program.document["scene"],
        "contact",
        delivery_background=cfg["delivery_background"],
        pickup_tray_height_m=cfg["pickup_tray_height_m"],
    )
    if physics_fingerprint(model) != audit["physics_fingerprint"]:
        raise ValueError("Recording scene physics changed")
    with np.load(source / "recording.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    times = data["time_s"]
    if (
        not len(times)
        or data["qpos"].shape != (len(times), model.nq)
        or not np.isfinite(data["qpos"]).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("Invalid recorded state dimensions or chronology")
    state = mujoco.MjData(model)
    for index in (0, len(times) // 2, len(times) - 1):
        state.qpos[:] = data["qpos"][index]
        mujoco.mj_forward(model, state)
        for side_i, side in enumerate(("left", "right")):
            actual = body_transform(state, model.body(side + "_hand_roll_link").id)
            if not np.allclose(
                actual, data["wrist_world"][index, side_i], atol=1e-6, rtol=0
            ):
                raise ValueError("Recorded wrist FK differs from scene")
        if not np.allclose(
            state.xpos[model.body("test_cylinder").id],
            data["object_position"][index],
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError("Recorded bottle FK differs from scene")
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = (
        (2.8, 135.0, -18.0) if view == "overview" else (2.8, 0.0, -8.0)
    )
    camera.lookat[:] = [0.25, 0.2, 0.85]
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    renderer = mujoco.Renderer(model, width=854, height=480)
    requested = np.arange(times[0], times[-1] + 0.5 / fps, 1 / fps)
    indices = np.clip(np.searchsorted(times, requested), 0, len(times) - 1)
    output.mkdir(parents=True)
    video = output / "contact.mp4"
    font = ImageFont.load_default(size=17)
    try:
        with imageio.get_writer(
            video,
            fps=fps,
            codec="libx264",
            quality=7,
            macro_block_size=1,
            ffmpeg_params=[
                "-threads",
                "2",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        ) as writer:
            for index in indices:
                state.qpos[:] = data["qpos"][index]
                mujoco.mj_forward(model, state)
                renderer.update_scene(state, camera=camera, scene_option=options)
                frame = Image.new("RGB", (854, 560), (20, 28, 36))
                frame.paste(Image.fromarray(renderer.render()), (0, 56))
                draw = ImageDraw.Draw(frame)
                draw.text(
                    (10, 5),
                    "CONTACT | SINGLE 28-DOF NETWORK | "
                    + ("FULL SUCCESS" if raw["success"] else "FAILED"),
                    font=font,
                    fill="yellow",
                )
                draw.text(
                    (10, 28),
                    f"{times[index]:.2f}s | {data['phase'][index]} | 1x",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (10, 537),
                    "Independent fingers | No teacher / IK | Actual physical recording",
                    font=font,
                    fill="white",
                )
                writer.append_data(np.asarray(frame))
            frame.save(output / "final.png")
    finally:
        renderer.close()
    result = dict(
        schema="native_contact_video_v1",
        video_sha256=digest(video),
        recording_sha256=audit["recording_sha256"],
        report_sha256=audit["report_sha256"],
        physical_success=raw["success"],
        view=view,
        fps=fps,
        frames=len(indices),
        source_physics_duration_s=float(times[-1]),
        playback_speed=1,
        terminal_freeze_frames=0,
        sampled_fk_verified=True,
        rendering_is_not_new_physical_evidence=True,
    )
    (output / "video.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--view", choices=("overview", "side"), default="overview")
    args = parser.parse_args(argv)
    print(json.dumps(render(args.source, args.output, fps=args.fps, view=args.view)))


if __name__ == "__main__":
    main()
