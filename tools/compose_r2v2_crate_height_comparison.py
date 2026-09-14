"""Compose five policy-driven height trials on a common simulation timeline.

The five right-camera crops use recorded simulation frames, never interpolated
poses. A terminated trial is explicitly frozen for the remaining comparison.
The sixth tile contains provenance and terminal outcomes, not a sixth trial.
"""

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path
import sys
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy_mujoco.r2v2_tabletop_demo import _write_json

FPS = 30
TILE_WIDTH = 640
TILE_HEIGHT = 640
BACKGROUND = (20, 28, 36)
SOURCE_CROP = (640, 160, 1280, 720)


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load_trial(directory):
    directory = Path(directory).resolve()
    report = json.loads((directory / "report.json").read_text())
    if report.get("runtime_error") or report.get("cleanup_errors"):
        raise ValueError(f"Incomplete artifact: {directory}: {report.get('runtime_error')}")
    video = directory / report.get("video_file", "videoheight.mp4")
    if not video.is_file():
        raise ValueError(f"Missing video: {video}")
    times = report.get("video_frame_times_s", [])
    if not times or any(float(b) < float(a) for a, b in zip(times, times[1:])):
        raise ValueError(f"Missing/nonmonotonic source frame times: {directory}")
    trace = json.loads((directory / "trace.json").read_text())
    transitions = json.loads((directory / "transitions.json").read_text())
    return {"directory": directory, "report": report, "video": video,
            "times": [float(x) for x in times], "trace": trace,
            "trace_times": [float(x["time_s"]) for x in trace],
            "transitions": transitions,
            "start_s": float(times[0]), "end_s": float(times[-1]),
            "duration_s": float(times[-1]) - float(times[0])}


def _sample(trial, elapsed):
    simulation_time = min(trial["start_s"] + elapsed, trial["end_s"])
    frame_index = max(0, bisect.bisect_right(trial["times"], simulation_time + 1e-9) - 1)
    trace_index = max(0, bisect.bisect_right(trial["trace_times"], simulation_time + 1e-9) - 1)
    sample = trial["trace"][trace_index] if trial["trace"] else {}
    frozen = elapsed > trial["duration_s"] + 1e-9
    return frame_index, sample, frozen, simulation_time


def _validate_scene_variant(trials):
    """Require one X shift and prop mode; legacy reports mean physical X=0."""
    x_offsets, virtual_modes = set(), set()
    for trial in trials:
        report = trial["report"]
        x_offset = float(report.get("delta_x_m", report.get("x_offset_m", 0.)))
        if "x_offset_m" in report and float(report["x_offset_m"]) != x_offset:
            raise ValueError("Conflicting canonical delta_x_m and renderer x_offset_m")
        virtual = report.get("virtual_props", False)
        if not math.isfinite(x_offset) or not isinstance(virtual, bool):
            raise ValueError("Each trial needs a finite X offset and boolean virtual-prop mode")
        x_offsets.add(x_offset)
        virtual_modes.add(virtual)
    if len(x_offsets) != 1 or len(virtual_modes) != 1:
        raise ValueError("Trials must share one table/crate X offset and virtual-prop mode")
    return next(iter(x_offsets)), next(iter(virtual_modes))


def _draw_summary(trials, font, small_font):
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    x_offset, virtual = _validate_scene_variant(trials)
    draw.text((16, 12), "FIVE HEIGHTS / " + ("VIRTUAL PROPS" if virtual else "REAL CONTACTS")
              + " / NO GRASP", font=font, fill=(255, 207, 95))
    lines = ["Same prepared robot state and path construction.",
             f"Table + crate X shift {x_offset*100:+.0f} cm; negative = closer.",
             ("Static line props; no contacts / prop-collision stops." if virtual else
              "Solid table + crate; real contact response / stops."),
             "Down20 / yaw15 recorded hand-only trajectory.",
             "Robot safety retained; tracking errors not gated.",
             "Common elapsed physics time, no speed adjustment.",
             "Red FREEZE = terminated physics; repeated image."]
    y = 48
    for line in lines:
        draw.text((16, y), line, font=small_font, fill=(205, 215, 225))
        y += 23
    y += 16
    for trial in trials:
        report = trial["report"]
        failure = report.get("failure")
        phase = report.get("failure_phase") or report.get("phase", "unknown")
        draw.text((16, y), f"{float(report['height_offset_m'])*100:+.0f} cm  | "
                  f"{trial['duration_s']:.2f} s  | {phase}", font=font,
                  fill=(255, 145, 125) if failure else (170, 225, 185))
        y += 26
        detail = str(failure) if failure else "Playback ended; see report for actual tracking quality."
        for line in textwrap.wrap(detail, width=67)[:2]:
            draw.text((16, y), line, font=small_font, fill=(205, 215, 225))
            y += 21
        y += 13
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", nargs=5, type=Path, required=True,
                        help="Five renderer output directories, ordered high to low")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", choices=("hands", "body"), default="hands",
                        help="Compare real crate close-ups or full-body side-view crops")
    parser.add_argument("--terminal-hold-seconds", type=float, default=2.)
    args = parser.parse_args()
    source_crop = SOURCE_CROP if args.view == "hands" else (0, 160, 640, 720)
    if not math.isfinite(args.terminal_hold_seconds) or args.terminal_hold_seconds < 0:
        parser.error("terminal hold must be finite and nonnegative")
    out = args.output.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    trials = [_load_trial(path) for path in args.trials]
    try:
        x_offset, virtual = _validate_scene_variant(trials)
    except ValueError as error:
        parser.error(str(error))
    if len({float(t["report"]["height_offset_m"]) for t in trials}) != 5:
        parser.error("Five distinct height offsets are required")
    configuration_hashes = {_sha256(t["report"]["reach_configuration_path"]) for t in trials}
    if len(configuration_hashes) != 1:
        parser.error("Trials must use identical frozen reach configuration")
    policy_hashes = {}
    for key in ("checkpoint_sha256", "onnx_sha256"):
        values = {t["report"].get(key) for t in trials}
        if len(values) != 1 or None in values:
            parser.error(f"Trials must carry one matching frozen {key}")
        policy_hashes[key] = next(iter(values))
    prepared_paths = {t["report"].get("prepared_state_path") for t in trials}
    if len(prepared_paths) != 1 or None in prepared_paths:
        parser.error("Trials must identify one shared prepared state")
    prepared_path = Path(next(iter(prepared_paths)))
    prepared_hashes = {t["report"].get("prepared_state_sha256") for t in trials}
    if len(prepared_hashes) != 1 or None in prepared_hashes:
        parser.error("Trials must carry one matching prepared-state content hash")
    prepared_hash = next(iter(prepared_hashes))
    prepared_manifest_hash = _sha256(prepared_path)
    prepared_description = json.loads(prepared_path.read_text())
    prepared_payload = prepared_path.parent / prepared_description["state_file"]
    if _sha256(prepared_payload) != prepared_hash:
        parser.error("Prepared state changed after trials were recorded")
    out.mkdir(parents=True, exist_ok=True)

    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 20) if font_path.exists() else ImageFont.load_default()
    small_font = ImageFont.truetype(str(font_path), 16) if font_path.exists() else ImageFont.load_default()
    summary = _draw_summary(trials, font, small_font)
    for trial in trials:
        trial["reader"] = imageio.get_reader(str(trial["video"]), format="ffmpeg",
                                            input_params=["-threads", "1"],
                                            output_params=["-threads", "1"])
        trial["read_index"] = -1
        trial["frame"] = None
    duration = max(t["duration_s"] for t in trials) + args.terminal_hold_seconds
    frames = max(1, math.ceil(duration * FPS) + 1)
    writer = imageio.get_writer(str(out / "height_comparison.mp4"), fps=FPS,
                                codec="libx264", quality=8, macro_block_size=1,
                                ffmpeg_params=["-threads", "2"])
    try:
        for frame_number in range(frames):
            elapsed = frame_number / FPS
            canvas = Image.new("RGB", (3 * TILE_WIDTH, 2 * TILE_HEIGHT), BACKGROUND)
            for tile_index, trial in enumerate(trials):
                source_index, sample, frozen, simulation_time = _sample(trial, elapsed)
                while trial["read_index"] < source_index:
                    trial["frame"] = trial["reader"].get_next_data()
                    trial["read_index"] += 1
                tile = Image.new("RGB", (TILE_WIDTH, TILE_HEIGHT), BACKGROUND)
                tile.paste(Image.fromarray(trial["frame"]).crop(source_crop), (0, 48))
                draw = ImageDraw.Draw(tile)
                report = trial["report"]
                phase = sample.get("phase", report.get("phase", "unknown"))
                failure = bool(report.get("failure"))
                end_label = ("FAILED FREEZE" if failure else "PATH ENDED FREEZE") if frozen else phase
                color = (255, 135, 115) if (frozen and failure) or phase == "FAILED" else (255, 207, 95)
                draw.text((10, 5), f"{float(report['height_offset_m'])*100:+.0f} cm  | "
                          f"t={simulation_time - trial['start_s']:.2f}s | {end_label}", font=font, fill=color)
                draw.text((10, 29), f"X {x_offset*100:+.0f}cm | "
                          + ("virtual props | " if virtual else "real contacts | ")
                          + f"elapsed {elapsed:.2f}s | "
                          + ("physics stopped" if frozen else "physics advancing"),
                          font=small_font, fill=(190, 200, 210))
                metrics = sample.get("metrics", {})
                errors = metrics.get("wrist_errors", {})
                values = []
                for side in ("left", "right"):
                    e = errors.get(side, {})
                    values.append(f"{side[0].upper()} {float(e.get('position_m', 0.))*1000:.0f}mm/"
                                  f"{float(e.get('orientation_deg', 0.)):.0f}deg")
                draw.text((10, 614), "  ".join(values) +
                          f"  body {float(metrics.get('base_tilt_deg', 0.)):.1f}deg",
                          font=small_font, fill=(200, 220, 240))
                canvas.paste(tile, ((tile_index % 3) * TILE_WIDTH, (tile_index // 3) * TILE_HEIGHT))
            canvas.paste(summary, (2 * TILE_WIDTH, TILE_HEIGHT))
            array = np.asarray(canvas)
            writer.append_data(array)
            if frame_number == 0:
                imageio.imwrite(out / "initial.png", array)
            if frame_number == frames - 1:
                imageio.imwrite(out / "final.png", array)
    finally:
        writer.close()
        for trial in trials:
            trial["reader"].close()
    _write_json(out / "comparison_manifest.json", {
        "video_file": "height_comparison.mp4", "fps": FPS, "frames": frames,
        "common_elapsed_duration_s": duration,
        "terminal_hold_seconds": args.terminal_hold_seconds,
        "stopped_trials_are_frozen_not_simulated": True,
        "delta_x_m": x_offset, "x_offset_m": x_offset, "virtual_props": virtual,
        "prop_rendering": "static_render_only_component_edges" if virtual else "physical_solids",
        "source_camera_view": args.view, "source_camera_crop_xyxy": source_crop,
        "reach_configuration_sha256": next(iter(configuration_hashes)),
        **policy_hashes,
        "prepared_state_path": str(prepared_path), "prepared_state_sha256": prepared_hash,
        "prepared_state_manifest_sha256": prepared_manifest_hash,
        "trials": [{"directory": str(t["directory"]), "video": str(t["video"]),
                    "video_sha256": _sha256(t["video"]),
                    "height_offset_m": t["report"]["height_offset_m"],
                    "delta_x_m": t["report"].get("delta_x_m", t["report"].get("x_offset_m", 0.)),
                    "x_offset_m": t["report"].get("delta_x_m", t["report"].get("x_offset_m", 0.)),
                    "virtual_props": t["report"].get("virtual_props", False),
                    "physical_start_s": t["start_s"], "physical_end_s": t["end_s"],
                    "terminal_phase": t["report"].get("phase"),
                    "failure": t["report"].get("failure")} for t in trials],
    })
    print(out / "height_comparison.mp4", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
