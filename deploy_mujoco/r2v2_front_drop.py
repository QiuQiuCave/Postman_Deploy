"""Record the final frontal Reach policy with independent binary grasp/drop FSM.

AIR never commands closure; CONTACT requires a matching deployment AIR report.
Exit 0 means artifacts were recorded, not successful pickup or crate deposit.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MUJOCO_GL', 'egl')

from deploy_mujoco.r2v2_tabletop_demo import _write_json, _json_safe
from deploy_mujoco.r2v2_top_grasp_fullbody import _render_frame, _camera_record, PANEL_WIDTH, PANEL_HEIGHT


def cameras():
    import mujoco
    result = []
    for distance, azimuth, elevation, target in (
            (3.1, 90., 0., (.16, .12, .88)), (1.72, 135., -25., (.29, .13, 1.1))):
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
        camera.lookat[:] = target
        result.append(camera)
    return result


def picture(exp, renderer, cams, options, font, freeze=False):
    import numpy as np
    from PIL import Image, ImageDraw
    base = _render_frame(exp, renderer, cams, options, font, exp.mode, terminal_freeze=freeze)
    image = Image.fromarray(base); draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1280, 32), fill=(20, 28, 36))
    scope = 'AIR / NO GRASP' if exp.mode == 'air' else 'REAL CONTACT / NO ASSISTANCE'
    draw.text((14, 8), f'FRONT PICK -> CRATE DROP | model_9999 / 10000 updates | {scope} | {exp.phase} | {exp.data.time:.2f}s',
              font=font, fill=(255, 207, 95))
    draw.rectangle((0, 728, 1280, 752), fill=(20, 28, 36))
    draw.text((14, 729), f'Left finger command {exp.hands.controllers["left"].command} | pickup {exp.grasp_verified} | '
        f'release {exp.release_commanded} | deposited {getattr(exp, "drop_verified", False)} | 50/100/1000 Hz body/hands/physics',
        font=font, fill='white')
    return np.asarray(image)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'config/r2v2_front_drop.json')
    parser.add_argument('--mode', choices=('air', 'contact'), default='air')
    parser.add_argument('--air-evidence', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no-video', action='store_true')
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()): parser.error('Refusing nonempty output directory')
    out.mkdir(parents=True, exist_ok=True)
    exp = renderer = writer = None
    error, captured, count, freeze_count = None, False, 0, 0
    frames, stills = [], []
    try:
        import mujoco
        from common.r2v2_front_drop import FrontDropExperiment, load_config
        cfg = load_config(args.config)
        if args.air_evidence is not None: cfg['air_evidence'] = str(args.air_evidence.resolve())
        exp = FrontDropExperiment(cfg, mode=args.mode)
        cams = cameras()
        if not args.no_video:
            import imageio.v2 as imageio
            from PIL import ImageFont
            renderer = mujoco.Renderer(exp.model, height=PANEL_HEIGHT, width=PANEL_WIDTH)
            writer = imageio.get_writer(str(out/'front_drop.mp4'), fps=25, codec='libx264', quality=8,
                macro_block_size=1, ffmpeg_params=['-movflags', '+faststart', '-threads', '2'])
            options = mujoco.MjvOption(); options.geomgroup[3] = 0
            font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
        next_frame, last_phase = 0., None
        while True:
            changed = exp.phase != last_phase
            if changed: print(f't={exp.data.time:.3f} {exp.phase} {exp.failure or ""}', flush=True)
            due = writer is not None and (exp.data.time >= next_frame-1e-9 or exp.done)
            if renderer is not None and (changed or due):
                frame = picture(exp, renderer, cams, options, font)
                if changed:
                    name = f'{len(stills):02d}_{exp.phase.lower()}.png'
                    imageio.imwrite(out/name, frame); stills.append(dict(time_s=float(exp.data.time), phase=exp.phase, file=name))
                if due:
                    writer.append_data(frame); count += 1; frames.append(float(exp.data.time)); next_frame += .04
            last_phase = exp.phase
            if exp.done:
                captured = True
                if renderer is not None:
                    final = picture(exp, renderer, cams, options, font, freeze=True)
                    imageio.imwrite(out/'final.png', final)
                    for _ in range(50): writer.append_data(final); count += 1; freeze_count += 1
                break
            exp.step()
    except Exception:
        error = traceback.format_exc(); print(error, file=sys.stderr, flush=True)
    finally:
        if writer is not None: writer.close()
        if renderer is not None: renderer.close()
        report = exp.report() if exp is not None else dict(phase='INITIALIZATION', success=False)
        report.update(runtime_error=error, renderer_runtime_completed=captured and error is None,
            renderer_exit_code_semantics='zero means recording completed, not physical success',
            video_file=None if args.no_video else 'front_drop.mp4', video_fps=25, video_frames=count,
            terminal_freeze_frames=freeze_count, terminal_freeze_is_not_simulation=True,
            video_frame_times_s=frames, phase_screenshots=stills,
            camera_configuration=[] if exp is None else _camera_record(cams))
        _write_json(out/'report.json', report)
        _write_json(out/'trace.json', [] if exp is None else exp.samples)
        _write_json(out/'transitions.json', [] if exp is None else exp.transitions)
        _write_json(out/'targets.json', [] if exp is None else exp.targets)
    print(json.dumps(_json_safe({k: report.get(k) for k in ('success', 'air_passed', 'phase', 'failure',
        'duration_s', 'grasp_verified', 'release_commanded', 'drop_verified', 'runtime_error')}), indent=2), flush=True)
    return 0 if captured and error is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
