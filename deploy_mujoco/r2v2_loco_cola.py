"""Render staged Loco -> Reach cola pickup, including unsuccessful trials."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MUJOCO_GL','egl')

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common.r2v2_loco_cola import LocoColaExperiment
from deploy_mujoco.r2v2_tabletop_demo import _json_safe


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'config/r2v2_loco_cola.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stop-after',choices=('approach','pickup','handoff','carry','place'),default='pickup')
    parser.add_argument('--no-video',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source=args.output/'source'; source.mkdir()
    repo=Path(__file__).resolve().parents[1]
    for filename in (Path(__file__).resolve(),repo/'common/r2v2_loco_cola.py',repo/'common/r2v2_loco_policy.py',repo/'common/r2v2_reach_policy.py'):
        dest=source/filename.relative_to(repo);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(filename,dest)
    shutil.copy2(args.config,source/'config.json')
    exp=LocoColaExperiment(json.loads(args.config.read_text()),args.stop_after)
    renderer=writer=None
    error=None
    phases=[]
    if not args.no_video:
        renderer=mujoco.Renderer(exp.model,height=560,width=640)
        writer=imageio.get_writer(str(args.output/'experiment.mp4'),fps=25,codec='libx264',quality=8,
            macro_block_size=1,ffmpeg_params=['-movflags','+faststart','-threads','2'])
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    options=mujoco.MjvOption(); options.geomgroup[3]=0
    cameras=[]
    for azimuth,distance,elevation in ((90,3.1,-5),(35,2.7,-18)):
        cam=mujoco.MjvCamera();cam.azimuth=azimuth;cam.distance=distance;cam.elevation=elevation
        cam.lookat[:]=[.10,.05,.95];cameras.append(cam)

    def frame(freeze=False):
        exp.sync(); row=exp.metrics(); panels=[]
        for cam in cameras:
            if exp.config.get('carry_route')=='backward':
                cam.lookat[:]=[float(exp.scratch.xpos[exp.base,0])+.25,.15,.95]
            renderer.update_scene(exp.scratch,camera=cam,scene_option=options)
            if exp.mode=='reach':
                for side,color in (('left',[.1,1,.2,.8]),('right',[1,.3,.1,.8])):
                    g=renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_SPHERE,np.array([.008]*3),
                        exp.reach.references[side].goal_position,np.eye(3).ravel(),np.array(color,dtype=np.float32))
                    renderer.scene.ngeom+=1
            panels.append(renderer.render().copy())
        canvas=Image.new('RGB',(1280,720),(20,28,36))
        canvas.paste(Image.fromarray(np.concatenate(panels,axis=1)),(0,160))
        draw=ImageDraw.Draw(canvas)
        w=row['wrists']['left']
        lines=[f'LOCO -> REACH -> COLA | {exp.phase} | {exp.data.time:.2f}s | stage: {args.stop_after}',
            f'REAL CONTACT / NO ASSISTANCE | body {exp.mode.upper()} | left hand command {row["left_hand_command"]} | pickup verified {exp.grasp_verified}',
            f'Left wrist error {1000*w["position_error_m"]:.1f}mm / {w["orientation_error_deg"]:.1f}deg (Reach only) | speed {w["speed_mps"]:.3f}m/s',
            f'Can table clearance {1000*row["bottom_clearance_m"]:.1f}mm | opposed contact {row["contacts"]["opposed"]} | slip {row["slip_m"]}',
            'Selected handoff: Loco pitch 15deg; intermediate back/down 2/6cm; blend in/out 0.8/1.2s',
            ('FROZEN FINAL FRAME | ' if freeze else '')+(exp.failure or 'Stage experiment; not yet a qualified two-table closed loop')]
        for i,line in enumerate(lines): draw.text((12,7+25*i),line,font=font,fill=(255,160,110) if exp.failure else 'white')
        return np.asarray(canvas)

    next_frame=0.;last_phase=None
    try:
        while True:
            if renderer is not None and (exp.data.time>=next_frame-1e-9 or exp.done):
                picture=frame()
                writer.append_data(picture);next_frame+=.04
                if exp.phase!=last_phase:
                    filename=f'{len(phases):02d}_{exp.phase.lower()}.png'
                    imageio.imwrite(args.output/filename,picture)
                    phases.append(dict(phase=exp.phase,time_s=float(exp.data.time),file=filename))
                last_phase=exp.phase
            if exp.done: break
            exp.step()
        if renderer is not None:
            picture=frame(True);imageio.imwrite(args.output/'final.png',picture)
            for _ in range(50): writer.append_data(picture)
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
    finally:
        if writer is not None: writer.close()
        if renderer is not None: renderer.close()
        report=exp.report();report.update(runtime_error=error,terminal_freeze_s=0 if args.no_video else 2,phase_screenshots=phases)
        for name,value in [('report',report),('trace',exp.samples),('transitions',exp.transitions)]:
            (args.output/(name+'.json')).write_text(json.dumps(_json_safe(value),indent=2))
        print(json.dumps({k:report[k] for k in ('success','phase','failure','duration_s','grasp_verified','runtime_error')},indent=2),flush=True)
    return int(error is not None)


if __name__=='__main__': raise SystemExit(main())
