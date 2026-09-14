"""Export complete planned wrist paths, including portions not reached by RL.

These are desired poses only, not successful demonstrations. Scene heights,
frozen anchor and reference-continuous starts are recovered from trial reports.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.r2v2_crate_height_path import HeightPath
from common.r2v2_crate_motion_recording import load_crate_motion
from common.r2v2_reach_sim import sha256
from common.r2v2_tabletop_demo import serializable


def export_trial(trial, output):
    trial, output = Path(trial), Path(output)
    report = json.loads((trial/"report.json").read_text())
    targets = json.loads((trial/"targets.json").read_text())
    if not targets or report.get("path_metadata") is None:
        raise ValueError("No initial reference-continuous path was generated")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite existing planned path")
    meta = report["path_metadata"]
    motion = load_crate_motion(meta["motion_manifest"])
    insert_start = next(e["time_s"] for e in motion.manifest["transitions"] if e["state"] == "INSERT")
    crate_world = np.array(meta["world_anchor"]) @ motion.sample(insert_start)["T_anchor_crate"]
    path = HeightPath(report["delta_z_m"], np.array(targets[0]["T_world_wrist_goal"]),
                      world_crate_pose=crate_world, delta_x=report.get("delta_x_m", 0.))
    records, elapsed = [], 0.
    for segment in path.segments:
        duration = segment["duration_s"]+2.
        count = round(duration/.01)
        for step in range(count):
            local = .01*step
            sample = path.sample(segment["name"], local)
            records.append(dict(time_s=elapsed+local, phase=segment["name"],
                T_world_wrist_goal=sample["T_world_wrist"], T_world_crate_desired=sample["T_world_crate"],
                source_time_s=np.nan if sample["source_time_s"] is None else sample["source_time_s"],
                source_hand_command=sample["hand_command"], screened_hand_command=np.zeros(2,dtype=np.int8)))
        elapsed += duration
    sample = path.sample(path.segments[-1]["name"], path.segments[-1]["duration_s"]+2.)
    records.append(dict(time_s=elapsed, phase="COMPLETE", T_world_wrist_goal=sample["T_world_wrist"],
        T_world_crate_desired=sample["T_world_crate"], source_time_s=sample["source_time_s"],
        source_hand_command=sample["hand_command"], screened_hand_command=np.zeros(2,dtype=np.int8)))
    arrays = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    # Explicit relative transforms make this archive useful for future target
    # sampling without mistaking crate centres for wrist goals.
    arrays["T_crate_wrist_goal"] = np.linalg.inv(arrays["T_world_crate_desired"])[:,None] @ arrays["T_world_wrist_goal"]
    output.mkdir(parents=True,exist_ok=True)
    archive = output/"planned_path.npz"
    with archive.open("xb") as f:
        np.savez_compressed(f,**arrays)
    metadata = dict(schema_version=1, scope="DESIRED complete path; unexecuted suffix is NOT a successful demonstration",
        trajectory_file=archive.name, trajectory_sha256=sha256(archive), samples=len(records), sample_dt_s=.01,
        duration_s=elapsed, starts_at_trial_time_s=targets[0]["time_s"], units="m, s, proper SE3 column transforms",
        source_time_nan_meaning="added approach stage, no source recording time",
        command_meaning="source_hand_command archived only; screened_hand_command always0, no grasp executed",
        delta_z_m=path.delta_z, delta_x_m=path.delta_x, virtual_props=report.get("virtual_props", False),
        path_metadata=path.metadata(), trial_report=str((trial/"report.json").resolve()),
        trial_report_sha256=sha256(trial/"report.json"), trial_completed=report["path_sequence_completed"],
        trial_stopped_at_s=report["duration_s"], trial_failure=report["failure"],
        checkpoint_sha256=report["checkpoint_sha256"], prepared_state_sha256=report["prepared_state_sha256"])
    with (output/"manifest.json").open("x") as f:
        json.dump(serializable(metadata),f,indent=2,allow_nan=False)
    with np.load(archive,allow_pickle=False) as check:
        np.testing.assert_allclose(check["T_world_crate_desired"][:,None] @ check["T_crate_wrist_goal"],
                                   check["T_world_wrist_goal"],atol=1e-10)
        assert np.allclose(np.diff(check["time_s"]),.01)
    return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    for cm in (0,5,10,15,20):
        name=f"down_{cm:02d}cm"
        m=export_trial(args.trials/name,args.output/name)
        print(name,m["samples"],m["duration_s"],flush=True)


if __name__=="__main__":
    main()
