"""Run independent CPU wrist-fixture candidates, with honest failure records."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import asdict
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common.r2v2_crate_oblique import ObliqueCrateExperiment, ObliqueParameters


def safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    return value


def run_case(item):
    name, candidate, output = item
    candidate = dict(candidate)
    if "active_sides" in candidate:
        candidate["active_sides"] = tuple(candidate["active_sides"])
    start = time.monotonic()
    exp = ObliqueCrateExperiment(ObliqueParameters(**candidate), keep_trace=False)
    insertion = dict(max_crate_tilt_deg=0., max_hand_crate_penetration_m=0.,
                     max_hand_crate_normal_force_N=0., completed=False)
    while not exp.done:
        exp.step()
        if exp.steps % 10 == 0 and exp.phase in ("INSERT", "INSERT_SETTLE"):
            m = exp.current_metrics
            insertion["max_crate_tilt_deg"] = max(insertion["max_crate_tilt_deg"], m["crate_tilt_deg"])
            insertion["max_hand_crate_penetration_m"] = max(insertion["max_hand_crate_penetration_m"], m["max_hand_crate_penetration_m"])
            insertion["max_hand_crate_normal_force_N"] = max(insertion["max_hand_crate_normal_force_N"], sum(m["hands"][s]["normal_force_N"] for s in exp.active_sides))
        if exp.data.time > 25:
            exp.fail("Probe total timeout")
    insertion["completed"] = any(row["state"] == "CLOSE" for row in exp.transitions)
    result = safe(exp.report())
    result.update(name=name, wall_time_s=time.monotonic()-start, insertion=insertion)
    (Path(output)/(name+".json")).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    m = result["final_metrics"]
    return dict(name=name, candidate=candidate, pickup_verified=result["pickup_verified"],
        lift_passed=result["lift_passed"], failure=result["failure"], duration_s=result["duration_s"],
        clearance_mm=m["clearance_m"]*1000, tilt_deg=m["crate_tilt_deg"],
        slip_mm=None if m["grasp_slip_m"] is None else m["grasp_slip_m"]*1000,
        net_load_N={s:m["hands"][s]["vertical_force_N"] for s in ("left", "right")},
        insertion=insertion, report=str(Path(output)/(name+".json")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", type=Path, help="JSON mapping case names to candidate kwargs")
    parser.add_argument("--workers", default=2, type=int)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Refusing nonempty output")
    args.output.mkdir(parents=True, exist_ok=True)
    candidates = {"flat": {}}
    candidates.update({f"tip_{a:+d}": dict(tip_up_deg=a) for a in (-30, -20, -15, 15)})
    candidates.update({f"yaw_{a:+d}": dict(yaw_deg=a) for a in (-20, 20, 30)})
    candidates.update({"roll_-15": dict(roll_deg=-15)})
    if args.cases:
        candidates = json.loads(args.cases.read_text())
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_case, [(n, c, str(args.output.resolve())) for n, c in candidates.items()]):
            rows.append(result)
            print(json.dumps(result, allow_nan=False), flush=True)
            (args.output/"summary.json").write_text(json.dumps(rows, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
