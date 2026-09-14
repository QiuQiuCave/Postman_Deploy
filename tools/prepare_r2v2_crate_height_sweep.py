"""Generate one real-policy, open-hand initial state for all height trials."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.r2v2_crate_height_state import prepare_common_start, save_common_start
from common.r2v2_tabletop_demo import serializable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reach-config", type=Path, required=True)
    p.add_argument("--parity-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Use a new or empty preparation directory")
    exp=prepare_common_start(args.reach_config,args.parity_report)
    previous=None
    while not exp.done:
        exp.step()
        if exp.phase!=previous:
            print(f"{exp.data.time:.3f} {exp.phase}",flush=True)
            previous=exp.phase
    args.output.mkdir(parents=True,exist_ok=True)
    for name,value in (("report",exp.report()),("trace",exp.samples),("transitions",exp.transitions)):
        with (args.output/f"{name}.json").open("x") as f:
            json.dump(serializable(value),f,indent=2,allow_nan=False)
    if exp.failure:
        raise SystemExit(exp.failure)
    print(save_common_start(exp,args.output),flush=True)


if __name__=="__main__":
    main()
