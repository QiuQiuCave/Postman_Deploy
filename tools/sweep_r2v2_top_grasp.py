"""Bounded CPU sweep of real top-grasp dynamics; never changes shared assets."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def safe(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    return value


def run_case(job):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    from common.r2v2_top_grasp import TopGraspExperiment
    from common.r2v2_top_grasp_scene import TopGraspCandidate
    case, output = job
    candidate = TopGraspCandidate(**case['candidate'])
    try:
        experiment = TopGraspExperiment(profile=case['profile'], candidate=candidate, keep_trace=False)
        while not experiment.done:
            experiment.step()
        report = experiment.report()
        report['outcome'] = 'passed' if report['success'] else 'physical_or_geometry_rejection'
    except (ValueError, RuntimeError) as error:
        report = dict(success=False, outcome='rejected', failure=f'{type(error).__name__}: {error}',
                      candidate=asdict(candidate), profile=case['profile'])
    report['case_id'] = case['case_id']
    path = Path(output)/f"{case['case_id']}.json"
    path.write_text(json.dumps(safe(report), indent=2, allow_nan=False)+'\n')
    metrics = report.get('final_metrics', {})
    return dict(case_id=case['case_id'], profile=case['profile'], candidate=case['candidate'],
        success=report['success'], failure=report.get('failure'),
        phase=report.get('phase'), duration_s=report.get('duration_s'),
        grasp_verified=report.get('grasp_verified', False),
        peak_clearance_m=report.get('peaks', {}).get('clearance_m'),
        object_tilt_deg=metrics.get('object_tilt_deg'),
        contacts=metrics.get('fingers_normal_force_N'),
        hand_vertical_force_N=metrics.get('hand_vertical_force_N'), report=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-json', type=Path, required=True,
                        help='JSON list of {case_id,profile,candidate}, all explicitly recorded')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    cases = json.loads(args.cases_json.read_text())
    if not isinstance(cases, list) or not 1 <= len(cases) <= 200:
        parser.error('Require 1..200 explicit candidates')
    names = [case['case_id'] for case in cases]
    if len(set(names)) != len(names) or any(not n or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in n) for n in names):
        parser.error('Unique safe case_id required')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Refusing nonempty sweep output')
    output.mkdir(parents=True, exist_ok=True)
    (output/'cases.json').write_text(json.dumps(cases, indent=2)+'\n')
    results = []
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, 4))) as pool:
        futures = [pool.submit(run_case, (case, str(output))) for case in cases]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
            (output/'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    print(f'{sum(x["success"] for x in results)}/{len(results)} complete controlled pick/place successes', flush=True)


if __name__ == '__main__':
    main()
