"""Read-only CPU parity for payload archive sampling and shared RPY ramp."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent/'AMO_R2/src'))
from r2v2_loco.tasks.reach.wrist_path_data import WristPathData
from r2v2_loco.tasks.reach.wrist_payload_commands import rigid_payload_neighborhood


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    path = WristPathData(manifest_path=args.path/'manifest.json', device='cpu', dtype=torch.float64)
    with np.load(args.path/path.manifest['trajectory_file'], allow_pickle=False) as data:
        times = data['time_s']+path.preparation_duration_s
        transforms = data['T_world_wrist_goal'].copy()
        phases = data['phase'].copy()
    p, q, phase = path.sample(times)
    rotation = Rotation.from_quat(q.numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(-1, 2, 3, 3)
    position_error = float(np.max(np.abs(p.numpy()-transforms[:, :, :3, 3])))
    orientation_error = float(np.max(np.abs(rotation-transforms[:, :, :3, :3])))
    # phase_start is itself formed by adding preparation, so archive-frame
    # phase identity is exact apart from no more than 1e-9 of a boundary.
    mismatches = [i for i, v in enumerate(phase.tolist()) if path.phase_names[v] != phases[i]]
    assert all(float(torch.min(torch.abs(path.phase_start_s-times[i]))) < 1e-9 for i in mismatches)
    indices = np.linspace(0, len(times)-1, 101).round().astype(int)
    close = path.phase_names.index('CLOSE_SEAT')
    source_p, source_q, _ = path.sample(times[indices])
    alpha = ((torch.tensor(times[indices])-path.phase_start_s[close])
             /(path.phase_end_s[close]-path.phase_start_s[close])).clamp(0., 1.)
    alpha = alpha.square()*(3.-2.*alpha)
    anchor = path.anchor_world
    maximum_p = maximum_r = 0.
    for signs in itertools.product((-1., 1.), repeat=6):
        xyz = source_p.new_tensor(signs[:3])* .002
        rpy = source_p.new_tensor(signs[3:])*np.deg2rad(.5)
        actual_p, actual_q = rigid_payload_neighborhood(source_p, source_q,
            xyz.expand(len(indices), -1), rpy.expand(len(indices), -1), alpha, anchor)
        delta_r = Rotation.from_euler('xyz', alpha.numpy()[:, None]*rpy.numpy()).as_matrix()
        expected_p = np.einsum('nij,nkj->nki', delta_r, source_p.numpy()-anchor.numpy())
        expected_p += anchor.numpy()+alpha.numpy()[:, None, None]*xyz.numpy()
        original_r = Rotation.from_quat(source_q.numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(-1, 2, 3, 3)
        expected_r = delta_r[:, None]@original_r
        actual_r = Rotation.from_quat(actual_q.numpy().reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(-1, 2, 3, 3)
        maximum_p = max(maximum_p, float(np.max(np.abs(expected_p-actual_p.numpy()))))
        maximum_r = max(maximum_r, float(np.max(np.abs(expected_r-actual_r))))
    report = dict(scope='CPU numerical parity only; no learned policy or dynamics run',
        trajectory_sha256=path.sha256, path_manifest_sha256=hashlib.sha256((args.path/'manifest.json').read_bytes()).hexdigest(),
        archive_samples=len(times), phase_names=list(path.phase_names),
        archive_position_max_abs_m=position_error, archive_rotation_matrix_max_abs=orientation_error,
        sub_nanosecond_phase_boundary_mismatches=len(mismatches),
        neighborhood_samples=101*64, neighborhood_position_max_abs_m=maximum_p,
        neighborhood_rotation_matrix_max_abs=maximum_r,
        passed=max(position_error, orientation_error, maximum_p, maximum_r) < 1e-10)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
