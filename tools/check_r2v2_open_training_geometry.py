"""Compare deployed open FK with the actual rigid-open training asset on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent/'AMO_R2/src'))

from r2v2_description.model import BODY_JOINTS, build_model, initialize_hands
from r2v2_loco.robots.r2v2_wrist_constants import get_r2v2_wrist_source_manifest, get_r2v2_wrist_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT/'reference_motion_bank/r2v2_crate/down20_yaw15_60mm/manifest.json').read_text())['hand_configuration']
    deployment = build_model(cfg)
    training = get_r2v2_wrist_spec().compile()
    models = [deployment, training]
    datas = [mujoco.MjData(model) for model in models]
    initialize_hands(deployment, datas[0], cfg)
    ids = [[model.joint(name).qposadr[0] for name in BODY_JOINTS] for model in models]
    ranges = [[model.joint(name).range.copy() for name in BODY_JOINTS] for model in models]
    np.testing.assert_array_equal(ranges[0], ranges[1])
    limits = np.array(ranges[0])
    geoms = sorted(model.geom(i).name for model in [training] for i in range(model.ngeom)
                   if model.geom_contype[i] or model.geom_conaffinity[i])
    names = sorted(training.body(i).name for i in range(training.nbody) if training.body(i).name not in (None, 'world'))
    result = dict(scope='CPU FK/inertial/collision-shape equivalence, not dynamics or policy parity',
        hand_open_rad=cfg['hands']['left']['open'], training_source=get_r2v2_wrist_source_manifest(),
        collision_geometry_names=geoms, body_names=names, states=[])
    rng = np.random.default_rng(9182)
    for number in range(3):
        q = np.zeros(28) if number == 0 else rng.uniform(limits[:, 0]*.6+limits[:, 1]*.4,
                                                       limits[:, 0]*.4+limits[:, 1]*.6)
        q = np.clip(q, limits[:, 0]+.05*np.diff(limits).ravel(), limits[:, 1]-.05*np.diff(limits).ravel())
        for model, data, indices in zip(models, datas, ids):
            data.qpos[indices] = q
            root = model.joint('floating_base_joint').qposadr[0]
            data.qpos[root:root+7] = [0., 0., .82, 1., 0., 0., 0.]
            mujoco.mj_forward(model, data)
        errors = dict(geom_position=0., geom_rotation=0., geom_size=0., body_mass=0.,
                      body_world_com=0., body_world_inertia=0.)
        for name in geoms:
            a, b = deployment.geom(name).id, training.geom(name).id
            assert deployment.geom_type[a] == training.geom_type[b]
            for key, av, bv in (
                ('geom_position', datas[0].geom_xpos[a], datas[1].geom_xpos[b]),
                ('geom_rotation', datas[0].geom_xmat[a], datas[1].geom_xmat[b]),
                ('geom_size', deployment.geom_size[a], training.geom_size[b]),
            ):
                errors[key] = max(errors[key], float(np.max(np.abs(av-bv))))
        for name in names:
            a, b = deployment.body(name).id, training.body(name).id
            errors['body_mass'] = max(errors['body_mass'], abs(float(deployment.body_mass[a]-training.body_mass[b])))
            errors['body_world_com'] = max(errors['body_world_com'], float(np.max(np.abs(datas[0].xipos[a]-datas[1].xipos[b]))))
            ra, rb = datas[0].ximat[a].reshape(3, 3), datas[1].ximat[b].reshape(3, 3)
            ia, ib = ra@np.diag(deployment.body_inertia[a])@ra.T, rb@np.diag(training.body_inertia[b])@rb.T
            errors['body_world_inertia'] = max(errors['body_world_inertia'], float(np.max(np.abs(ia-ib))))
        result['states'].append(dict(state_index=number, body_q_rad=q.tolist(), max_absolute_errors=errors,
            passed=all(value < 1e-8 for value in errors.values())))
    result['passed'] = all(state['passed'] for state in result['states'])
    result['collision_filter_note'] = 'Training locks finger joints and excludes rigid internal hand pairs; deployment checks are a conservative superset, not identical contact filtering.'
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'report.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(passed=result['passed'], report=str(args.output/'report.json'),
                         errors=[state['max_absolute_errors'] for state in result['states']])), flush=True)


if __name__ == '__main__':
    main()
