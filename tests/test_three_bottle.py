"""Resource portability and numerical regression for the complete workflow."""

from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
os.environ["R2V2_RESOURCE_ROOT"] = str(ROOT)

import numpy as np
from r2v2_deploy.tasks.three_bottle.geometry import bounded_step
from r2v2_deploy.tasks.three_bottle.resources import resource, verify_runtime
from r2v2_deploy.tasks.three_bottle.workflow import Workflow


class ThreeBottleTests(unittest.TestCase):
    def test_published_resources(self):
        verify_runtime()
        self.assertTrue(resource("scene.xml").is_file())
        with self.assertRaises(ValueError):
            resource("../../outside")

    def test_bounded_ik_respects_active_joint_limits(self):
        step = bounded_step(
            np.eye(3),
            np.array([2.0, -2.0, 0.5]),
            0.0,
            np.zeros(3),
            np.array([-1.0, -0.2, -0.1]),
            np.array([0.3, 1.0, 1.0]),
        )
        np.testing.assert_allclose(step, [0.3, -0.2, 0.5], atol=1e-12)

    def test_fresh_execution_matches_recorded_physics(self):
        s = Workflow()
        self.assertEqual(s.data.time, 0.0)
        self.assertEqual(s.model.nq, 85)
        self.assertEqual(s.native.INPUT_DIM, 1461)
        self.assertEqual(s.loco.session.get_inputs()[0].shape, [1, 560])
        with np.load(resource("demonstration_50hz.npz")) as expected:
            qpos = expected["physical_qpos"][:50]
            qvel = expected["physical_qvel"][:50]
        for tick in range(1000):
            if tick % 20 == 0:
                np.testing.assert_allclose(
                    s.data.qpos, qpos[tick // 20], rtol=0, atol=1e-10
                )
                np.testing.assert_allclose(
                    s.data.qvel, qvel[tick // 20], rtol=0, atol=1e-10
                )
            s.step()
        self.assertIsNone(s.failure)
        self.assertEqual(s.steps, 1000)
        self.assertEqual(len(s.states), 50)
        self.assertEqual(s.native.last_observation.shape, (1461,))


if __name__ == "__main__":
    unittest.main()
