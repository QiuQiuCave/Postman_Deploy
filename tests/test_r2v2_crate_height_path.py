import numpy as np
import pytest

from common.r2v2_crate_height_path import (
    ADDED_SEGMENTS, BASE_TABLE_TOP_M, HEIGHT_OFFSETS_M, HeightPath,
    default_start_wrists, inspect_approach_hand_sweep,
)


def test_five_heights_share_source_and_preserve_relative_trajectory():
    paths = [HeightPath(delta) for delta in HEIGHT_OFFSETS_M]
    for path, delta in zip(paths, HEIGHT_OFFSETS_M):
        assert path.table_top_m == pytest.approx(BASE_TABLE_TOP_M+delta)
        np.testing.assert_array_equal(path.start_wrists_world, default_start_wrists())
        for name, elapsed in (("TURN_WRISTS", 2.), ("PREALIGN", 1.5),
                              ("READY", .5), ("INSERT", 1.), ("PROBE_LIFT", 3.)):
            actual = path.sample(name, elapsed)
            baseline = paths[0].sample(name, elapsed)
            np.testing.assert_allclose(actual["T_world_wrist"][:, :3, :3],
                                       baseline["T_world_wrist"][:, :3, :3], atol=1e-12)
            np.testing.assert_allclose(actual["T_world_wrist"][:, :3, 3],
                baseline["T_world_wrist"][:, :3, 3]+[0., 0., delta], atol=1e-12)


def test_source_lift_is_not_erased_by_relative_replay():
    path = HeightPath(-.1)
    early = path.sample("PROBE_LIFT", 0.)
    late = path.sample("PROBE_LIFT", 4.)
    assert late["T_world_crate"][2, 3] - early["T_world_crate"][2, 3] > .09
    assert np.all(late["T_world_wrist"][:, 2, 3] - early["T_world_wrist"][:, 2, 3] > .099)
    for sample in (early, late):
        source = path.motion.sample(sample["source_time_s"])
        np.testing.assert_allclose(np.linalg.inv(sample["T_world_crate"]) @ sample["T_world_wrist"],
                                   source["T_crate_wrist"], atol=1e-12)


def test_segments_continuous_open_until_source_close():
    path = HeightPath(-.15)
    for first, second in zip(path.segments[:-1], path.segments[1:]):
        np.testing.assert_allclose(path.sample(first["name"], first["duration_s"])["T_world_wrist"],
                                   path.sample(second["name"], 0.)["T_world_wrist"], atol=1e-12)
    for name in ADDED_SEGMENTS:
        assert np.array_equal(path.sample(name, 1.)["hand_command"], [0, 0])
        assert path.sample(name, 1.)["source_time_s"] is None
    assert np.array_equal(path.sample("CLOSE", .01)["hand_command"], [1, 1])
    np.testing.assert_allclose(path.sample("OUTSIDE", 999.)["T_world_wrist"], path.outside_wrists_world)


def test_open_hand_approach_clearance_for_all_heights():
    reports = [inspect_approach_hand_sweep(HeightPath(delta), intervals_per_segment=16)
               for delta in HEIGHT_OFFSETS_M]
    for report in reports:
        assert report["passed"]
        assert not report["dynamic_grasp_or_fullbody_success_evaluated"]
        assert min(row["minimum_crate_clearance_lower_bound_m"] for row in report["segments"]) > .02
        assert min(row["minimum_table_clearance_lower_bound_m"] for row in report["segments"]) > .075
    for first, last in zip(reports[0]["segments"], reports[-1]["segments"]):
        if first["segment"] in ("TURN_WRISTS", "PREALIGN"):
            assert first["minimum_tabletop_vertical_clearance_lower_bound_m"] == pytest.approx(
                last["minimum_tabletop_vertical_clearance_lower_bound_m"], abs=1e-10)


def test_invalid_path_inputs_rejected():
    with pytest.raises(ValueError):
        HeightPath(float("nan"))
    with pytest.raises(ValueError):
        HeightPath(-.3)
    with pytest.raises(ValueError):
        HeightPath(0., start_wrists_world=np.zeros((2, 4, 4)))
    path = HeightPath(0.)
    with pytest.raises(ValueError):
        path.sample("OUTSIDE", -.1)
