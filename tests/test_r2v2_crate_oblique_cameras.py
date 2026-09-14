"""Camera options must not change the hand-fixture experiment configuration."""

from types import SimpleNamespace

import numpy as np

from deploy_mujoco.r2v2_crate_oblique import _camera_record, _make_cameras, _parser


def _scene():
    return SimpleNamespace(
        sync=lambda: None,
        scratch=SimpleNamespace(xpos=np.array([[.1, .2, .401]])),
        model=SimpleNamespace(body=lambda name: SimpleNamespace(id=0)),
        crate_params=SimpleNamespace(width=.26, height=.16,
                                     handle_opening_bottom=.085,
                                     handle_opening_top=.14),
    )


def test_top_view_is_opt_in():
    assert not _parser().parse_args(["--output", "unused"]).top_view
    assert _parser().parse_args(["--output", "unused", "--top-view"]).top_view


def test_default_camera_poses_are_preserved():
    overview, handle = _make_cameras(_scene())
    assert (overview.distance, overview.elevation, overview.azimuth) == (1.12, -26., 135.)
    assert (handle.distance, handle.elevation, handle.azimuth) == (.71, -15., 65.)
    np.testing.assert_allclose(overview.lookat, [.1, .255, .513])
    np.testing.assert_allclose(handle.lookat, [.1, .33, .5385])


def test_top_view_keeps_overview_and_records_distinct_view():
    scene = _scene()
    original = _make_cameras(scene)
    cameras = _make_cameras(scene, top_view=True)
    original_record = _camera_record(original)
    records = _camera_record(cameras, top_view=True)
    for key in ("distance_m", "elevation_deg", "azimuth_deg", "fixed_world_camera", "view"):
        assert records[0][key] == original_record[0][key]
    np.testing.assert_array_equal(cameras[0].lookat, original[0].lookat)
    assert records[1]["view"] == "top"
    assert records[1]["fixed_world_camera"] is True
    assert cameras[1].elevation == -90.
    assert cameras[1].distance == 1.35
    np.testing.assert_allclose(cameras[1].lookat, [.1, .2, .513])
    assert original_record[1]["view"] == "left_handle"
