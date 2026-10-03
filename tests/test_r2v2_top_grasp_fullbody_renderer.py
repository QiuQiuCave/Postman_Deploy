"""CPU recorder tests: fixed cameras, honest modes, no experiment mutation."""

import hashlib
import json
import sys
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from deploy_mujoco import r2v2_top_grasp_fullbody as entry


class _Experiment:
    def __init__(self, *, raise_on_step=False):
        self.model = SimpleNamespace(nq=3, nv=3, nu=3, nmocap=0)
        self.data = SimpleNamespace(time=4., qpos=np.array([.1, .2, .3]),
                                    qvel=np.array([.4, .5, .6]), ctrl=np.array([.7, .8, .9]))
        self.scratch = SimpleNamespace(qpos=self.data.qpos.copy())
        self.phase, self.failure, self.side = "APPROACH", None, "left"
        self.layout = SimpleNamespace(cylinder_initial_position=np.array([.3, .1, .7]),
                                      cylinder_place_position=np.array([.4, .1, .7]))
        self.hands = SimpleNamespace(controllers={
            side: SimpleNamespace(command=0) for side in ("left", "right")})
        self.current_metrics = {}
        self.samples = [{"time_s": 4., "phase": "APPROACH", "sentinel": 123}]
        self.transitions, self.targets = [{"time_s": 4.}], [{"goal": [1., 2., 3.]}]
        self.render_wireframes = []
        self.step_calls, self.sync_calls = 0, 0
        self.raise_on_step = raise_on_step

    @property
    def done(self):
        return self.phase in ("COMPLETE", "FAILED")

    def sync(self):
        self.sync_calls += 1

    def step(self):
        self.step_calls += 1
        self.data.time += .01
        self.samples.append({"time_s": self.data.time, "phase": self.phase, "sentinel": 123})
        if self.raise_on_step:
            raise RuntimeError("intentional CPU test exception")
        if self.step_calls == 2:
            self.phase = "COMPLETE"

    def fail(self, reason):
        raise AssertionError("Recorder must never mutate experiment failure state")

    def report(self):
        return {"phase": self.phase, "failure": self.failure, "success": False,
                "air_passed": False, "grasp_verified": False, "release_commanded": False,
                "duration_s": self.data.time - 4., "config_sha256": "normalized-experiment-config-hash"}


class _Renderer:
    instances = []

    def __init__(self, model, height, width):
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.scene = SimpleNamespace(ngeom=0, maxgeom=1024,
                                     geoms=[SimpleNamespace() for _ in range(1024)])
        self.closed, self.options_seen = False, []
        self.instances.append(self)

    def update_scene(self, data, *, camera, scene_option):
        self.scene.ngeom = 0
        self.options_seen.append(scene_option.geomgroup.copy())

    def render(self):
        return self.frame

    def close(self):
        self.closed = True


class _Writer:
    def __init__(self):
        self.frames, self.closed = [], False

    def append_data(self, frame):
        self.frames.append(frame)

    def close(self):
        self.closed = True


@pytest.fixture
def cpu_runtime(monkeypatch, tmp_path):
    def configure(*, raise_on_step=False):
        exp, writer, seen = _Experiment(raise_on_step=raise_on_step), _Writer(), {}

        def construct(**kwargs):
            seen.update(kwargs)
            return exp

        fake_mujoco = SimpleNamespace(
            __version__="cpu-test-stub", Renderer=_Renderer,
            MjvCamera=lambda: SimpleNamespace(lookat=np.zeros(3)),
            MjvOption=lambda: SimpleNamespace(geomgroup=np.ones(6, dtype=int)),
            mjtGeom=SimpleNamespace(mjGEOM_BOX=6, mjGEOM_CYLINDER=5, mjGEOM_LINE=100),
            mjtCatBit=SimpleNamespace(mjCAT_DECOR=4),
            mjv_initGeom=lambda *args: None,
            mjv_connector=lambda *args: None)
        monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
        monkeypatch.setitem(sys.modules, "common.r2v2_top_grasp_fullbody",
                            SimpleNamespace(TopGraspFullbodyExperiment=construct))
        monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: writer)
        monkeypatch.setattr(imageio, "imwrite", lambda *args, **kwargs: None)
        monkeypatch.setenv("MUJOCO_GL", "cpu-test-no-context")
        _Renderer.instances.clear()
        config = tmp_path / "config.json"
        config.write_text(json.dumps({"policy_path": "frozen.onnx", "scene": {"x": .3}}))
        return exp, writer, seen, tmp_path / "trial", config
    return configure


def _args(output, config, *rest):
    return ["--config", str(config), "--output", str(output), *rest]


def _assert_untouched(exp):
    np.testing.assert_array_equal(exp.data.qpos, [.1, .2, .3])
    np.testing.assert_array_equal(exp.data.qvel, [.4, .5, .6])
    np.testing.assert_array_equal(exp.data.ctrl, [.7, .8, .9])
    assert all(controller.command == 0 for controller in exp.hands.controllers.values())


def test_air_terminal_capture_is_not_grasp_success(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    assert entry.main(_args(output, config)) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["renderer_runtime_completed"] is True
    assert report["success"] is report["air_passed"] is report["grasp_verified"] is False
    assert report["renderer_scope"] == "AIR: NO GRASP EVIDENCE"
    assert seen == {"config": json.loads(config.read_text()), "mode": "air", "keep_trace": True}
    assert report["requested_config_file_sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert report["config_sha256"] == "normalized-experiment-config-hash"
    assert exp.step_calls == 2
    assert exp.data.time == pytest.approx(4.02)
    assert report["terminal_freeze_frames"] == 60
    assert report["terminal_freeze_is_not_simulation"] is True
    assert report["video_frame_times_s"][-60:] == [exp.data.time] * 60
    assert len(writer.frames) == report["video_frames"]
    assert all(frame is writer.frames[-1] for frame in writer.frames[-60:])
    assert writer.frames[0].shape == (800, 1280, 3)
    assert writer.closed and _Renderer.instances[0].closed
    assert all(x[3] == x[4] == 0 for x in _Renderer.instances[0].options_seen)
    assert all(x[1] == x[2] == 1 for x in _Renderer.instances[0].options_seen)
    _assert_untouched(exp)


def test_contact_mode_is_explicit_and_visual_props_retained(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    assert entry.main(_args(output, config, "--mode", "contact")) == 0
    report = json.loads((output / "report.json").read_text())
    assert seen["mode"] == "contact"
    assert report["renderer_scope"] == "CONTACT: FREE ROBOT AND OBJECT"
    assert report["success"] is False
    assert all(x[3] == 0 and x[4] == 1 for x in _Renderer.instances[0].options_seen)
    _assert_untouched(exp)


def test_exception_keeps_controller_state_and_failure_records(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime(raise_on_step=True)
    assert entry.main(_args(output, config)) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["renderer_runtime_completed"] is False
    assert report["phase"] == "APPROACH"
    assert report["failure"] is None
    assert "intentional CPU test exception" in report["runtime_error"]
    assert exp.step_calls == 1
    assert report["video_frame_times_s"][-60:] == [exp.data.time] * 60
    assert writer.closed
    _assert_untouched(exp)


def test_experiment_failure_is_successful_capture_not_acceptance(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    exp.phase, exp.failure = "FAILED", "tracking did not settle"
    assert entry.main(_args(output, config)) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["phase"] == "FAILED"
    assert report["success"] is False
    assert exp.step_calls == 0


def test_nonempty_output_is_preserved_before_initialization(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    output.mkdir()
    existing = output / "keep.txt"
    existing.write_text("existing user result")
    with pytest.raises(SystemExit) as error:
        entry.main(_args(output, config))
    assert error.value.code == 2
    assert existing.read_text() == "existing user result"
    assert list(output.iterdir()) == [existing]
    assert not seen and exp.step_calls == exp.sync_calls == 0


def test_no_video_saves_exact_raw_samples_without_interpolation(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    assert entry.main(_args(output, config, "--no-video")) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["video_file"] is None
    assert report["video_frames"] == report["terminal_freeze_frames"] == 0
    assert report["camera_configuration"] == report["phase_screenshots"] == []
    assert exp.sync_calls == 0 and exp.step_calls == 2
    assert not writer.frames
    assert json.loads((output / "trace.json").read_text()) == exp.samples
    assert json.loads((output / "targets.json").read_text()) == exp.targets
    assert json.loads((output / "transitions.json").read_text()) == exp.transitions
    np.testing.assert_allclose(np.diff([s["time_s"] for s in exp.samples]), .01)


@pytest.mark.parametrize("contents", ["[]", "null", "not json"])
def test_bad_config_saved_without_fallback(cpu_runtime, contents):
    exp, writer, seen, output, config = cpu_runtime()
    config.write_text(contents)
    assert entry.main(_args(output, config, "--no-video")) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["runtime_error"]
    assert report["success"] is None
    assert not seen and exp.step_calls == 0


def test_construction_failure_preserves_reason_no_physics(cpu_runtime, monkeypatch):
    exp, writer, seen, output, config = cpu_runtime()

    def reject(**kwargs):
        raise ValueError("contact preflight requires passed AIR report")

    monkeypatch.setitem(sys.modules, "common.r2v2_top_grasp_fullbody",
                        SimpleNamespace(TopGraspFullbodyExperiment=reject))
    assert entry.main(_args(output, config, "--mode", "contact", "--no-video")) == 1
    report = json.loads((output / "report.json").read_text())
    assert "requires passed AIR" in report["runtime_error"]
    assert report["success"] is None
    assert exp.step_calls == exp.sync_calls == 0


def test_missing_metrics_are_not_displayed_as_zero():
    assert entry._metric(None) == "N/A"
    assert entry._metric(float("nan")) == "NONFINITE"
    assert entry._boolean(None) == "N/A"
    terminal = SimpleNamespace(done=True, failure=None)
    assert "does not imply acceptance" in entry._status(terminal, "air")[0]
    assert "NOT extra physics" in entry._status(terminal, "air", terminal_freeze=True)[0]


def test_cameras_fixed_in_world_without_modifying_state(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    cameras = entry._make_cameras(exp)
    records = entry._camera_record(cameras)
    assert records[0]["elevation_deg"] == 0.
    assert records[0]["azimuth_deg"] == 90.
    assert records[0]["view"] == "exact_side_full_body"
    assert all(record["fixed_world_camera"] for record in records)
    np.testing.assert_allclose(cameras[1].lookat, [.35, .1, .82])
    exp.current_metrics["object_position_m"] = [5., 6., 7.]
    np.testing.assert_allclose(cameras[1].lookat, [.35, .1, .82])
    _assert_untouched(exp)


def test_nonadvancing_clock_fails_and_reports(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    exp.step = lambda: None
    assert entry.main(_args(output, config, "--no-video")) == 1
    report = json.loads((output / "report.json").read_text())
    assert "did not advance" in report["runtime_error"]
    assert report["renderer_runtime_completed"] is False


def test_stdout_uses_actual_success_not_complete(cpu_runtime, capsys):
    exp, writer, seen, output, config = cpu_runtime()
    assert entry.main(_args(output, config, "--no-video")) == 0
    stdout = capsys.readouterr().out
    assert '"phase": "COMPLETE"' in stdout
    assert '"success": false' in stdout
    assert '"air_passed": false' in stdout
    assert '"grasp_verified": false' in stdout


def test_air_wireframe_edges_respect_orientation_and_size(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    exp.render_wireframes = [
        {"kind": "box", "position": [1., 2., 3.], "size": [.1, .2, .3]},
        {"kind": "cylinder", "position": [0., 0., .7], "size": [.02, .06]},
    ]
    edges = entry._virtual_prop_edges(exp, "air")
    assert len(edges) == 12 + 48 + 8
    vertices = np.array([p for edge in edges[:12] for p in edge[:2]])
    np.testing.assert_allclose(vertices.min(axis=0), [.9, 1.8, 2.7])
    np.testing.assert_allclose(vertices.max(axis=0), [1.1, 2.2, 3.3])
    cylinder_vertices = np.array([p for edge in edges[12:] for p in edge[:2]])
    np.testing.assert_allclose(cylinder_vertices.min(axis=0), [-.02, -.02, .64])
    np.testing.assert_allclose(cylinder_vertices.max(axis=0), [.02, .02, .76])
    assert entry._virtual_prop_edges(exp, "contact") == []
    _assert_untouched(exp)


def test_invalid_wireframe_data_is_not_silently_replaced(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    exp.render_wireframes = [{"kind": "cylinder", "position": [0., 0., float("nan")], "size": [.02, .06]}]
    with pytest.raises(ValueError, match="finite"):
        entry._virtual_prop_edges(exp, "air")


def test_line_markers_only_modify_render_scene(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    exp.render_wireframes = [{"kind": "box", "position": [0., 0., .5], "size": [.1, .2, .02]}]
    exp.goal_wrist_transforms = {"left": np.eye(4), "right": np.eye(4)}
    exp.current_metrics["wrist_errors"] = {side: {"T_world_wrist": np.eye(4)} for side in ("left", "right")}
    scene = SimpleNamespace(ngeom=0, maxgeom=100, geoms=[SimpleNamespace() for _ in range(100)])
    entry._add_render_markers(scene, exp, "air")
    assert scene.ngeom == 12 + 6 + 6
    _assert_untouched(exp)


def test_line_capacity_failure_is_explicit(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    scene = SimpleNamespace(ngeom=1, maxgeom=1)
    with pytest.raises(RuntimeError, match="capacity"):
        entry._line(scene, [0., 0., 0.], [1., 0., 0.], [1., 1., 1., 1.])


def test_air_outlines_prefer_static_layout_over_moving_object_metrics(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    del exp.render_wireframes
    exp.layout.table_center = [0., 0., .68]
    exp.layout.table_half_size = [.27, .30, .02]
    exp.profile = {"radius_m": .02, "height_m": .12}
    exp.current_metrics["object_position_m"] = [10., 20., 30.]
    edges = entry._virtual_prop_edges(exp, "air")
    cylinder_vertices = np.array([p for edge in edges[12:] for p in edge[:2]])
    np.testing.assert_allclose(cylinder_vertices.min(axis=0), [.28, .08, .64])
    np.testing.assert_allclose(cylinder_vertices.max(axis=0), [.32, .12, .76])


def test_air_outlines_model_fallback_accepts_real_tabletop_geom(cpu_runtime):
    exp, writer, seen, output, config = cpu_runtime()
    del exp.render_wireframes
    exp.model.geom = lambda name: SimpleNamespace(id={"tabletop_geom": 0, "cylinder_geom": 1}[name])
    exp.model.geom_type = [6, 5]
    exp.model.geom_size = [[.27, .30, .02], [.02, .06, 0.]]
    exp.scratch.geom_xpos = np.array([[0., 0., .68], [.3, .1, .7]])
    exp.scratch.geom_xmat = np.array([np.eye(3).ravel(), np.eye(3).ravel()])
    edges = entry._virtual_prop_edges(exp, "air")
    assert len(edges) == 12 + 48 + 8
