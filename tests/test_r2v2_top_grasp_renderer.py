"""CPU orchestration tests: rendering must not change physical experiment state."""

import json
import sys
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from deploy_mujoco import r2v2_top_grasp as entry


class _Experiment:
    def __init__(self, *, raise_on_step=False):
        self.model = SimpleNamespace(nq=3, nv=3, nu=3, nmocap=1)
        self.data = SimpleNamespace(time=4., qpos=np.array([.1, .2, .3]),
                                    qvel=np.array([.4, .5, .6]), ctrl=np.array([.7, .8, .9]))
        self.scratch = SimpleNamespace(qpos=self.data.qpos.copy())
        self.phase, self.failure, self.side = "APPROACH", None, "left"
        self.layout = SimpleNamespace(cylinder_initial_position=np.array([0., .1, .7]),
                                      cylinder_place_position=np.array([.1, .1, .7]))
        self.hands = SimpleNamespace(controllers={
            side: SimpleNamespace(command=0) for side in ("left", "right")})
        self.profile = {"profile_id": "baseline_40mm_100g", "radius_m": .02,
                        "height_m": .12, "mass_kg": .1}
        self.current_metrics = {}
        self.samples, self.transitions, self.targets = [], [], []
        self.step_calls, self.sync_calls = 0, 0
        self.raise_on_step = raise_on_step

    @property
    def done(self):
        return self.phase in ("COMPLETE", "FAILED")

    def sync(self):
        self.sync_calls += 1

    def step(self):
        self.step_calls += 1
        self.data.time += .02
        if self.raise_on_step:
            raise RuntimeError("intentional CPU test exception")
        if self.step_calls == 2:
            self.phase = "COMPLETE"

    def fail(self, reason):
        self.phase, self.failure = "FAILED", reason

    def report(self):
        return {"phase": self.phase, "failure": self.failure, "strict_success": False,
                "success": False, "grasp_verified": False, "release_commanded": False,
                "duration_s": self.data.time - 4.}


class _Renderer:
    def __init__(self, model, height, width):
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.closed = False

    def update_scene(self, data, *, camera, scene_option):
        pass

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
            MjvOption=lambda: SimpleNamespace(geomgroup=np.ones(6, dtype=int)))
        monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
        monkeypatch.setitem(sys.modules, "common.r2v2_top_grasp", SimpleNamespace(TopGraspExperiment=construct))
        monkeypatch.setitem(sys.modules, "common.r2v2_top_grasp_scene", SimpleNamespace(
            TopGraspCandidate=lambda **kwargs: kwargs))
        monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: writer)
        monkeypatch.setattr(imageio, "imwrite", lambda *args, **kwargs: None)
        monkeypatch.setenv("MUJOCO_GL", "cpu-test-no-context")
        output = tmp_path / "trial"
        return exp, writer, seen, output
    return configure


def _assert_physical_sentinels(exp):
    np.testing.assert_array_equal(exp.data.qpos, [.1, .2, .3])
    np.testing.assert_array_equal(exp.data.qvel, [.4, .5, .6])
    np.testing.assert_array_equal(exp.data.ctrl, [.7, .8, .9])
    assert all(controller.command == 0 for controller in exp.hands.controllers.values())


def test_terminal_freeze_is_not_extra_physics_or_grasp_success(cpu_runtime):
    exp, writer, seen, output = cpu_runtime()
    assert entry.main(["--output", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["renderer_runtime_completed"] is True
    assert report["strict_success"] is False
    assert report["full_body_policy_used"] is False
    assert "KINEMATIC WRIST FIXTURE" in report["renderer_scope"]
    assert seen["keep_trace"] is True
    assert exp.step_calls == 2
    assert exp.data.time == pytest.approx(4.04)
    held = round(entry.FPS * entry.TERMINAL_HOLD_SECONDS)
    assert report["terminal_freeze_frames"] == held == 60
    assert report["terminal_freeze_is_not_simulation"] is True
    assert report["video_frame_times_s"][-held:] == [exp.data.time] * held
    assert len(writer.frames) == report["video_frames"]
    assert all(frame is writer.frames[-1] for frame in writer.frames[-held:])
    assert writer.frames[0].shape == (800, 1280, 3)
    assert writer.closed
    assert {p.name for p in output.iterdir()} >= {
        "report.json", "trace.json", "transitions.json", "targets.json"}
    _assert_physical_sentinels(exp)


def test_exception_keeps_failure_records_at_last_actual_time(cpu_runtime):
    exp, writer, seen, output = cpu_runtime(raise_on_step=True)
    assert entry.main(["--output", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["renderer_runtime_completed"] is False
    assert report["phase"] == "FAILED"
    assert "intentional CPU test exception" in report["runtime_error"]
    assert exp.step_calls == 1
    assert exp.data.time == pytest.approx(4.02)
    assert report["video_frame_times_s"][-60:] == [exp.data.time] * 60
    assert writer.closed
    _assert_physical_sentinels(exp)


def test_normal_experiment_failure_is_not_grasp_success(cpu_runtime):
    exp, writer, seen, output = cpu_runtime()
    exp.fail("grasp attempt did not establish opposing contacts")
    assert entry.main(["--output", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["phase"] == "FAILED"
    assert report["strict_success"] is False
    assert exp.step_calls == 0
    assert report["video_frame_times_s"] == [4.] * report["video_frames"]


def test_nonempty_output_preserved_without_constructing_experiment(cpu_runtime):
    exp, writer, seen, output = cpu_runtime()
    output.mkdir()
    existing = output / "keep.txt"
    existing.write_text("existing user result")
    with pytest.raises(SystemExit) as error:
        entry.main(["--output", str(output)])
    assert error.value.code == 2
    assert existing.read_text() == "existing user result"
    assert list(output.iterdir()) == [existing]
    assert not seen
    assert exp.step_calls == exp.sync_calls == 0
    assert not writer.frames


def test_no_video_still_saves_measurements_and_does_not_render(cpu_runtime):
    exp, writer, seen, output = cpu_runtime()
    assert entry.main(["--output", str(output), "--no-video"]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["video_file"] is None
    assert report["video_frames"] == report["terminal_freeze_frames"] == 0
    assert report["camera_configuration"] == report["phase_screenshots"] == []
    assert exp.sync_calls == 0
    assert exp.step_calls == 2
    assert not writer.frames


def test_cli_native_units_and_explicit_overrides_json(cpu_runtime, tmp_path):
    exp, writer, seen, output = cpu_runtime()
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({"depth_m": .027, "yaw_deg": 10., "future_field": "data"}))
    assert entry.main(["--output", str(output), "--no-video", "--candidate-json", str(path),
                       "--profile", "sleek_330ml_approx_full", "--yaw-deg", "15",
                       "--lateral-mm", "-3", "--tilt-deg", "5", "--side", "left"]) == 0
    assert seen["candidate"] == {"depth_m": .027, "yaw_deg": 15., "future_field": "data",
                                  "lateral_m": -.003, "tilt_deg": 5., "side": "left"}
    assert seen["profile"] == "sleek_330ml_approx_full"


def test_candidate_json_type_error_saved_without_fallback(cpu_runtime, tmp_path):
    exp, writer, seen, output = cpu_runtime()
    path = tmp_path / "candidate.json"
    path.write_text("[]")
    assert entry.main(["--output", str(output), "--no-video", "--candidate-json", str(path)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert "must contain an object" in report["runtime_error"]
    assert not seen
    assert exp.step_calls == 0


def test_construction_failure_saved_without_fallback(cpu_runtime, monkeypatch):
    exp, writer, seen, output = cpu_runtime()

    def reject(**kwargs):
        raise ValueError("invalid candidate or physical geometry")

    monkeypatch.setitem(sys.modules, "common.r2v2_top_grasp", SimpleNamespace(TopGraspExperiment=reject))
    assert entry.main(["--output", str(output), "--no-video"]) == 1
    report = json.loads((output / "report.json").read_text())
    assert "invalid candidate" in report["runtime_error"]
    assert report["strict_success"] is None
    assert exp.step_calls == exp.sync_calls == 0


def test_missing_measurements_and_status_do_not_fabricate_success():
    assert entry._metric(None) == "N/A"
    assert entry._metric(float("nan")) == "NONFINITE"
    assert entry._metric(.01, ".1f", 1000.) == "10.0"
    assert entry._boolean(None) == "N/A"
    terminal = SimpleNamespace(done=True, failure=None)
    assert "does not mean grasp/place success" in entry._status(terminal)[0]
    assert "NOT additional hold evidence" in entry._status(terminal, terminal_freeze=True)[0]


def test_cameras_fixed_on_transfer_region_without_modifying_state(cpu_runtime):
    exp, writer, seen, output = cpu_runtime()
    cameras = entry._make_cameras(exp)
    records = entry._camera_record(cameras)
    assert len(records) == 2
    assert records[0]["elevation_deg"] == 0.
    assert records[0]["azimuth_deg"] == 90.
    assert records[1]["elevation_deg"] == -35.
    assert all(record["fixed_world_camera"] for record in records)
    np.testing.assert_allclose(cameras[0].lookat, [.05, .1, .78])
    exp.current_metrics["object_position_m"] = [5., 6., 7.]
    np.testing.assert_allclose(cameras[0].lookat, [.05, .1, .78])
    _assert_physical_sentinels(exp)


def test_clock_that_does_not_advance_fails_and_saves_diagnostics(cpu_runtime):
    exp, writer, seen, output = cpu_runtime()
    exp.step = lambda: None
    assert entry.main(["--output", str(output), "--no-video"]) == 1
    report = json.loads((output / "report.json").read_text())
    assert "did not advance" in report["runtime_error"]
    assert report["renderer_runtime_completed"] is False


def test_stdout_uses_actual_success_fields_not_completion(cpu_runtime, capsys):
    exp, writer, seen, output = cpu_runtime()
    assert entry.main(["--output", str(output), "--no-video"]) == 0
    stdout = capsys.readouterr().out
    # The fake experiment reaches COMPLETE but deliberately does not claim a
    # verified grasp, release or success; the renderer must not infer them.
    assert '"phase": "COMPLETE"' in stdout
    assert '"success": false' in stdout
    assert '"grasp_verified": false' in stdout
    assert '"release_commanded": false' in stdout
    assert '"grasp_passed"' not in stdout
    assert '"place_passed"' not in stdout
