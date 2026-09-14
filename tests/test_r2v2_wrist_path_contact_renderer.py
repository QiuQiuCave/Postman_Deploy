"""CPU-only renderer orchestration tests; no MuJoCo physics or GPU is run.

The fake experiment owns the only clock changes. Real HUD composition and the
entry point run against a memory-only renderer/writer, preserving physical-state
sentinels and making terminal image repeats directly inspectable.
"""

import json
import sys
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from deploy_mujoco import r2v2_wrist_path_contact as entry


class _Experiment:
    def __init__(self, *, raise_on_step=False):
        self.model = SimpleNamespace(nq=3, nv=3, nu=3, nmocap=0)
        self.data = SimpleNamespace(time=4., qpos=np.array([.1, .2, .3]),
                                    qvel=np.array([.4, .5, .6]), ctrl=np.array([.7, .8, .9]))
        self.scratch = SimpleNamespace(qpos=self.data.qpos.copy())
        self.phase, self.failure = "APPROACH", None
        self.table_height = .91
        self.hands = SimpleNamespace(controllers={
            side: SimpleNamespace(command=0) for side in ("left", "right")})
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
                "duration_s": self.data.time-4.}


class _Renderer:
    def __init__(self, model, height, width):
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.scene = SimpleNamespace()
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
        # Keep a reference: the final hold must repeat one unchanged image.
        self.frames.append(frame)

    def close(self):
        self.closed = True


@pytest.fixture
def cpu_runtime(monkeypatch, tmp_path):
    def configure(*, raise_on_step=False):
        exp, writer = _Experiment(raise_on_step=raise_on_step), _Writer()
        fake_mujoco = SimpleNamespace(
            __version__="cpu-test-stub", Renderer=_Renderer,
            MjvOption=lambda: SimpleNamespace(geomgroup=np.ones(6, dtype=int)))
        monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
        monkeypatch.setitem(sys.modules, "common.r2v2_wrist_path_contact", SimpleNamespace(
            WristPathContactExperiment=lambda **kwargs: exp))
        monkeypatch.setattr(entry, "_make_cameras", lambda exp: ("side", "closeup"))
        monkeypatch.setattr(entry, "_camera_record", lambda cameras: ["fixed side", "fixed closeup"])
        monkeypatch.setattr(entry, "_add_target_markers", lambda scene, exp: None)
        monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: writer)
        monkeypatch.setattr(imageio, "imwrite", lambda *args, **kwargs: None)
        monkeypatch.setenv("MUJOCO_GL", "cpu-test-no-context")
        output = tmp_path / "trial"
        monkeypatch.setattr(sys, "argv", ["r2v2_wrist_path_contact.py",
            "--path-manifest", str(tmp_path / "path.json"),
            "--reach-config", str(tmp_path / "reach.yaml"),
            "--parity-report", str(tmp_path / "parity.json"), "--output", str(output)])
        return exp, writer, output
    return configure


def _assert_physical_sentinels(exp):
    np.testing.assert_array_equal(exp.data.qpos, [.1, .2, .3])
    np.testing.assert_array_equal(exp.data.qvel, [.4, .5, .6])
    np.testing.assert_array_equal(exp.data.ctrl, [.7, .8, .9])
    assert all(controller.command == 0 for controller in exp.hands.controllers.values())


def test_terminal_hold_repeats_image_without_advancing_time_or_claiming_grasp(cpu_runtime):
    exp, writer, output = cpu_runtime()
    assert entry.main() == 0
    report = json.loads((output / "report.json").read_text())
    assert report["renderer_runtime_completed"] is True
    assert report["strict_success"] is False
    assert exp.step_calls == 2
    assert exp.data.time == pytest.approx(4.04)
    held = round(entry.FPS * entry.TERMINAL_HOLD_SECONDS)
    assert report["terminal_freeze_frames"] == held == 60
    assert report["terminal_freeze_is_not_simulation"] is True
    assert report["video_frame_times_s"][-held:] == [exp.data.time] * held
    assert len(writer.frames) == report["video_frames"]
    assert all(frame is writer.frames[-1] for frame in writer.frames[-held:])
    assert writer.closed
    assert {p.name for p in output.iterdir()} >= {
        "report.json", "trace.json", "transitions.json", "targets.json"}
    _assert_physical_sentinels(exp)


def test_runtime_exception_preserves_failure_and_freezes_last_actual_time(cpu_runtime):
    exp, writer, output = cpu_runtime(raise_on_step=True)
    assert entry.main() == 1
    report = json.loads((output / "report.json").read_text())
    assert report["renderer_runtime_completed"] is False
    assert report["phase"] == "FAILED"
    assert "intentional CPU test exception" in report["runtime_error"]
    assert exp.step_calls == 1
    assert exp.data.time == pytest.approx(4.02)
    assert report["video_frame_times_s"][-60:] == [exp.data.time] * 60
    assert writer.closed
    _assert_physical_sentinels(exp)


def test_normal_physical_stop_is_not_reported_as_grasp_success(cpu_runtime):
    exp, writer, output = cpu_runtime()
    exp.fail("simulated terminal hard-stop report")
    assert entry.main() == 0  # Normal artifact handling, not a successful grasp.
    report = json.loads((output / "report.json").read_text())
    assert report["renderer_runtime_completed"] is True
    assert report["phase"] == "FAILED"
    assert report["failure"] == "simulated terminal hard-stop report"
    assert report["strict_success"] is False
    assert exp.step_calls == 0
    assert report["video_frame_times_s"] == [4.] * report["video_frames"]
    _assert_physical_sentinels(exp)


def test_nonempty_output_is_rejected_without_touching_files_or_experiment(cpu_runtime):
    exp, writer, output = cpu_runtime()
    output.mkdir()
    existing = output / "keep.txt"
    existing.write_text("user-owned existing result")
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
    assert existing.read_text() == "user-owned existing result"
    assert sorted(p.name for p in output.iterdir()) == ["keep.txt"]
    assert exp.step_calls == exp.sync_calls == 0
    assert not writer.frames


def test_status_and_missing_measurements_do_not_fabricate_success_or_zero_error():
    assert entry._metric(None) == "N/A"
    assert entry._metric(float("nan")) == "NONFINITE"
    assert entry._metric(.01, ".1f", 1000.) == "10.0"
    terminal = SimpleNamespace(done=True, failure=None)
    assert "not a grasp-success claim" in entry._status(terminal)[0]
    assert "not extra stability evidence" in entry._status(terminal, terminal_freeze=True)[0]


def test_grasp_plan_explicitly_selects_fsm_and_distinct_video(cpu_runtime, monkeypatch, tmp_path):
    exp, writer, output = cpu_runtime()
    seen = {}

    def construct(**kwargs):
        seen.update(kwargs)
        return exp

    monkeypatch.setitem(sys.modules, "common.r2v2_wrist_path_grasp_fsm", SimpleNamespace(
        WristPathGraspFSMExperiment=construct))
    plan = tmp_path / "geometry_checked_grasp.json"
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--grasp-plan", str(plan)])
    assert entry.main() == 0
    report = json.loads((output / "report.json").read_text())
    assert seen["grasp_plan"] == plan
    assert report["grasp_plan_path"] == str(plan.resolve())
    assert report["video_file"] == "wrist_path_grasp_fsm.mp4"
    assert report["renderer_scope"].startswith("CONTACT-GATED GRASP")
    assert report["strict_success"] is False
    assert "require verified physical pickup" in entry._status(
        SimpleNamespace(done=False, failure=None), grasp_mode=True)[0]
    _assert_physical_sentinels(exp)


def test_grasp_plan_construction_failure_is_saved_without_fallback_to_timed(cpu_runtime, monkeypatch, tmp_path):
    exp, writer, output = cpu_runtime()

    def reject(**kwargs):
        raise ValueError("Geometry plan/model binding mismatch")

    monkeypatch.setitem(sys.modules, "common.r2v2_wrist_path_grasp_fsm", SimpleNamespace(
        WristPathGraspFSMExperiment=reject))
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--grasp-plan", str(tmp_path / "bad.json"), "--no-video"])
    assert entry.main() == 1
    report = json.loads((output / "report.json").read_text())
    assert "binding mismatch" in report["runtime_error"]
    assert report["renderer_runtime_completed"] is False
    assert report["video_file"] is None
    assert exp.step_calls == exp.sync_calls == 0
    assert not writer.frames
