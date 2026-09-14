"""Free-space, free-base diagnostic for reaching recorded crate READY poses.

There is no physical table or crate in this diagnostic. The wire outline is
render-only and cannot contact, support, or move the robot. A successful state
may seed a separately initialized scene, never teleport an ongoing simulation.
"""

import argparse
import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from common.r2v2_crate_motion_recording import load_crate_motion, pose_from_transform, transform_from_pose
from common.r2v2_hand_control import DualHandControl
from common.r2v2_reach_sim import (
    ReachCompatibilityExperiment, load_reach_config, require_parity, sha256,
)
from deploy_mujoco.r2v2_crate_reach import _add_target_markers
from deploy_mujoco.r2v2_tabletop_demo import _json_safe, _phase_filename, _write_json
from r2v2_description.model import SIDES, initialize_hands, urdf_hand_joints


PLANNED_ANCHOR_XYZ = (.38, 0., 1.01191896965)


class FreeSpaceCratePrepareExperiment(ReachCompatibilityExperiment):
    """Actual policy-controlled preparation; no old standby or legacy TCP gate."""

    def __init__(self, motion_path, reach_config, parity_report):
        self.motion_path = Path(motion_path).resolve()
        if self.motion_path.is_dir():
            self.motion_path /= "manifest.json"
        self.motion = load_crate_motion(self.motion_path)
        cfg = load_reach_config(reach_config)
        if cfg.get("endpoint_contract") != "wrist_world_v2":
            raise ValueError("Preparation requires the true wrist-world-v2 contract")
        self.parity = require_parity(parity_report, cfg)
        super().__init__(cfg)
        source_hand_cfg = copy.deepcopy(self.motion.manifest["hand_configuration"])
        for key in ("simulation_dt", "control_dt", "servo", "trajectory"):
            if source_hand_cfg[key] != self.hand_cfg[key]:
                raise ValueError(f"Source/model hand configuration mismatch: {key}")
        self.hand_cfg = source_hand_cfg
        physical_limits = self.model.jnt_range[self.body_map.joints]
        midpoint = np.mean(physical_limits, axis=1)
        soft_half_range = .45 * (physical_limits[:, 1]-physical_limits[:, 0])
        before = self.data.qpos[self.body_map.qpos].copy()
        after = np.clip(before, midpoint-soft_half_range, midpoint+soft_half_range)
        self.data.qpos[self.body_map.qpos] = after
        self.initialization_correction = dict(
            meaning="Native training reset clips default body q into its 90% soft limits at t=0 only",
            physical_joint_ranges_unchanged=True, soft_joint_range_factor=.9,
            changed_joints=[dict(name=self.model.joint(int(joint)).name,
                before_rad=float(a), after_rad=float(b))
                for joint, a, b in zip(self.body_map.joints, before, after) if abs(a-b) > 1e-10])
        # Initialization only, before any physics. The robot is subsequently
        # moved exclusively by its policy and independent finger controller.
        initialize_hands(self.model, self.data, self.hand_cfg)
        self.hands = DualHandControl(self.model, self.data, self.hand_cfg)
        self.policy.reset(self.data)
        self.world_anchor = np.eye(4)
        self.world_anchor[:3, 3] = PLANNED_ANCHOR_XYZ
        target_array = self.world_anchor @ self.motion.sample(0.)["T_anchor_wrist"]
        self.preparation_targets = dict(zip(SIDES, target_array))
        self.goal_wrist_transforms = {side: transform_from_pose(*self.policy.tcp_pose(self.data, side))
                                      for side in SIDES}
        self.hand_joints = [self.model.joint(name).id for name in urdf_hand_joints()]
        self.hand_mimics = []
        for name, element in urdf_hand_joints().items():
            mimic = element.find("mimic")
            if mimic is not None:
                self.hand_mimics.append((self.model.joint(name).qposadr[0],
                    self.model.joint(mimic.get("joint")).qposadr[0],
                    float(mimic.get("multiplier", "1")), float(mimic.get("offset", "0"))))
        self.peaks.update(hand_joint_violation_rad=0., hand_mimic_error_rad=0.)
        self.limit_failure_details = None
        self.contact_pairs = {}
        self.initial_contacts = self._contacts()
        self.sync()
        self.record()

    def safety(self):
        reason = super().safety()
        if reason:
            if reason == "Measured body joint exceeded physical limit tolerance":
                limits = self.model.jnt_range[self.body_map.joints]
                measured = self.data.qpos[self.body_map.qpos]
                violations = np.maximum(limits[:, 0]-measured, measured-limits[:, 1])
                index = int(np.argmax(violations))
                self.limit_failure_details = dict(
                    joint_name=self.model.joint(int(self.body_map.joints[index])).name,
                    measured_rad=float(measured[index]), physical_range_rad=limits[index].copy(),
                    violation_rad=float(violations[index]), tolerance_rad=self.cfg["gates"]["max_joint_violation_rad"])
            return reason
        d, m, g = self.data, self.model, self.cfg["gates"]
        tilt = float(np.rad2deg(np.arccos(np.clip(d.xmat[self.base].reshape(3, 3)[2, 2], -1, 1))))
        self.peaks["base_tilt_deg"] = max(self.peaks["base_tilt_deg"], tilt)
        if tilt > g["max_base_tilt_deg"] or d.xpos[self.base, 2] < g["min_base_height_m"]:
            return "Base tilt/height safety limit"
        if m.nmocap or np.any(m.eq_type == mujoco.mjtEq.mjEQ_WELD):
            return "Unexpected fixture or weld"
        if np.any(d.qfrc_applied) or np.any(d.xfrc_applied):
            return "Unexpected auxiliary force"
        limits = m.jnt_range[self.hand_joints]
        measured = d.qpos[m.jnt_qposadr[self.hand_joints]]
        violation = float(max(0., np.max(limits[:, 0]-measured), np.max(measured-limits[:, 1])))
        mimic = max(abs(d.qpos[a]-ratio*d.qpos[b]-offset) for a, b, ratio, offset in self.hand_mimics)
        self.peaks["hand_joint_violation_rad"] = max(self.peaks["hand_joint_violation_rad"], violation)
        self.peaks["hand_mimic_error_rad"] = max(self.peaks["hand_mimic_error_rad"], mimic)
        if violation > .01:
            return "Measured finger joint exceeded physical limit tolerance"
        if mimic > .015:
            return "Finger mimic constraint error"
        return None

    def update_gate(self):
        # Safety remains in the inherited step at every 1 kHz physics step.
        if self.phase == "RESET_SETTLE":
            if self.data.time < .4 - 1e-8:
                self.policy.follow_current(self.scratch)
            else:
                for side in SIDES:
                    p, q, _, _ = self.tcp(side)
                    self.policy.set_target_world(side, p, q)
                    self.goal_wrist_transforms[side] = transform_from_pose(p, q)
                self.transition("STAND_WORLD_LOCK")
            return
        if self.phase == "STAND_WORLD_LOCK":
            if self.data.time-self.phase_start >= 2.-1e-8:
                self.standing_passed = True
                for side, target in self.preparation_targets.items():
                    self.policy.set_target_world(side, *pose_from_transform(target))
                self.goal_wrist_transforms = self.preparation_targets
                self.transition("FREE_SPACE_PREALIGN")
            return
        errors = {side: self.errors(side) for side in SIDES}
        at_goal = all(e["wrist_position_m"] < .02 and e["orientation_deg"] < 10.
                      and e["wrist_linear_speed_mps"] < .02 for e in errors.values())
        if self.stable_for_required_time(at_goal):
            self.transition("PASSED", errors)
        elif self.data.time-self.phase_start >= 10.-1e-8:
            self.fail("Recorded READY wrist targets did not settle within 10 s (20 mm / 10 deg / 20 mm/s)")

    def report(self):
        result = super().report()
        result.update(scope="FREE-SPACE PREALIGN ONLY: no physical crate/table, no insertion or grasp",
            planned_crate_anchor_world=self.world_anchor.copy(),
            target_source_time_s=0., target_source="measured READY wrist poses, not fixture commands",
            target_transforms_world=self.preparation_targets,
            preparation_gate=dict(position_m=.02, orientation_deg=10., speed_mps=.02, stable_s=.3, timeout_s=10.),
            reference_motion_sha256=sha256(self.motion_path),
            preparation_runtime_sha256=sha256(Path(__file__)),
            prepared_state_available=self.phase == "PASSED", physical_table_or_crate=False,
            visual_crate_outline_is_not_physics=True,
            initialization_correction=self.initialization_correction,
            limit_failure_details=self.limit_failure_details,
            standing_passed_meaning="Short 2 s survival with locked wrist goals only; not the 20 s certification gate")
        return result

    def save_prepared_state(self, path):
        if self.phase != "PASSED":
            raise ValueError("Only a successfully reached state may initialize the next scene")
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        integration = np.empty(mujoco.mj_stateSize(self.model, spec))
        mujoco.mj_getState(self.model, self.data, integration, spec)
        payload = dict(integration_state=integration, integration_state_spec=np.array(int(spec)),
            simulation_time_s=np.array(self.data.time), steps=np.array(self.steps),
            qpos=self.data.qpos.copy(), qvel=self.data.qvel.copy(), ctrl=self.data.ctrl.copy(),
            act=self.data.act.copy(), qacc_warmstart=self.data.qacc_warmstart.copy(),
            joint_names=np.asarray([self.model.joint(j).name for j in range(self.model.njnt)]),
            joint_qposadr=self.model.jnt_qposadr.copy(), joint_dofadr=self.model.jnt_dofadr.copy(),
            actuator_names=np.asarray([self.model.actuator(a).name for a in range(self.model.nu)]),
            last_action=self.policy.last_action.copy(), q_des=self.policy.q_des.copy(),
            last_torque=self.policy.last_torque.copy(), last_observation=self.policy.last_observation.copy())
        for name, history in self.policy.histories.items():
            payload[f"history_{name}"] = history.copy()
        for side, reference in self.policy.references.items():
            for name, value in asdict(reference).items():
                payload[f"reference_{side}_{name}"] = np.asarray(value)
        np.savez_compressed(path, **payload)
        _write_json(path.with_suffix(".json"), dict(
            scope="Initialize a new scene only; never restore into an ongoing rollout",
            state_file=path.name, sha256=sha256(path), integration_spec=int(spec),
            simulation_time_s=float(self.data.time), endpoint_contract=self.policy.endpoint_contract,
            onnx_sha256=self.parity["onnx_sha256"], checkpoint_sha256=self.parity["checkpoint_sha256"],
            adapter_sha256=self.parity["adapter_sha256"],
            hand_configuration=self.hand_cfg, hand_commands={s: self.hands.controllers[s].command for s in SIDES},
            planned_crate_anchor_world=self.world_anchor, target_transforms_world=self.preparation_targets,
            final_errors={s: self.errors(s) for s in SIDES}))


def _cameras():
    cameras = []
    for distance, elevation, azimuth, target in ((3.05, 0., 90., (.13, 0., .89)),
            (1.65, -22., 135., (.20, 0., 1.15))):
        camera = mujoco.MjvCamera()
        camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
        camera.lookat[:] = target
        cameras.append(camera)
    return cameras


def _virtual_crate(scene, exp):
    params = exp.motion.manifest["crate_parameters"]
    x, y, z = params["depth"]*.5, params["width"]*.5, params["height"]
    vertices = np.array([(a, b, c) for a in (-x, x) for b in (-y, y) for c in (0., z)])
    vertices += exp.world_anchor[:3, 3]
    for i in range(8):
        for bit in (1, 2, 4):
            j = i ^ bit
            if j <= i or scene.ngeom >= scene.maxgeom:
                continue
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, np.zeros(3), np.zeros(3),
                              np.eye(3).ravel(), np.array([1., .65, .15, 1.], dtype=np.float32))
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2., vertices[i], vertices[j])
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            scene.ngeom += 1


def _frame(exp, renderer, cameras, options, font):
    from PIL import Image, ImageDraw
    exp.sync()
    panels = []
    for camera in cameras:
        renderer.update_scene(exp.scratch, camera=camera, scene_option=options)
        _virtual_crate(renderer.scene, exp)
        _add_target_markers(renderer.scene, exp)
        panels.append(renderer.render().copy())
    canvas = Image.new("RGB", (1280, 800), (20, 28, 36))
    canvas.paste(Image.fromarray(np.concatenate(panels, axis=1)), (0, 150))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 8), f"FREE-SPACE PREALIGN / TRUE FREE-BASE ROBOT | t={exp.data.time:05.2f}s | {exp.phase}",
              font=font, fill=(255, 207, 95))
    draw.text((14, 35), "VIRTUAL CRATE OUTLINE ONLY | NO PHYSICAL TABLE / CRATE / CONTACT SUPPORT", font=font, fill="white")
    for i, side in enumerate(SIDES):
        e = exp.errors(side)
        draw.text((14, 63+25*i), f"{side.upper()} WRIST: {e['wrist_position_m']*1000:.1f} mm / "
            f"{e['orientation_deg']:.1f} deg | actual speed {e['wrist_linear_speed_mps']:.3f} m/s",
            font=font, fill=(160, 210, 240))
    draw.text((650, 63), "SOURCE READY: down 20 deg + inward yaw 15 deg", font=font, fill="white")
    draw.text((650, 88), "Pass: both wrists <20 mm / 10 deg / 20 mm/s for 0.3s", font=font, fill="white")
    draw.text((14, 124), "EXACT SIDE VIEW / ALL JOINTS PHYSICAL", font=font, fill="white")
    draw.text((650, 124), "GREEN / BLUE = WORLD WRIST TARGETS", font=font, fill="white")
    draw.text((14, 725), "Policy 50 Hz | fingers 100 Hz | physics + safety 1 kHz | no IK / weld / pose replay / auxiliary forces",
              font=font, fill="white")
    status = "FAILED: " + exp.failure if exp.failure else ("PREALIGN PASSED (NOT A GRASP)" if exp.done else "IN PROGRESS")
    draw.text((14, 751), status[:140], font=font, fill=(255, 145, 120) if exp.failure else (155, 225, 175))
    draw.text((14, 776), "Diagnostic omits obstacles explicitly; successful prealignment alone does not prove collision-free crate manipulation.",
              font=font, fill=(190, 200, 210))
    return np.asarray(canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--reach-config", type=Path, required=True)
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Refusing to overwrite nonempty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    import imageio.v2 as imageio
    from PIL import ImageFont
    exp = renderer = writer = None
    errors, screenshots, frame_times = [], [], []
    runtime_error = None
    final = None
    frame_count = freeze_frames = 0
    report = dict(passed=False, phase="INITIALIZATION")
    try:
        exp = FreeSpaceCratePrepareExperiment(args.motion, args.reach_config, args.parity_report)
        renderer = mujoco.Renderer(exp.model, height=560, width=640)
        writer = imageio.get_writer(str(out / "free_space_prepare.mp4"), fps=30, codec="libx264", quality=8)
        cameras, options = _cameras(), mujoco.MjvOption()
        options.geomgroup[3] = 0
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        phase, next_frame = None, 0.
        while True:
            changed = phase != exp.phase
            due = exp.data.time+1e-9 >= next_frame or exp.done
            if changed:
                print(f"{exp.data.time:.3f}s {exp.phase}: {exp.failure or ''}", flush=True)
            if changed or due:
                array = _frame(exp, renderer, cameras, options, font)
                if changed:
                    filename = _phase_filename(len(screenshots), exp.phase)
                    imageio.imwrite(out / filename, array)
                    screenshots.append(dict(time_s=float(exp.data.time), phase=exp.phase, file=filename))
                    if phase is None:
                        imageio.imwrite(out / "initial.png", array)
                if due:
                    writer.append_data(array)
                    frame_count += 1
                    frame_times.append(float(exp.data.time))
                    next_frame += 1./30
                if exp.done:
                    final = array
                    imageio.imwrite(out / "final.png", final)
            phase = exp.phase
            if exp.done:
                break
            exp.step()
        if exp.phase == "PASSED":
            exp.save_prepared_state(out / "prepared_state.npz")
    except BaseException as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
        errors.append(traceback.format_exc())
        print(runtime_error, flush=True)
    finally:
        if writer is not None and final is not None:
            for _ in range(60):
                writer.append_data(final)
                frame_count += 1
                freeze_frames += 1
                frame_times.append(float(exp.data.time))
        for item in (writer, renderer):
            if item is not None:
                item.close()
        if exp is not None:
            report = exp.report()
        report.update(runtime_error=runtime_error, runtime_tracebacks=errors, video_file="free_space_prepare.mp4",
            video_frames=frame_count, video_fps=30, video_frame_times_s=frame_times,
            terminal_freeze_frames=freeze_frames, terminal_freeze_is_not_simulation=True,
            phase_screenshots=screenshots)
        _write_json(out / "report.json", report)
        _write_json(out / "trace.json", [] if exp is None else exp.samples)
        _write_json(out / "transitions.json", [] if exp is None else exp.transitions)
    print(json.dumps(_json_safe({key: report.get(key) for key in
        ("passed", "phase", "failure", "duration_s", "final_errors", "peaks", "runtime_error")}), indent=2), flush=True)
    return 0 if report.get("passed") and runtime_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
