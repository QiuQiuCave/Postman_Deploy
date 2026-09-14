"""Matched-height, empty-hand trajectory screening with a frozen body policy.

Tracking errors never become grasp permission. All fingers remain open. Nominal
motion segments are followed by equal settle windows, while collisions, limits
and unstable physics stop the trial. No successful object lift is claimed.
"""
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np

from common.path_config import PROJECT_ROOT
from common.r2v2_crate_height_path import (
    BASE_TABLE_TOP_M, DEFAULT_MOTION_PATH, HEIGHT_OFFSETS_M, HeightPath, inspect_approach_hand_sweep,
)
from common.r2v2_crate_motion_replay import CrateMotionReplayExperiment
from common.r2v2_crate_motion_recording import transform_from_pose
from common.r2v2_grasp_recording import body_transform
from common.r2v2_reach_sim import sha256
from r2v2_description.model import SIDES


class HeightSweepExperiment(CrateMotionReplayExperiment):
    def __init__(self, reach_config, parity_report, delta_z, prepared_state=None, *, delta_x=0., virtual_props=False):
        if not any(abs(float(delta_z)-x) < 1e-8 for x in HEIGHT_OFFSETS_M):
            raise ValueError("Use exactly one of the five preregistered height offsets")
        if prepared_state is None:
            raise ValueError("Height comparison requires the SAME policy-prepared, collision-free initial robot state")
        self.delta_z = float(delta_z)
        if not np.isfinite(delta_x) or not -.15 <= delta_x <= .15:
            raise ValueError("Horizontal offset must be finite and in [-0.15, +0.15] m")
        if not isinstance(virtual_props, bool):
            raise TypeError("virtual_props must be an explicit boolean")
        self.delta_x, self.virtual_props = float(delta_x), virtual_props
        self.prepared_state_path = str(Path(prepared_state).resolve())
        self.path = None
        self.segment_index = -1
        self.segment_results = []
        self.foot_anchor = None
        self.contact_stop_details = []
        self.limit_stop_details = None
        self.insertion_reference_executed = False
        super().__init__(DEFAULT_MOTION_PATH, reach_config, parity_report,
                         table_top_m=BASE_TABLE_TOP_M+self.delta_z, prepared_state=prepared_state,
                         crate_center_xy=(.38+self.delta_x, 0.), table_center_xy=(.48+self.delta_x, 0.),
                         virtual_props=self.virtual_props)
        self.initial_geometry_contacts = self._prop_contacts()
        # This is a NEW scene at t=0 with the prepared robot's continuous body,
        # reference and finger state. No prepare motion occurred among props.
        if self.done:
            return
        self.phase, self.phase_start = "START_HOLD", 0.
        self.transitions = [dict(time_s=0., state=self.phase, source_time_s=0.)]
        self.samples, self.targets = [], []
        self.foot_anchor = np.array([self.scratch.xpos[self.model.body(f"{s}_ankle_roll_link").id] for s in SIDES])
        self.record()

    def _prop_contacts(self):
        result = []
        for contact in self.scratch.contact:
            pair = set(map(int, contact.geom))
            if contact.dist <= 0 and pair & self.robot_geoms and pair & (self.crate_geoms | self.table_geoms):
                result.append(dict(geoms=[self.model.geom(int(g)).name for g in contact.geom],
                                   distance_m=float(contact.dist), time_s=float(self.data.time)))
        return result

    def _safety(self):
        super()._safety()
        q = self.data.qpos[self.body_map.qpos]
        limits = self.model.jnt_range[self.body_map.joints]
        excess = np.maximum(limits[:, 0]-q, q-limits[:, 1])
        if np.max(excess) > self.cfg["gates"]["max_joint_violation_rad"]:
            index = int(np.argmax(excess))
            self.limit_stop_details = dict(joint=self.model.joint(int(self.body_map.joints[index])).name,
                actual_rad=float(q[index]), hard_range_rad=limits[index].copy(), excess_rad=float(excess[index]))
        prop_contacts = self._prop_contacts()
        if prop_contacts:
            self.contact_stop_details = prop_contacts
        # Before actual insertion there is no permitted robot/prop contact.
        # Hand contacts inside a hole are logged but retain the same penetration
        # and crate-stability stops; there is never a close command in this probe.
        if not self.done and not self.virtual_props and prop_contacts and self.phase in (
                "START_HOLD", "OUTSIDE", "TURN_WRISTS", "PREALIGN", "READY"):
            self.fail("Premature hand/robot contact with table or crate during outside approach")

    def _stage_summary(self):
        if self.segment_index < 0 or len(self.segment_results) > self.segment_index:
            return
        rows = [r for r in self.samples if r["phase"] == self.phase]
        if not rows:
            return
        times = np.array([r["time_s"] for r in rows])
        elapsed = float(self.data.time-self.phase_start)
        nominal = self.path.segments[self.segment_index]["duration_s"]
        # A partial, failed segment cannot have a fictitious settled tail.
        settling_rows = [r for r in rows if r["time_s"] >= self.phase_start+nominal-1e-8]
        selected = settling_rows if settling_rows else rows[-min(100, len(rows)):]
        errors = {}
        for side in SIDES:
            es = [r["metrics"]["wrist_errors"][side] for r in selected]
            errors[side] = {f"mean_{key}": float(np.mean([e[key] for e in es]))
                for key in ("position_m", "orientation_deg", "linear_speed_mps")}
            positions = np.array([r["metrics"]["hands"][side]["T_world_wrist"] for r in selected])[:, :3, 3]
            errors[side]["jitter_rms_m"] = float(np.sqrt(np.mean(np.sum((positions-positions.mean(0))**2, axis=1))))
        completed = elapsed >= nominal+2.-1e-8
        good = np.array([all(r["metrics"]["wrist_errors"][s]["position_m"] < .02
            and r["metrics"]["wrist_errors"][s]["orientation_deg"] < 10.
            and r["metrics"]["wrist_errors"][s]["linear_speed_mps"] < .02 for s in SIDES)
            for r in settling_rows])
        from common.r2v2_simple_dual_reach import sustained_duration
        hold = sustained_duration([r["time_s"] for r in settling_rows], good)
        self.segment_results.append(dict(name=self.phase, duration_s=elapsed, completed=completed,
            source_motion_duration_s=nominal, settled_tail_available=bool(settling_rows), wrist_errors=errors,
            coarse_reach_passed=bool(completed and hold >= .3-1e-8), coarse_hold_s=hold,
            max_base_tilt_deg=max(r["metrics"]["base_tilt_deg"] for r in rows),
            max_foot_origin_drift_m=max(r.get("foot_origin_drift_m", 0.) for r in rows)))

    def enter(self, phase):
        # Override the grasp replay's CLOSED event: this is a path screen.
        if phase == "FAILED" and hasattr(self, "samples"):
            self._stage_summary()
        self.phase, self.phase_start = phase, float(self.data.time)
        self.transitions.append(dict(time_s=self.phase_start, state=phase, source_time_s=self.source_time_s))
        self.stable_since = self.bad_since = None

    def _next_segment(self):
        self._stage_summary()
        self.segment_index += 1
        if self.segment_index >= len(self.path.segments):
            self.enter("COMPLETE")
            return
        self.enter(self.path.segments[self.segment_index]["name"])
        if self.phase == "INSERT":
            self.insertion_reference_executed = True

    def _gate(self):
        if self.phase == "START_HOLD":
            if self.data.time >= 2.-1e-8:
                # The controller has a nonzero steady tracking bias. Starting
                # the new *goal* path at the measured wrist would jump its goal
                # inward by that bias (up to 5cm on the right in this snapshot).
                # Continue the already-settled smooth reference instead. Actual
                # robot/prop clearance is checked independently by physics.
                reference_start = {s: transform_from_pose(r.position, r.quaternion)
                                   for s, r in self.policy.references.items()}
                self.actual_wrist_at_path_start = {s: body_transform(self.scratch,
                    self.model.body(f"{s}_hand_roll_link").id) for s in SIDES}
                self.path = HeightPath(self.delta_z, reference_start,
                    world_crate_pose=body_transform(self.scratch, self.model.body("cargo_crate").id),
                    delta_x=getattr(self, "delta_x", 0.))
                self.approach_geometry = inspect_approach_hand_sweep(self.path)
                if not self.approach_geometry["passed"] and not getattr(self, "virtual_props", False):
                    self.fail("Planned open-hand approach lacks conservative crate/table clearance")
                    return
                self.world_anchor = self.path.world_anchor
                self._next_segment()
        elif self.path is not None:
            segment = self.path.segments[self.segment_index]
            if self.data.time-self.phase_start >= segment["duration_s"]+2.-1e-8:
                self._next_segment()

    def _update_source_clock(self):
        pass  # Segment source time is owned by the frozen HeightPath sampler.

    def _stream_targets(self):
        if self.path is None or self.done:
            return
        segment = self.path.segments[self.segment_index]
        sample = self.path.sample(segment["name"], max(0., float(self.data.time-self.phase_start)))
        self.source_time_s = float(sample["source_time_s"] or 0.)
        self.desired_crate_world = sample["T_world_crate"]
        for side, target in zip(SIDES, sample["T_world_wrist"]):
            self._set_goal(side, target)
        self.targets.append(dict(time_s=float(self.data.time), phase=self.phase, source_time_s=sample["source_time_s"],
            T_world_wrist_goal=sample["T_world_wrist"], T_world_crate_desired=self.desired_crate_world,
            recorded_hand_command=sample["hand_command"], executed_hand_command=[0, 0]))

    def step(self):
        if self.done:
            return
        if self.steps % 10 == 0:
            self.sync(); self._safety()
            if not self.done:
                self._gate()
            if self.done:
                self.record()
                return
            self.hands.update()
        if self.steps % 20 == 0:
            self.sync(); self._stream_targets(); self.policy.act(self.scratch)
        self.policy.apply(self.data); self.hands.apply(self.data)
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        # Critical contacts/limits are checked every physics substep. Refresh
        # contact/kinematic scratch only if an event or a 100Hz tick is due.
        q = self.data.qpos[self.body_map.qpos]
        limits = self.model.jnt_range[self.body_map.joints]
        critical = (not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel))
            or np.any(self.data.warning.number) or np.max(np.maximum(limits[:, 0]-q, q-limits[:, 1])) > .01)
        for c in self.data.contact:
            pair = set(map(int, c.geom))
            if c.dist <= 0 and ((pair & self.robot_geoms and pair & self.table_geoms)
                    or (self.floor in pair and pair & (self.robot_geoms-self.foot_geoms))
                    or (pair <= self.robot_geoms and c.dist < -.002)
                    or (pair & self.crate_geoms and pair & self.robot_geoms)):
                critical = True
        if critical or self.steps % 10 == 0:
            self.sync(); self._safety()
        if self.steps % 10 == 0 or self.done:
            self.record()

    def record(self):
        super().record()
        if self.foot_anchor is None:
            drift = 0.
        else:
            feet = np.array([self.scratch.xpos[self.model.body(f"{s}_ankle_roll_link").id] for s in SIDES])
            drift = float(np.max(np.linalg.norm(feet-self.foot_anchor, axis=1)))
        self.samples[-1]["foot_origin_drift_m"] = drift
        q, limits = self.data.qpos[self.body_map.qpos], self.model.jnt_range[self.body_map.joints]
        self.samples[-1]["min_body_joint_margin_rad"] = float(np.min(np.minimum(q-limits[:, 0], limits[:, 1]-q)))

    def report(self):
        result = super().report()
        result.update(scope="EMPTY-HAND PATH SCREEN: real table/crate collisions, no grasp or loaded lift",
            delta_z_m=self.delta_z, delta_x_m=self.delta_x, virtual_props=self.virtual_props,
            prop_collisions_enabled=not self.virtual_props, table_top_m=self.table_height,
            prepared_state_path=self.prepared_state_path,
            prepared_state_sha256=self.prepared_state_info["sha256"],
            prepared_initialization_scope="Identical policy-generated robot state in five NEW scenes; empty preparation is not obstacle avoidance",
            initial_geometry_contacts=self.initial_geometry_contacts,
            actual_wrist_at_path_start=getattr(self, "actual_wrist_at_path_start", None),
            path_start_convention="Continue current smooth reference; never jump goal back to biased measured wrist",
            phase_results=self.segment_results, path_metadata=None if self.path is None else self.path.metadata(),
            approach_geometry=getattr(self, "approach_geometry", None),
            path_passed=bool(self.phase == "COMPLETE" and self.path is not None
                and [s["name"] for s in self.segment_results] == [s["name"] for s in self.path.segments]
                and all(s["coarse_reach_passed"] for s in self.segment_results)),
            path_sequence_completed=self.phase == "COMPLETE", lift_passed=False, grasp_commanded=False,
            insertion_reference_executed=self.insertion_reference_executed,
            command_schedule="Recorded wrist path plus outside/turn/prealign, each segment followed by 2s settling; all fingers command0",
            target_schedule="Fixed nominal path segments with equal 2s settle windows; errors logged, hard safety stops",
            hand_schedule="All fingers remain command0; source closing event is archived but not executed",
            critical_event_safety_check_hz=1000,
            contact_stop_details=self.contact_stop_details, limit_stop_details=self.limit_stop_details,
            final_joint_positions={self.model.joint(int(j)).name: float(q) for j,q in zip(self.body_map.joints,self.data.qpos[self.body_map.qpos])},
            max_foot_origin_drift_m=max(r["foot_origin_drift_m"] for r in self.samples),
            min_body_joint_margin_rad=min(r["min_body_joint_margin_rad"] for r in self.samples))
        if self.virtual_props:
            result.update(scope="EMPTY-HAND VIRTUAL-PROP PATH SCREEN: static wireframe table/crate, no contacts, no grasp or loaded lift",
                prepared_initialization_scope="Identical policy-generated robot state in five NEW scenes; virtual props cannot collide or support the robot",
                prop_geometry_gate_enforced=False,
                target_schedule="Fixed nominal path plus equal 2s settle; tracking errors logged, props ignored; original robot/floor/self/limits safety retained",
                contact_metric_scope="Table/crate contact, load, slip and lift metrics are NOT physically evaluated; placeholders are static",
                virtual_crate_motion="Static occupancy at original anchor; never visually attached to hands or animated as a successful lift")
        result["source_sha256"].update({f: sha256(PROJECT_ROOT/f) for f in (
            "common/r2v2_crate_height_sweep.py", "common/r2v2_crate_height_path.py", "common/r2v2_crate_height_state.py")})
        return result
