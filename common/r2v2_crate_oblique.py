"""Oblique insertion probes using externally supported wrists, never a body policy.

Left-only by default: the real right hand stays parked >1m away and its measured
contact load must remain zero. The free crate has no weld, force or runtime reset.
Geometry, hand force limits, contact solver and the original bilateral baseline
are retained. The 45deg exploratory tilt stop is NOT a successful-level-lift gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from common.r2v2_crate import load_crate_config
from common.r2v2_crate_hand_preview import _local_part_clouds, _wrist_rotation
from common.r2v2_crate_lift import CrateLiftExperiment, CrateLiftParameters, _blend
from common.r2v2_grasp_recording import transform_pose
from r2v2_description.model import SIDES


@dataclass(frozen=True)
class ObliqueParameters:
    tip_up_deg: float = 0.
    yaw_deg: float = 0.
    roll_deg: float = 0.
    insertion_m: float = .060
    closed_curl_rad: float = .80
    seating_m: float = .0175
    active_sides: tuple[str, ...] = ("left",)

    def __post_init__(self):
        for name in ("tip_up_deg", "yaw_deg", "roll_deg", "insertion_m", "closed_curl_rad", "seating_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"Invalid {name}")
        if any(abs(getattr(self, name)) > 45 for name in ("tip_up_deg", "yaw_deg", "roll_deg")):
            raise ValueError("Probe angles must be within +/-45deg")
        if not .02 <= self.insertion_m <= .08 or not 0 <= self.seating_m <= .03:
            raise ValueError("Require insertion 20..80mm and seating 0..30mm")
        if not 0 <= self.closed_curl_rad <= 1.4:
            raise ValueError("Curl outside source limits")
        if self.active_sides not in (("left",), ("left", "right")):
            raise ValueError("Use left only or the mirrored pair")


def rotation(axis, degrees):
    q = np.empty(4)
    mujoco.mju_axisAngle2Quat(q, np.asarray(axis, dtype=float), np.deg2rad(degrees))
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, q)
    return result.reshape(3, 3)


def oblique_layout(model, data, side, crate, candidate, table_height):
    """Place swept FOUR-finger cross-section at the aperture, then inspect all contacts.

    +tip_up points fingertips upward, +yaw points inward fingers toward box +X;
    roll banks the hand about its finger-extension axis (mirrored on the right).
    Insertion depth is measured NORMAL to the wall, not along the oblique path.
    Swept bounds are conservative mesh-vertex bounds, not collision certification.
    """
    sign = 1 if side == "left" else -1
    r = (rotation((0, 0, 1), sign*candidate.yaw_deg)
         @ rotation((1, 0, 0), -sign*candidate.tip_up_deg)
         @ _wrist_rotation(side) @ rotation((1, 0, 0), sign*candidate.roll_deg))
    direction = r[:, 0]
    inward = np.array([0., -sign, 0.])
    approach_cos = float(inward @ direction)
    if approach_cos <= .3:
        raise ValueError("Insufficient inward component")
    parts = {name: vertices @ r.T for name, vertices in _local_part_clouds(model, data, side).items()}
    four = np.concatenate([parts[name] for name in ("index", "middle", "ring", "pinky")])
    all_points = np.concatenate(list(parts.values()))
    shortest_tip = min(float((parts[name] @ inward).max()) for name in ("index", "middle", "ring", "pinky"))
    wall = -crate.width/2
    p = inward*(wall+crate.wall_thickness+candidate.insertion_m-shortest_tip)
    # Thick top beam extends farther inward than the ordinary thin sidewall.
    crossings = []
    for plane in (wall, wall+crate.wall_thickness, wall+crate.handle_beam_thickness):
        distances = (plane-(four+p) @ inward)/approach_cos
        crossings.append(four+p+distances[:, None]*direction)
    swept = np.concatenate(crossings)
    low, high = swept.min(0), swept.max(0)
    center = (low+high)/2
    p[0] -= center[0]
    p[2] += table_height+(crate.handle_opening_bottom+crate.handle_opening_top)/2-center[2]
    retreat = (float(((all_points+p) @ inward).max())-wall+.025)/approach_cos
    initial = p-retreat*direction
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, r.ravel())
    tip_depths = {name: float(((vertices+p) @ inward).max()-wall-crate.wall_thickness)
                  for name, vertices in parts.items() if name in ("index", "middle", "ring", "pinky")}
    return dict(initial=initial, inserted=p, quaternion=quaternion, rotation=r,
        frame_note="Target pose relative to nominal resting crate bottom at table height; not the initial 1mm settling gap or a measured final pose",
        insertion_direction_world=direction, sweep_width_m=float(high[0]-low[0]),
        sweep_height_m=float(high[2]-low[2]),
        aperture_width_margin_m=float(crate.handle_opening_width-(high[0]-low[0])),
        aperture_height_margin_m=float(crate.handle_opening_height-(high[2]-low[2])),
        initial_all_hand_clearance_m=float(wall-((all_points+initial) @ inward).max()),
        per_finger_normal_insertion_m=tip_depths,
        pose_in_crate_frame=transform_pose(np.block([[r, (p-[0, 0, table_height])[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])))


class ObliqueCrateExperiment(CrateLiftExperiment):
    def __init__(self, candidate=None, keep_trace=True):
        self.candidate = candidate or ObliqueParameters()
        self.active_sides = self.candidate.active_sides
        self.keep_trace = keep_trace
        self.pickup_hold_s = self.max_pickup_hold_s = 0.
        self.level_hold_s = self.max_level_hold_s = 0.
        self.pickup_verified = False
        # Tilt may be observed up to 45deg in this isolated probe. A level lift
        # still requires <=8deg; original bilateral experiment is unchanged.
        params = replace(CrateLiftParameters(), insertion_m=self.candidate.insertion_m,
            closed_curl_rad=max(.000001, self.candidate.closed_curl_rad),
            grasp_enabled=self.candidate.closed_curl_rad > 0,
            closure_seating_m=self.candidate.seating_m,
            max_tilt_deg=45., hold_s=2.)
        super().__init__(params, crate_params=replace(load_crate_config(), width=.26))
        # These baseline clearance fields describe the old flat fixture, not
        # the newly placed oblique wrists. New clearances live in geometry.
        self.layout.pop("initial_leading_finger_clearance_m", None)
        self.layout.pop("initial_palm_clearance_m", None)
        self.geometry = {}
        for side in SIDES:
            if side in self.active_sides:
                g = oblique_layout(self.model, self.data, side, self.crate_params, self.candidate, self.table_height)
                self.geometry[side] = g
                initial, inserted, quat = g["initial"], g["inserted"], g["quaternion"]
            else:
                initial = inserted = np.array([1.5, -1.5, .8])
                quat = self.layout["wrist_quaternions"][side].copy()
            self.initial_wrists[side] = initial.copy()
            self.inserted_wrists[side] = inserted.copy()
            self.layout["initial_wrist_positions"][side] = initial.copy()
            self.layout["inserted_wrist_positions"][side] = inserted.copy()
            self.layout["wrist_quaternions"][side] = quat.copy()
            self.data.mocap_pos[self.layout["mocap_ids"][side]] = initial
            self.data.mocap_quat[self.layout["mocap_ids"][side]] = quat
            address = self.model.joint(self.layout["wrist_free_joints"][side]).qposadr[0]
            self.data.qpos[address:address+7] = np.r_[initial, quat]
        # Pose edits are initialization only, before any physics integration.
        assert self.data.time == 0.
        mujoco.mj_forward(self.model, self.data)
        self.sync()
        self.samples.clear()
        self.record()

    def sync(self):
        result = super().sync()
        if self.baseline_relations is not None:
            result["grasp_slip_m"] = max(result["side_slip_m"][s] for s in self.active_sides)
            angles = {}
            for s in self.active_sides:
                now = np.asarray(result["hands"][s]["T_wrist_crate"])
                rel = now[:3, :3] @ self.baseline_relations[s][:3, :3].T
                angles[s] = float(np.rad2deg(np.arccos(np.clip((np.trace(rel)-1)/2, -1, 1))))
            result["active_angular_slip_deg"] = angles
        return result

    def enter(self, phase):
        self.phase, self.phase_start = phase, float(self.data.time)
        self.stable_since = self.bad_since = None
        self.transitions.append({"time_s": self.phase_start, "state": phase})
        if phase == "CLOSE":
            for side in self.active_sides:
                self.hands.command(side, int(self.params.grasp_enabled))
        if phase == "PROBE_LIFT":
            self.baseline_relations = {s: np.asarray(self.current_metrics["hands"][s]["T_wrist_crate"]).copy() for s in SIDES}

    def _supported_pickup(self):
        m = self.current_metrics
        angles = m.get("active_angular_slip_deg")
        if not isinstance(angles, dict) or any(
                s not in angles or not np.isfinite(angles[s]) or not 0. <= angles[s] < 5.
                for s in self.active_sides):
            return False
        load = sum(m["hands"][s]["vertical_force_N"] for s in self.active_sides)
        bearing = all(m["hands"][s]["has_bearing_finger_contact"] for s in self.active_sides)
        no_other_hand = all(not m["hands"][s]["contacts"] for s in SIDES if s not in self.active_sides)
        return (m["clearance_m"] >= .008 and not m["table_bearing_contact"]
            and load >= .8*m["crate_weight_N"] and bearing and no_other_hand
            and m["grasp_slip_m"] is not None and m["grasp_slip_m"] < .015
            and m["crate_linear_speed_m_s"] < .02 and m["crate_angular_speed_rad_s"] < .15)

    def _gate(self):
        p, m = self.params, self.current_metrics
        elapsed = float(self.data.time-self.phase_start)
        if self.phase == "READY" and elapsed >= p.ready_s:
            self.enter("INSERT")
        elif self.phase == "INSERT" and elapsed >= p.insert_s:
            self.enter("INSERT_SETTLE")
        elif self.phase == "INSERT_SETTLE" and elapsed >= p.insert_settle_s:
            self.enter("CLOSE")
        elif self.phase == "CLOSE":
            contact = all(m["hands"][s]["finger_normal_force_N"] > .05 for s in self.active_sides)
            if self._stable(elapsed >= p.close_minimum_s and contact, p.contact_stable_s):
                self.enter("PROBE_LIFT")
            elif elapsed >= p.close_timeout_s:
                self.fail("No sustained active-hand finger contact after closure")
        elif self.phase == "PROBE_LIFT" and elapsed >= p.lift_s:
            self.enter("HOLD")
        elif self.phase == "HOLD":
            pickup = self._supported_pickup()
            level = pickup and m["clearance_m"] >= .08 and m["crate_tilt_deg"] <= 8.
            self.pickup_hold_s = self.pickup_hold_s+.01 if pickup else 0.
            self.level_hold_s = self.level_hold_s+.01 if level else 0.
            self.max_pickup_hold_s = max(self.max_pickup_hold_s, self.pickup_hold_s)
            self.max_level_hold_s = max(self.max_level_hold_s, self.level_hold_s)
            self.pickup_verified |= self.pickup_hold_s >= 1.-1e-9
            self.hold_samples.append(dict(time_s=float(self.data.time), valid=bool(level),
                pickup=bool(pickup), clearance_m=m["clearance_m"], tilt_deg=m["crate_tilt_deg"]))
            if elapsed >= p.hold_s:
                self.enter("COMPLETE")

    def _motion(self):
        elapsed, p = self.data.time-self.phase_start, self.params
        alpha = 0. if self.phase == "READY" else (_blend(elapsed/p.insert_s) if self.phase == "INSERT" else 1.)
        lift = 0.
        if p.grasp_enabled and self.phase == "CLOSE":
            lift = p.closure_seating_m*_blend(elapsed/p.close_minimum_s)
        elif self.phase in ("PROBE_LIFT", "HOLD", "COMPLETE"):
            lift = (p.closure_seating_m if p.grasp_enabled else 0.)
            lift += p.lift_height_m*(_blend(elapsed/p.lift_s) if self.phase == "PROBE_LIFT" else 1.)
        for side in SIDES:
            target = self.layout["mocap_ids"][side]
            if side in self.active_sides:
                self.data.mocap_pos[target] = (1-alpha)*self.initial_wrists[side]+alpha*self.inserted_wrists[side]+[0., 0., lift]
            else:
                self.data.mocap_pos[target] = self.initial_wrists[side]
            self.data.mocap_quat[target] = self.layout["wrist_quaternions"][side]

    def _safety(self):
        super()._safety()
        for side in SIDES:
            if side not in self.active_sides and self.current_metrics["hands"][side]["contacts"]:
                self.fail("Parked hand contacted the crate")

    def record(self):
        if self.keep_trace:
            super().record()

    def report(self):
        completed = self.phase == "COMPLETE" and self.failure is None
        passed = completed and self.level_hold_s >= self.params.hold_s-.02
        relative = {s: self.current_metrics["hands"][s]["T_wrist_crate"] for s in self.active_sides}
        return dict(lift_passed=bool(passed), pickup_verified=bool(completed and self.pickup_hold_s >= 1.-1e-9),
            experiment_completed=completed, phase=self.phase, failure=self.failure,
            candidate=asdict(self.candidate), parameters=asdict(self.params), crate_parameters=asdict(self.crate_params),
            scope="isolated wrist-fixture probe; left-only unless explicitly bilateral; NOT full-body reachability",
            duration_s=float(self.data.time), final_metrics=self.current_metrics, peaks=self.peaks,
            geometry=getattr(self, "geometry", {}), final_T_wrist_crate=relative,
            max_pickup_hold_s=self.max_pickup_hold_s, max_level_hold_s=self.max_level_hold_s,
            hold_samples=self.hold_samples, transitions=self.transitions, layout=self.layout,
            criteria=dict(exploratory_tilt_stop_deg=45., level_tilt_max_deg=8.,
                level_clearance_m=.08, level_hold_s=2., pickup_clearance_m=.008, pickup_hold_s=1.,
                slip_max_m=.015, angular_slip_max_deg=5., min_net_weight_fraction=.8,
                no_table_bearing=True, real_finger_handle_contact=True, inactive_hand_no_contact=True),
            hand_configuration=self.hand_cfg, crate_welds=0, crate_runtime_resets=0,
            no_external_crate_forces=not np.any(self.data.xfrc_applied) and not np.any(self.data.qfrc_applied),
            mujoco_version=mujoco.__version__, source_sha256={str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
