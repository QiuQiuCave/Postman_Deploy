import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

from common.path_config import PROJECT_ROOT

import time
import mujoco.viewer
import mujoco
import numpy as np
import yaml
import os
from common.ctrlcomp import *
from FSM.FSM import *
from common.utils import get_gravity_orientation
from common.joystick import JoyStick, JoystickButton



def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    mujoco_yaml_path = os.path.join(current_dir, "config", "mujoco.yaml")
    with open(mujoco_yaml_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        xml_path = os.path.join(PROJECT_ROOT, config["xml_path"])
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    mj_per_step_duration = simulation_dt * control_decimation
    num_joints = m.nu
    policy_output_action = np.zeros(num_joints, dtype=np.float32)
    kps = np.zeros(num_joints, dtype=np.float32)
    kds = np.zeros(num_joints, dtype=np.float32)
    sim_counter = 0
    
    state_cmd = StateAndCmd(num_joints)
    policy_output = PolicyOutput(num_joints)
    FSM_controller = FSM(state_cmd, policy_output)

    # Anchor body (torso_link) id — DualAgentTracking reads its world pose.
    anchor_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")

    # Transport box handles — see deploy_mujoco_keyboard_input.py for rationale.
    box_id          = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "transport_box")
    box_jnt_idx     = m.body_jntadr[box_id]
    box_qpos_adr    = m.jnt_qposadr[box_jnt_idx]
    box_qvel_adr    = m.jnt_dofadr[box_jnt_idx]
    box_park_pos    = np.array([100.0, 100.0, 0.15], dtype=np.float64)
    box_offset_base = np.array([0.32, 0.0, 0.26], dtype=np.float64)
    box_hold_dur    = 1.0
    box_active      = False
    box_hold_until  = 0.0
    box_hold_pos    = np.zeros(3, dtype=np.float64)
    box_hold_quat   = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    joystick = JoyStick()
    Running = True
    with mujoco.viewer.launch_passive(m, d) as viewer:
        sim_start_time = time.time()
        while viewer.is_running() and Running:
            try:
                if(joystick.is_button_pressed(JoystickButton.SELECT)):
                    Running = False

                joystick.update()
                if joystick.is_button_released(JoystickButton.L3):
                    state_cmd.skill_cmd = FSMCommand.PASSIVE
                if joystick.is_button_released(JoystickButton.START):
                    state_cmd.skill_cmd = FSMCommand.POS_RESET
                # R1 group: locomotion / box-transport / mimic family.
                if joystick.is_button_released(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.R1):
                    state_cmd.skill_cmd = FSMCommand.LOCO
                if joystick.is_button_released(JoystickButton.B) and joystick.is_button_pressed(JoystickButton.R1):
                    state_cmd.skill_cmd = FSMCommand.SKILL_BOX_TRANSPORT_V
                if joystick.is_button_released(JoystickButton.X) and joystick.is_button_pressed(JoystickButton.R1):
                    state_cmd.skill_cmd = FSMCommand.DUAL_AGENT_BOX_TRANS_VEL
                if joystick.is_button_released(JoystickButton.Y) and joystick.is_button_pressed(JoystickButton.R1):
                    state_cmd.skill_cmd = FSMCommand.SKILL_4
                # L1 group: motion-tracking demos. b/x/y+l1 reserved for future demos.
                if joystick.is_button_released(JoystickButton.A) and joystick.is_button_pressed(JoystickButton.L1):
                    state_cmd.skill_cmd = FSMCommand.DUAL_AGENT_TRACK

                state_cmd.vel_cmd[0] = -joystick.get_axis_value(1)
                state_cmd.vel_cmd[1] = -joystick.get_axis_value(0)
                state_cmd.vel_cmd[2] = -joystick.get_axis_value(3)
                
                step_start = time.time()
                
                tau = pd_control(policy_output_action,
                                 d.qpos[7:7+num_joints], kps, np.zeros_like(kps),
                                 d.qvel[6:6+num_joints], kds)
                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                sim_counter += 1

                # 硬 pin:悬吊期内每个 sim step 把箱子 qpos/qvel 钉死,
                # 物理上等价于理想刚性绳,防 contact 冲击漂移。
                if box_hold_until > 0.0:
                    if time.time() < box_hold_until:
                        d.qpos[box_qpos_adr:box_qpos_adr+3]   = box_hold_pos
                        d.qpos[box_qpos_adr+3:box_qpos_adr+7] = box_hold_quat
                        d.qvel[box_qvel_adr:box_qvel_adr+6]   = 0.0
                    else:
                        box_hold_until = 0.0
                        print("BoxTransport: released gravity hold.")

                if sim_counter % control_decimation == 0:

                    qj = d.qpos[7:7+num_joints]
                    dqj = d.qvel[6:6+num_joints]
                    quat = d.qpos[3:7]
                    
                    omega = d.qvel[3:6]
                    gravity_orientation = get_gravity_orientation(quat)

                    # body-frame linear velocity of the base (body index 1).
                    cvel = np.zeros(6)
                    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, 1, cvel, 1)
                    base_lin_vel_body = cvel[3:6].astype(np.float32)

                    state_cmd.q = qj.copy()
                    state_cmd.dq = dqj.copy()
                    state_cmd.gravity_ori = gravity_orientation.copy()
                    state_cmd.base_quat = quat.copy()
                    state_cmd.ang_vel = omega.copy()
                    state_cmd.base_lin_vel = base_lin_vel_body
                    state_cmd.anchor_pos_w  = d.xpos[anchor_body_id].copy().astype(np.float32)
                    state_cmd.anchor_quat_w = d.xquat[anchor_body_id].copy().astype(np.float32)

                    FSM_controller.run()
                    policy_output_action = policy_output.actions.copy()
                    kps = policy_output.kps.copy()
                    kds = policy_output.kds.copy()

                    cur = FSM_controller.cur_policy
                    # `needs_transport_box` is declared on policy classes that
                    # want the box in their grasp region (BoxTransport, DualAgent).
                    is_box = getattr(cur, "needs_transport_box", False)
                    ramp_complete = (not getattr(cur, "ramping", True))
                    if is_box and ramp_complete and not box_active:
                        pelvis_pos  = d.qpos[0:3].copy()
                        pelvis_quat = d.qpos[3:7].copy()
                        offset_world = np.zeros(3, dtype=np.float64)
                        mujoco.mju_rotVecQuat(offset_world, box_offset_base, pelvis_quat)
                        box_hold_pos[:]  = pelvis_pos + offset_world
                        box_hold_quat[:] = pelvis_quat
                        d.qpos[box_qpos_adr:box_qpos_adr+3]   = box_hold_pos
                        d.qpos[box_qpos_adr+3:box_qpos_adr+7] = box_hold_quat
                        d.qvel[box_qvel_adr:box_qvel_adr+6]   = 0.0
                        box_hold_until = time.time() + box_hold_dur
                        mujoco.mj_forward(m, d)
                        box_active = True
                        print(f"BoxTransport: spawned box, pinned for {box_hold_dur:.1f}s.")
                    elif (not is_box) and box_active:
                        d.qpos[box_qpos_adr:box_qpos_adr+3]   = box_park_pos
                        d.qpos[box_qpos_adr+3:box_qpos_adr+7] = [1.0, 0.0, 0.0, 0.0]
                        d.qvel[box_qvel_adr:box_qvel_adr+6]   = 0.0
                        box_hold_until = 0.0
                        mujoco.mj_forward(m, d)
                        box_active = False
                        print("BoxTransport: parked box.")
            except ValueError as e:
                print(str(e))
            
            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
        