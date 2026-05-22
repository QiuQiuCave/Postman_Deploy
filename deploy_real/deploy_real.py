import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))

import csv
from datetime import datetime
from common.path_config import PROJECT_ROOT
from common.ctrlcomp import *
from FSM.FSM import *
from typing import Union
import numpy as np
import time

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation_real, transform_imu_data
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, config: Config):
        self.config = config
        self.remote_controller = RemoteController()
        self.num_joints = config.num_joints
        self.control_dt = config.control_dt
        
        
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()
        
        # inital connection
        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)
        
        self.wait_for_low_state()
        
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        
        self.policy_output_action = np.zeros(self.num_joints, dtype=np.float32)
        self.kps = np.zeros(self.num_joints, dtype=np.float32)
        self.kds = np.zeros(self.num_joints, dtype=np.float32)
        self.qj = np.zeros(self.num_joints, dtype=np.float32)
        self.dqj = np.zeros(self.num_joints, dtype=np.float32)
        self.quat = np.zeros(4, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.gravity_orientation = np.array([0,0,-1], dtype=np.float32)
        
        self.state_cmd = StateAndCmd(self.num_joints)
        self.policy_output = PolicyOutput(self.num_joints)
        self.FSM_controller = FSM(self.state_cmd, self.policy_output)
        
        self.running = True
        self.counter_over_time = 0
        self.combo_latches = {}
        self.last_command_name = "none"
        self.last_sent_action = None

        self.log_handle = None
        self.log_writer = None
        self.log_tick = 0
        self._init_real_logger()
        self._print_deploy_profile()
        
        
    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def _init_real_logger(self):
        if not self.config.log_real_run:
            return
        log_dir = Path(self.config.real_log_dir)
        if not log_dir.is_absolute():
            log_dir = Path(PROJECT_ROOT) / log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"real_deploy_{stamp}.csv"
        self.log_handle = log_path.open("w", newline="", encoding="utf-8")
        self.log_writer = csv.DictWriter(
            self.log_handle,
            fieldnames=[
                "time_s",
                "fsm_state",
                "last_command",
                "loop_time_s",
                "overtime_count",
                "buttons",
                "q",
                "dq",
                "q_cmd",
                "kp",
                "kd",
            ],
        )
        self.log_writer.writeheader()
        print(f"Real deploy log: {log_path}")

    def _print_deploy_profile(self):
        print("RealDeploy enabled skill gates:")
        print(f"  B+R1 BoxHandoffStand: {self.config.enable_box_handoff_stand}")
        print(f"  X+R1 BoxHoldStand:    {self.config.enable_box_hold_stand}")

        tracking_entries = [
            ("A+L1 walk", self.config.enable_walk_tracking, self.FSM_controller.dual_agent_tracking_policy),
            ("B+L1 run", self.config.enable_run_tracking, self.FSM_controller.dual_agent_run_tracking_policy),
            ("X+L1 jump", self.config.enable_jump_tracking, self.FSM_controller.dual_agent_jump_tracking_policy),
            ("Y+L1 dance", self.config.enable_dance_tracking, self.FSM_controller.dual_agent_dance_tracking_policy),
        ]
        print("RealDeploy tracking runtime clips:")
        for label, enabled, policy in tracking_entries:
            motion = getattr(policy, "motion", None)
            if motion is None:
                clip = "motion=unavailable"
            else:
                clip = (
                    f"motion={motion.total_frames} frames @ {motion.fps}Hz "
                    f"({motion.total_frames / motion.fps:.2f}s)"
                )
            print(f"  {label}: enabled={enabled} | {clip}")

    def close(self):
        if self.log_handle is not None:
            self.log_handle.flush()
            self.log_handle.close()
            self.log_handle = None

    @staticmethod
    def _fmt_array(values):
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        return " ".join(f"{x:.6g}" for x in arr)

    def _write_real_log(self, loop_time, q_cmd, kps, kds):
        if self.log_writer is None:
            return
        self.log_tick += 1
        if self.log_tick % self.config.real_log_interval != 0:
            return
        cur_policy = self.FSM_controller.cur_policy
        self.log_writer.writerow(
            {
                "time_s": f"{time.time():.6f}",
                "fsm_state": getattr(cur_policy, "name_str", "unknown"),
                "last_command": self.last_command_name,
                "loop_time_s": f"{loop_time:.6f}",
                "overtime_count": self.counter_over_time,
                "buttons": "".join(str(int(x)) for x in self.remote_controller.button),
                "q": self._fmt_array(self.qj),
                "dq": self._fmt_array(self.dqj),
                "q_cmd": self._fmt_array(q_cmd),
                "kp": self._fmt_array(kps),
                "kd": self._fmt_array(kds),
            }
        )
        self.log_handle.flush()

    def _combo_pressed_once(self, button_id, modifier_id):
        key = (button_id, modifier_id)
        active = (
            self.remote_controller.is_button_pressed(button_id)
            and self.remote_controller.is_button_pressed(modifier_id)
        )
        was_active = self.combo_latches.get(key, False)
        self.combo_latches[key] = active
        return active and not was_active

    def _set_skill_command(self, command, label):
        self.state_cmd.skill_cmd = command
        should_print = label != self.last_command_name
        self.last_command_name = label
        if should_print:
            print(f"RealDeploy command: {label}")
            hint = {
                FSMCommand.BOX_HANDOFF_STAND: "  Flow: open hands; operator can place the box between the wrists.",
                FSMCommand.BOX_HOLD_STAND: "  Flow: clamp/hold stand; confirm the box is stable before tracking.",
                FSMCommand.DUAL_AGENT_TRACK: "  Flow: walk tracking; use only after hold stand is stable.",
                FSMCommand.DUAL_AGENT_RUN_TRACK: "  Flow: run tracking; requires a separate hardware safety ladder.",
                FSMCommand.DUAL_AGENT_JUMP_TRACK: "  Flow: jump tracking is high risk on hardware.",
                FSMCommand.DUAL_AGENT_DANCE_TRACK: "  Flow: dance tracking is sim2sim-first; validate separately.",
            }.get(command)
            if hint:
                print(hint)

    def _set_if_enabled(self, enabled, command, label, disabled_reason):
        if enabled:
            self._set_skill_command(command, label)
        else:
            self.last_command_name = f"blocked:{label}"
            print(f"RealDeploy blocked {label}: {disabled_reason}")

    def _process_remote_commands(self):
        # F1 is level-triggered and always wins.
        if self.remote_controller.is_button_pressed(KeyMap.F1):
            self._set_skill_command(FSMCommand.PASSIVE, "F1->PASSIVE")
            return
        if self.remote_controller.is_button_pressed(KeyMap.start):
            self._set_skill_command(FSMCommand.POS_RESET, "Start->POS_RESET")
            return

        if self._combo_pressed_once(KeyMap.A, KeyMap.R1):
            self._set_skill_command(FSMCommand.LOCO, "A+R1->LOCO")
        elif self._combo_pressed_once(KeyMap.B, KeyMap.R1):
            self._set_if_enabled(
                self.config.enable_box_handoff_stand,
                FSMCommand.BOX_HANDOFF_STAND,
                "B+R1->BOX_HANDOFF_STAND",
                "enable_box_handoff_stand=false",
            )
        elif self._combo_pressed_once(KeyMap.X, KeyMap.R1):
            self._set_if_enabled(
                self.config.enable_box_hold_stand,
                FSMCommand.BOX_HOLD_STAND,
                "X+R1->BOX_HOLD_STAND",
                "enable_box_hold_stand=false",
            )
        elif self._combo_pressed_once(KeyMap.A, KeyMap.L1):
            self._set_if_enabled(
                self.config.enable_walk_tracking,
                FSMCommand.DUAL_AGENT_TRACK,
                "A+L1->DUAL_AGENT_TRACK",
                "enable_walk_tracking=false",
            )
        elif self._combo_pressed_once(KeyMap.B, KeyMap.L1):
            self._set_if_enabled(
                self.config.enable_run_tracking,
                FSMCommand.DUAL_AGENT_RUN_TRACK,
                "B+L1->DUAL_AGENT_RUN_TRACK",
                "enable_run_tracking=false; pass the run safety ladder first",
            )
        elif self._combo_pressed_once(KeyMap.X, KeyMap.L1):
            self._set_if_enabled(
                self.config.enable_jump_tracking,
                FSMCommand.DUAL_AGENT_JUMP_TRACK,
                "X+L1->DUAL_AGENT_JUMP_TRACK",
                "enable_jump_tracking=false; pass a dedicated jump safety ladder first",
            )
        elif self._combo_pressed_once(KeyMap.Y, KeyMap.L1):
            self._set_if_enabled(
                self.config.enable_dance_tracking,
                FSMCommand.DUAL_AGENT_DANCE_TRACK,
                "Y+L1->DUAL_AGENT_DANCE_TRACK",
                "enable_dance_tracking=false; pass a dedicated dance safety ladder first",
            )

    def _send_damping(self, reason):
        print(f"RealDeploy safety damping: {reason}")
        create_damping_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)
        self.state_cmd.skill_cmd = FSMCommand.PASSIVE

    def _sleep_remaining(self, loop_start_time):
        delta_time = time.time() - loop_start_time
        if delta_time < self.control_dt:
            time.sleep(self.control_dt - delta_time)
        return delta_time

    @staticmethod
    def _outputs_are_finite(*arrays):
        return all(np.isfinite(np.asarray(x)).all() for x in arrays)

    def _limit_target_delta(self, q_cmd):
        if self.config.max_target_delta <= 0.0:
            return q_cmd
        if self.last_sent_action is None:
            self.last_sent_action = self.qj.copy()
        limited = np.clip(
            q_cmd,
            self.last_sent_action - self.config.max_target_delta,
            self.last_sent_action + self.config.max_target_delta,
        )
        return limited.astype(np.float32)
        
    def run(self):
        try:
            loop_start_time = time.time()

            self._process_remote_commands()

            self.state_cmd.vel_cmd[0] =  self.remote_controller.ly
            self.state_cmd.vel_cmd[1] =  self.remote_controller.lx * -1
            self.state_cmd.vel_cmd[2] =  self.remote_controller.rx * -1

            for i in range(self.num_joints):
                self.qj[i] = self.low_state.motor_state[i].q
                self.dqj[i] = self.low_state.motor_state[i].dq

            # imu_state quaternion: w, x, y, z
            quat = np.asarray(self.low_state.imu_state.quaternion, dtype=np.float32).reshape(4)
            ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32).reshape(3)
            
            gravity_orientation = get_gravity_orientation_real(quat)
            
            self.state_cmd.q = self.qj.copy()
            self.state_cmd.dq = self.dqj.copy()
            self.state_cmd.gravity_ori = gravity_orientation.copy()
            self.state_cmd.ang_vel = ang_vel.copy()
            self.state_cmd.base_quat = quat
            # TODO: body-frame velocity estimator; LOCO_NEW is sim-only on hardware.
            self.state_cmd.base_lin_vel = np.zeros(3, dtype=np.float32)
            
            self.FSM_controller.run()
            policy_output_action = self.policy_output.actions.copy()
            kps = self.policy_output.kps.copy()
            kds = self.policy_output.kds.copy()

            if not self._outputs_are_finite(policy_output_action, kps, kds):
                self._send_damping("non-finite policy output")
                self._sleep_remaining(loop_start_time)
                return

            policy_output_action = self._limit_target_delta(policy_output_action)
            
            # Build low cmd
            for i in range(self.num_joints):
                self.low_cmd.motor_cmd[i].q = policy_output_action[i]
                self.low_cmd.motor_cmd[i].qd = 0
                self.low_cmd.motor_cmd[i].kp = kps[i]
                self.low_cmd.motor_cmd[i].kd = kds[i]
                self.low_cmd.motor_cmd[i].tau = 0
                
            # send the command
            # create_damping_cmd(controller.low_cmd) # only for debug
            self.send_cmd(self.low_cmd)
            self.last_sent_action = policy_output_action.copy()
            
            loop_end_time = time.time()
            delta_time = loop_end_time - loop_start_time
            self._write_real_log(delta_time, policy_output_action, kps, kds)
            if(delta_time < self.control_dt):
                time.sleep(self.control_dt - delta_time)
                self.counter_over_time = 0
            else:
                print("control loop over time.")
                self.counter_over_time += 1
                if self.counter_over_time >= self.config.max_control_over_time:
                    self._send_damping(
                        f"control loop overtime for {self.counter_over_time} consecutive ticks"
                    )
            pass
        except ValueError as e:
            print(str(e))
            pass
        
        pass
        
        
if __name__ == "__main__":
    config = Config()
    # Initialize DDS communication
    ChannelFactoryInitialize(0, config.net)

    controller = Controller(config)

    try:
        while True:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.is_button_pressed(KeyMap.select):
                break
    except KeyboardInterrupt:
        pass
    finally:
        create_damping_cmd(controller.low_cmd)
        controller.send_cmd(controller.low_cmd)
        controller.close()
    print("Exit")
    
