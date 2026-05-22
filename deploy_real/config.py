import yaml
import os


class Config:
    def __init__(self) -> None:
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_yaml_path = os.path.join(current_dir, "config", "real.yaml")
        with open(mujoco_yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            self.net = config["net"]
            self.num_joints = config["num_joints"]
            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]
            self.control_dt = config["control_dt"]
            self.error_over_time = config.get("error_over_time", 5)
            self.max_control_over_time = max(
                1,
                int(config.get("max_control_over_time", self.error_over_time)),
            )

            # Real-hardware skill gates. Keep dynamic demos opt-in: sim2sim
            # success is not a hardware safety-ladder pass.
            self.enable_box_handoff_stand = config.get("enable_box_handoff_stand", True)
            self.enable_box_hold_stand = config.get("enable_box_hold_stand", True)
            self.enable_walk_tracking = config.get("enable_walk_tracking", True)
            self.enable_run_tracking = config.get("enable_run_tracking", False)
            self.enable_jump_tracking = config.get("enable_jump_tracking", False)
            self.enable_dance_tracking = config.get("enable_dance_tracking", False)

            self.log_real_run = config.get("log_real_run", True)
            self.real_log_dir = config.get("real_log_dir", "logs/real_deploy")
            self.real_log_interval = max(1, int(config.get("real_log_interval", 1)))

            # Disabled by default. If set > 0, each commanded joint target is
            # clamped against the last sent target by this many radians/tick.
            self.max_target_delta = float(config.get("max_target_delta", 0.0))
