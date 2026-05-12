from FSM.FSMState import FSMStateName
from policy.dual_agent_run_tracking.DualAgentRunTracking import DualAgentRunTracking


class DualAgentDanceTracking(DualAgentRunTracking):
    """MuJoCo sim2sim runtime for the LAFAN dance lower-body tracking demo."""

    policy_dir_name = "dual_agent_dance_tracking"
    config_filename = "DualAgentDanceTracking.yaml"
    state_name = FSMStateName.DUAL_AGENT_DANCE_TRACK
    state_name_str = "dual_agent_dance_tracking_mode"
    display_name = "DualAgentDanceTracking"
