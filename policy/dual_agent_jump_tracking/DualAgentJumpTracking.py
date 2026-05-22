from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName
from policy.dual_agent_run_tracking.DualAgentRunTracking import DualAgentRunTracking


class DualAgentJumpTracking(DualAgentRunTracking):
    """Runtime for the jump lower-body tracking demo in sim2sim and deploy_real."""

    policy_dir_name = "dual_agent_jump_tracking"
    config_filename = "DualAgentJumpTracking.yaml"
    state_name = FSMStateName.DUAL_AGENT_JUMP_TRACK
    state_name_str = "dual_agent_jump_tracking_mode"
    display_name = "DualAgentJumpTracking"
