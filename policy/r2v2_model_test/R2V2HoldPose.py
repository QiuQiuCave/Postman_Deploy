import numpy as np

from FSM.FSMState import FSMState
from common.ctrlcomp import PolicyOutput, StateAndCmd
from common.utils import FSMCommand, FSMStateName


class R2V2HoldPose(FSMState):
    """Position-hold state used to validate the 28-DoF model and control path."""

    def __init__(
        self,
        state_cmd: StateAndCmd,
        policy_output: PolicyOutput,
        home_pose: np.ndarray,
    ):
        super().__init__(
            state_cmd,
            policy_output,
            FSMStateName.FIXEDPOSE,
            "r2v2_hold_pose",
        )
        self.home_pose = np.asarray(home_pose, dtype=np.float32)
        self.kps = np.full(state_cmd.num_joints, 40.0, dtype=np.float32)
        self.kds = np.full(state_cmd.num_joints, 2.0, dtype=np.float32)
        self.kps[:14] = 100.0
        self.kds[:14] = 5.0

    def enter(self) -> None:
        self.policy_output.kps = self.kps.copy()
        self.policy_output.kds = self.kds.copy()

    def run(self) -> None:
        self.policy_output.actions = self.home_pose.copy()
        self.policy_output.kps = self.kps.copy()
        self.policy_output.kds = self.kds.copy()

    def exit(self) -> None:
        self.policy_output.actions = self.state_cmd.q.copy()

    def checkChange(self) -> FSMStateName:
        if self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        self.state_cmd.skill_cmd = FSMCommand.INVALID
        return FSMStateName.FIXEDPOSE
