import numpy as np

from FSM.FSMState import FSMState
from common.ctrlcomp import PolicyOutput, StateAndCmd
from common.utils import FSMCommand, FSMStateName


class R2V2Passive(FSMState):
    """Zero-torque-like state: target the measured joint position with low gains."""

    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__(
            state_cmd,
            policy_output,
            FSMStateName.PASSIVE,
            "r2v2_passive",
        )
        self.kps = np.full(state_cmd.num_joints, 2.0, dtype=np.float32)
        self.kds = np.full(state_cmd.num_joints, 0.5, dtype=np.float32)

    def enter(self) -> None:
        self.policy_output.kps = self.kps.copy()
        self.policy_output.kds = self.kds.copy()

    def run(self) -> None:
        self.policy_output.actions = self.state_cmd.q.copy()
        self.policy_output.kps = self.kps.copy()
        self.policy_output.kds = self.kds.copy()

    def exit(self) -> None:
        pass

    def checkChange(self) -> FSMStateName:
        if self.state_cmd.skill_cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        self.state_cmd.skill_cmd = FSMCommand.INVALID
        return FSMStateName.PASSIVE
