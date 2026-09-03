import time

from common.ctrlcomp import PolicyOutput, StateAndCmd
from common.utils import FSMCommand, FSMStateName
from policy.r2v2_model_test import R2V2HoldPose, R2V2Passive


class R2V2FSM:
    """Small FSM adapter retaining Postman_Deploy's state/output contract."""

    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput, home_pose):
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.states = {
            FSMStateName.PASSIVE: R2V2Passive(state_cmd, policy_output),
            FSMStateName.FIXEDPOSE: R2V2HoldPose(
                state_cmd,
                policy_output,
                home_pose,
            ),
        }
        self.cur_policy = self.states[FSMStateName.FIXEDPOSE]
        self.cur_policy.enter()
        self._last_switch = time.monotonic()

    @property
    def state_name(self) -> str:
        return self.cur_policy.name_str

    def request(self, command: FSMCommand) -> None:
        self.state_cmd.skill_cmd = command

    def run(self) -> None:
        self.cur_policy.run()
        next_name = self.cur_policy.checkChange()
        if next_name != self.cur_policy.name:
            self.cur_policy.exit()
            self.cur_policy = self.states[next_name]
            self.cur_policy.enter()
            self._last_switch = time.monotonic()
            print(f"[R2V2FSM] switched to {self.cur_policy.name_str}")
