from abc import ABC, abstractmethod

from common.ctrlcomp import PolicyOutput, StateAndCmd
from common.utils import FSMStateName


class FSMState(ABC):
    def __init__(
        self,
        state_cmd: StateAndCmd,
        policy_output: PolicyOutput,
        name: FSMStateName,
        name_str: str,
    ):
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = name
        self.name_str = name_str

    def enter(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    def exit(self) -> None:
        pass

    @abstractmethod
    def checkChange(self) -> FSMStateName:
        raise NotImplementedError

