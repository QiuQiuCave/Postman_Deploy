from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName
from common.ctrlcomp import StateAndCmd, PolicyOutput
from policy.loco_new.LocoNew import LocoNew
import os


class LocoNewOnnx(LocoNew):
    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput):
        super().__init__(
            state_cmd,
            policy_output,
            config_dir=os.path.dirname(os.path.abspath(__file__)),
            config_name="LocoNewOnnx.yaml",
            state_name=FSMStateName.LOCO_NEW_ONNX,
            name_str="loco_new_onnx_mode",
        )
