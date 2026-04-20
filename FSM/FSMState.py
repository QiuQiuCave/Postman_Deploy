from common.path_config import PROJECT_ROOT

from common.utils import FSMStateName    

class FSMState:
    # Override to True in policies that need the MuJoCo transport_box body
    # teleported into the grasp region on entry. Deploy loops read this flag
    # so box-handling logic doesn't need to enumerate every such policy.
    needs_transport_box = False

    def __init__(self):
        self.name = FSMStateName.INVALID
        self.name_str = "invalid"
        self.control_dt = 0.02
    def enter(self):
        raise NotImplementedError("enter() function must be implement!")
    
    def run(self):
        raise NotImplementedError("run() function must be implement!")
    
    def exit(self):
        raise NotImplementedError("exit() function must be implement!")
    
    def checkChange(self):
        # joystick callback
        raise NotImplementedError("checkChange() function must be implement!")
        