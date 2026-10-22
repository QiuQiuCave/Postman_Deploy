"""Single native 1461-input Reach actor with an explicit final protocol."""

import numpy as np
from pathlib import Path
from r2v2_deploy.configuration import digest
from .reach_policy import ReachPolicy, TERM_DIMS

TASK_ID = "R2V2-ContactReachDelivery-v1-28DoF"
TASK_SCOPE = "fixed_scene_contact_reach_delivery_v1"


def model_identity():
    profile = Path(__file__).resolve().parents[1] / "compatibility/runtime_profile.json"
    return dict(
        task_id=TASK_ID,
        task_scope=TASK_SCOPE,
        condition_contract="left_hand_reference_progress_v1",
        delivery_adapter_sha256=digest(__file__),
        delivery_core_profile_sha256=digest(profile),
    )


class ExitPlacementReachPolicy(ReachPolicy):
    INPUT_DIM = 1461
    INPUT_NAMES = (*tuple(TERM_DIMS), "left_finger_closure_progress")

    def __init__(self, *args, **kwargs):
        from r2v2_deploy.compatibility.release import validate_runtime

        validate_runtime()
        self.closure_provider = lambda: 0.0
        super().__init__(*args, **kwargs)
        if any(self.metadata.get(k) != v for k, v in model_identity().items()):
            raise ValueError("Unknown or stale native Reach model identity")

    def observe(self, data, advance_reference=True, *, update_history=None):
        body = super().observe(data, advance_reference, update_history=update_history)
        condition = float(self.closure_provider())
        if not np.isfinite(condition) or not 0 <= condition <= 1:
            raise ValueError("Invalid real finger reference progress")
        self.last_observation = np.r_[body, np.float32(condition)].astype(np.float32)
        return self.last_observation.copy()
