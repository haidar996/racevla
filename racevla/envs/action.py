"""Step 8: turn the policy's 12 outputs into joint-position targets for the Go1."""
import numpy as np

N_ACTIONS = 12
ACTION_SCALE = 0.5  # rad: largest offset from the home pose the policy can command (was 0.25; widened 2026-09-28)


def action_to_target(action: np.ndarray, home_joints: np.ndarray,
                     ctrl_lo: np.ndarray, ctrl_hi: np.ndarray,
                     scale: float = ACTION_SCALE) -> np.ndarray:
    """policy action in [-1, 1]^12  ->  desired joint angles (rad), inside the joint limits."""
    action = np.clip(action, -1.0, 1.0)             # 1) PPO exploration noise can exceed [-1, 1]
    target = home_joints + scale * action           # 2) offset from the standing pose
    return np.clip(target, ctrl_lo, ctrl_hi)        # 3) never ask for an impossible angle
