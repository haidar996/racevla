"""Observation normalization for the policy network. Kept separate from Go1StandingEnv itself, so the raw
environment (and all its existing tests) is untouched -- this only wraps the numbers handed to the network."""
import numpy as np
import gymnasium as gym
from racevla.envs.action import ACTION_SCALE
from racevla.envs.observation import OBS_SLICES

# Fixed, physically-motivated scales -- NOT fit from data, so every seed and every algorithm normalizes
# identically. Dividing each observation slice by its scale keeps most real values within roughly [-1, 1].
JOINT_POS_SCALE = ACTION_SCALE                  # 0.5 rad: the largest offset the policy itself can command
JOINT_VEL_SCALE = 10.0                          # rad/s: observed max ~6.7 rad/s in practice, with margin
GRAVITY_SCALE = 1.0                             # already a unit vector, no change
ANG_VEL_SCALE = 2.0                             # rad/s: observed max ~1.8 rad/s, with margin
LIN_VEL_SCALE = 1.5                             # m/s: above the hardest push (1.2 m/s) plus reset noise
PREV_ACTION_SCALE = 1.0                         # already bounded to [-1, 1] by the action space

_SCALES = {
    "joint_pos": JOINT_POS_SCALE, "joint_vel": JOINT_VEL_SCALE, "gravity": GRAVITY_SCALE,
    "ang_vel": ANG_VEL_SCALE, "lin_vel": LIN_VEL_SCALE, "prev_action": PREV_ACTION_SCALE,
}


def _build_scale_vector():
    from racevla.envs.observation import OBS_DIM
    scale = np.ones(OBS_DIM, dtype=np.float32)
    for name, sl in OBS_SLICES.items():
        scale[sl] = _SCALES[name]
    return scale


OBS_SCALE = _build_scale_vector()   # (45,), one divisor per observation number


class FixedObsNormalize(gym.ObservationWrapper):
    """Divides each observation by a fixed physical scale (see OBS_SCALE), so the network sees every
    quantity in roughly the same range instead of joint velocities dwarfing linear velocity, etc."""

    def observation(self, obs):
        if obs.shape[-1] == OBS_SCALE.shape[0]: return (obs / OBS_SCALE).astype(np.float32)
        extra = np.ones(obs.shape[-1] - OBS_SCALE.shape[0], dtype=np.float32)       # walking: the appended command numbers (|vx| <= 0.6, |wz| <= 0.5) are already O(1)
        return (obs / np.concatenate([OBS_SCALE, extra])).astype(np.float32)
