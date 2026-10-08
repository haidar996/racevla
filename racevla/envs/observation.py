"""Step 7: the observation vector the policy receives (45 numbers)."""
import mujoco
import numpy as np

N_JOINTS = 12
OBS_DIM = 12 + 12 + 3 + 3 + 3 + 12  # = 45

# Slices into the observation vector, so other code never uses magic indices.
OBS_SLICES = {
    "joint_pos":   slice(0, 12),   # joint angle minus home pose
    "joint_vel":   slice(12, 24),
    "gravity":     slice(24, 27),  # world "down" expressed in the body frame
    "ang_vel":     slice(27, 30),  # body frame
    "lin_vel":     slice(30, 33),  # body frame
    "prev_action": slice(33, 45),
}


def build_obs(data: mujoco.MjData, home_joints: np.ndarray, prev_action: np.ndarray) -> np.ndarray:
    """Build the 45-D observation from the current simulator state.

    data.qpos = [x y z, qw qx qy qz, 12 joint angles]
    data.qvel = [vx vy vz (world frame), wx wy wz (body frame), 12 joint velocities]
    """
    R = data.xmat[1].reshape(3, 3)  # trunk rotation matrix: body frame -> world frame

    joint_pos = data.qpos[7:] - home_joints
    joint_vel = data.qvel[6:]
    gravity_body = R.T @ np.array([0.0, 0.0, -1.0])  # R.T maps world -> body
    ang_vel_body = data.qvel[3:6]                     # MuJoCo already gives this in the body frame
    lin_vel_body = R.T @ data.qvel[0:3]               # world-frame velocity rotated into body frame

    return np.concatenate([joint_pos, joint_vel, gravity_body, ang_vel_body, lin_vel_body, prev_action]).astype(np.float32)
