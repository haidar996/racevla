"""Step 9: our own joint-space PD controller (torque = kp*(target - q) - kd*qdot)."""
import mujoco
import numpy as np

KP = 100.0  # N*m/rad   same stiffness as the Menagerie built-in position actuators
KD = 0.5    # N*m*s/rad extra explicit damping (the joints also have passive damping of 1-2 in the XML)


def use_torque_actuators(model: mujoco.MjModel) -> np.ndarray:
    """Turn the 12 position actuators into plain torque motors. Returns the torque limits (12,)."""
    tau_max = model.actuator_forcerange[:, 1].copy()               # 23.7 (hip, thigh), 35.55 (calf)
    model.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED       # force = gain * ctrl ...
    model.actuator_gainprm[:, 0] = 1.0                             # ... with gain 1, so ctrl IS the torque
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE        # remove the built-in -kp*q spring
    model.actuator_biasprm[:] = 0.0
    model.actuator_ctrllimited[:] = 1
    model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1] = -tau_max, tau_max
    return tau_max


class PDController:
    def __init__(self, model: mujoco.MjModel, kp=KP, kd=KD):
        self.tau_max = use_torque_actuators(model)
        self.kp, self.kd = np.broadcast_to(kp, 12).astype(float), np.broadcast_to(kd, 12).astype(float)

    def torque(self, data: mujoco.MjData, target: np.ndarray) -> np.ndarray:
        q, qd = data.qpos[7:], data.qvel[6:]
        tau = self.kp * (target - q) - self.kd * qd
        return np.clip(tau, -self.tau_max, self.tau_max)

    def apply(self, data: mujoco.MjData, target: np.ndarray) -> None:
        """Call before EVERY mj_step (500 Hz), not once per policy step."""
        data.ctrl[:] = self.torque(data, target)
