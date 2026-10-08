"""Go1WalkingSineEnv: Phase 4, option B -- a built-in sine-shaped leg motion driven by the gait clock, plus a small learned residual.
Per foot, the clock phase gives a reference foot path relative to the home pose: during swing the foot moves forward and lifts along a half-sine arc (peak LIFT_HEIGHT), during stance it slides backward
along the floor (that is what pushes the body forward); stride length follows the commanded speed (and differs between the left and right side for turning). The path is converted to thigh and calf angles
with the measured linear leg kinematics of the Go1 and ADDED to the home pose; the policy output (scaled by RESIDUAL_SCALE) is added on top. So the metronome DRIVES the joints here, while in option A
(go1_walking_clock.py) it only guides through the observation and the reward.
Observation: the same 49 numbers as option A (robot, command, clock). Reward: Go1WalkingEnv's (tracking, slip, smoothness, capped air-time bonus) -- NOT option A's schedule and swing-height terms."""
import numpy as np

from racevla.envs.go1_walking import Go1WalkingEnv
from racevla.envs.go1_walking_clock import Go1WalkingClockEnv, GAIT_FREQUENCY, SWING_FRACTION, FOOT_PHASE_OFFSETS

LIFT_HEIGHT = 0.08          # m, peak of the swing arc (same as option A's target)
STRIDE_GAIN = 0.12          # m of half-stride per m/s of commanded foot speed  (stance lasts 0.24 s: speed = 2*S/0.24)
STRIDE_MIN = 0.01           # m, half-stride kept even at zero forward speed (so turning in place still steps)
HALF_TRACK = 0.127          # m, distance of the legs from the body centre line (turning = left/right speed difference)
RESIDUAL_SCALE = 0.25       # rad per unit of action (option A and the standing runs used 0.5)
GATE_RAMP = 0.05            # the reference fades in/out over 20 steps when the robot starts/stops walking
# Measured on the model at the home pose (thigh 0.9, calf -1.8): foot motion (dx forward, dz up) in m per rad of [thigh, calf]
_J = np.array([[-0.26, -0.157], [0.04, -0.143]])
_J_INV = np.linalg.inv(_J)


class Go1WalkingSineEnv(Go1WalkingClockEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_scale = RESIDUAL_SCALE
        self.gate = 0.0
        self.lift = LIFT_HEIGHT                              # m, peak of the swing arc (an attribute so that a test can raise it; default = LIFT_HEIGHT)

    def reset(self, seed=None, options=None):
        self.gate = 0.0
        return super().reset(seed=seed, options=options)

    def _foot_reference(self):
        """Reference foot displacement from the home pose, per foot: (dx forward, dz up) in metres."""
        vx, wz = self.command; v_side = np.array([vx + HALF_TRACK * wz, vx - HALF_TRACK * wz, vx + HALF_TRACK * wz, vx - HALF_TRACK * wz])   # FR FL RR RL (right side faster for a left turn)
        S = STRIDE_GAIN * np.maximum(v_side, 0.0) + STRIDE_MIN
        ph = self.foot_phases(); swing = ph < SWING_FRACTION
        s = np.clip(ph / SWING_FRACTION, 0.0, 1.0); u = np.clip((ph - SWING_FRACTION) / (1.0 - SWING_FRACTION), 0.0, 1.0)
        dx = np.where(swing, -S * np.cos(np.pi * s), S * np.cos(np.pi * u))           # swing: back -> front, stance: front -> back
        dz = np.where(swing, self.lift * np.sin(np.pi * s), 0.0)                      # half-sine arc during swing
        return dx, dz

    def _reference_offset(self):
        self.gate = float(np.clip(self.gate + (GATE_RAMP if self._moving() else -GATE_RAMP), 0.0, 1.0))
        if self.gate == 0.0: return 0.0
        dx, dz = self._foot_reference(); off = np.zeros(12)
        for i in range(4):
            off[3 * i + 1], off[3 * i + 2] = _J_INV @ np.array([dx[i], dz[i]])       # thigh and calf angles that move the foot there (hips stay at home)
        return self.gate * off

    def _compute_reward(self, action):
        return Go1WalkingEnv._compute_reward(self, action)                           # option A's schedule / swing-height terms are NOT used here
