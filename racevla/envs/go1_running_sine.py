"""Go1RunningSineEnv: Phase 5 -- the walking reference of option B (go1_walking_sine.py) made speed-dependent so that it can reach a FLIGHT phase.
Differences from Go1WalkingSineEnv:
  - gait frequency and swing fraction depend on the commanded forward speed (faster steps, shorter ground contact; swing fraction > 0.5 means that both diagonal pairs are in the air
    together for a part of the cycle = flight);
  - half-stride follows the kinematic rule  S = v * stance_time / 2  (the stance foot must slide back at body speed), so it changes with the gait timing;
  - the foot target is converted to thigh and calf angles with the EXACT two-link inverse kinematics of the Go1 leg (thigh and calf both 0.213 m), not the linear Jacobian that was only valid for small strides;
  - lift height can grow with speed;
  - optional stance PUSH-OFF (gait['thrust']): the leg compresses then extends during stance, to give the body the upward speed that a flight phase needs.
Observation, residual, reward and fall rule are those of option B (49 numbers: 45 robot + 2 command + 2 clock)."""
import numpy as np

from racevla.envs.go1_walking import Go1WalkingEnv, MOVE_THRESHOLD, W_VZ
from racevla.envs.go1_walking_clock import FOOT_PHASE_OFFSETS
from racevla.envs.go1_walking_sine import Go1WalkingSineEnv, HALF_TRACK, STRIDE_MIN, GATE_RAMP

LEG_L = 0.213                      # m, thigh and calf length (identical)

# default gait law (what Option B used for walking: 2.5 Hz, swing 0.4, lift 8 cm, no change with speed)
DEFAULT_GAIT = dict(f0=2.5, f_slope=0.0, sf0=0.4, sf_slope=0.0, v_knee=0.6, sf_max=0.7, lift0=0.08, lift_slope=0.0, lift_max=0.14, stride_scale=1.0, thrust=0.0)


# --- running reward changes (on top of the walking reward) ---------------------------------------------------------
RUN_W_VZ = -0.5                    # vertical body velocity^2 penalty; the walking value (-2.0) would punish the bounce of a flight phase
W_FLIGHT = 0.3                     # bonus per step with all four feet off the floor, multiplied by the velocity-tracking kernel (hopping in place earns nothing)
FLIGHT_MIN_CMD = 1.0               # m/s: flight is only asked for at running commands


def leg_ik(x, z):
    """Exact thigh and calf angles that put the foot at (x forward, z up) in the hip frame (z < 0 below the hip). Returns (thigh, calf); calf is negative (knee bent backwards)."""
    r = np.clip(np.hypot(x, z), 1e-6, 2 * LEG_L - 1e-6)
    half = np.arccos(r / (2 * LEG_L)); mid = np.arctan2(-x, -z)
    return mid + half, -2.0 * half


def leg_fk(thigh, calf):
    """Foot position (x forward, z up) in the hip frame for given thigh and calf angles."""
    return -LEG_L * (np.sin(thigh) + np.sin(thigh + calf)), -LEG_L * (np.cos(thigh) + np.cos(thigh + calf))


class Go1RunningSineEnv(Go1WalkingSineEnv):
    def __init__(self, gait=None, **kwargs):
        super().__init__(**kwargs)
        self.gait = {**DEFAULT_GAIT, **(gait or {})}
        h = self.home_joints.reshape(4, 3); self._home_x, self._home_z = leg_fk(h[:, 1], h[:, 2])      # foot at home, per leg, in the hip frame
        self.swing_fraction = self.gait["sf0"]; self.frequency = self.gait["f0"]; self.lift = self.gait["lift0"]; self.thrust = self.gait["thrust"]

    # ---- gait law -----------------------------------------------------------------------------------------------
    def gait_for_speed(self, v):
        """(frequency Hz, swing fraction, lift m, push-off m) for a commanded forward speed. gait['table'] = dict(speeds, f, sf, lift, thrust) is interpolated linearly (flat outside its range); otherwise the
        simple law (f0 + f_slope * (v - v_knee) ...) is used with a constant push-off gait['thrust']."""
        g = self.gait
        if g.get("table"):
            T = g["table"]; return tuple(float(np.interp(v, T["speeds"], T[k])) for k in ("f", "sf", "lift", "thrust"))
        over = max(float(v) - g["v_knee"], 0.0)
        return g["f0"] + g["f_slope"] * over, min(g["sf0"] + g["sf_slope"] * over, g["sf_max"]), min(g["lift0"] + g["lift_slope"] * over, g["lift_max"]), g["thrust"]

    def _update_gait(self):
        self.frequency, self.swing_fraction, self.lift, self.thrust = self.gait_for_speed(max(self.command[0], 0.0))

    def desired_swing(self): return self.foot_phases() < self.swing_fraction

    def step(self, action):
        self._update_gait()
        if self._moving(): self.phase = (self.phase + self.frequency * 0.02) % 1.0
        return Go1WalkingEnv.step(self, action)                      # (skips the clock env's fixed 2.5 Hz advance)

    # ---- reference ----------------------------------------------------------------------------------------------
    def _foot_reference(self):
        vx, wz = self.command; v_side = np.array([vx + HALF_TRACK * wz, vx - HALF_TRACK * wz, vx + HALF_TRACK * wz, vx - HALF_TRACK * wz])
        stance_time = (1.0 - self.swing_fraction) / self.frequency
        S = self.gait["stride_scale"] * np.maximum(v_side, 0.0) * stance_time / 2.0 + STRIDE_MIN
        ph = self.foot_phases(); swing = ph < self.swing_fraction
        s = np.clip(ph / self.swing_fraction, 0.0, 1.0); u = np.clip((ph - self.swing_fraction) / (1.0 - self.swing_fraction), 0.0, 1.0)
        dx = np.where(swing, -S * np.cos(np.pi * s), S * np.cos(np.pi * u))
        # stance push-off: the leg first shortens (body sinks, spring loads) then extends fast at the end of stance (thrust, m of foot height); zero net height change over the stance
        dz = np.where(swing, self.lift * np.sin(np.pi * s), self.thrust * np.sin(2.0 * np.pi * u))
        return dx, dz

    def _reference_offset(self):
        self._update_gait()
        self.gate = float(np.clip(self.gate + (GATE_RAMP if self._moving() else -GATE_RAMP), 0.0, 1.0))
        if self.gate == 0.0: return 0.0
        dx, dz = self._foot_reference(); off = np.zeros(12); h = self.home_joints.reshape(4, 3)
        for i in range(4):
            th, ca = leg_ik(self._home_x[i] + dx[i], self._home_z[i] + dz[i])
            off[3 * i + 1], off[3 * i + 2] = th - h[i, 1], ca - h[i, 2]
        return self.gate * off

    # ---- reward ---------------------------------------------------------------------------------------------------
    def _compute_reward(self, action):
        _, t = Go1WalkingEnv._compute_reward(self, action)           # walking reward (also updates self.in_contact)
        total = t["r_raw"] + (RUN_W_VZ - W_VZ) * t["p_vz"]           # weaker vertical-velocity penalty
        flight = float(not self.in_contact.any()) if self.command[0] >= FLIGHT_MIN_CMD else 0.0
        t.update(flight=flight, r_flight=W_FLIGHT * flight * t["r_lin_track"]); total += t["r_flight"]; t["r_raw"] = float(total)
        return float(max(total, 0.0)), t
