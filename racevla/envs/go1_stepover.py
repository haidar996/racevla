"""Go1StepOverEnv: Phase 7, hurdle crossing by STEPPING OVER, not jumping: the hurdle environment (Go1HurdleEnv: bar across the course at x = 5 m, far max-pooled height scan in the observation, same success / hit rules) with a high-stepping reference gait and 12 actions (no trigger).
Reference: the speed-dependent running gait (Go1RunningSineEnv) plus a terrain-following term from the height map, as in racevla/envs/go1_scan_gait.py but with a longer look-ahead (the bar is thin and the swing foot travels ~0.2-0.3 m): for every foot the reference foot target is raised by
  stance foot: h(foot reference position) - h(under the body)            swing foot: max over the next LOOK_AHEAD of h - h(under the body) + CLEARANCE,  total swing height = max(normal arc, rise ahead + CLEARANCE), capped at MAX_FOOT_RISE
so each foot is lifted just before the bar and put down behind it; the body keeps running (no crouch, no flight). The learned corrections (scale 0.25) go on top."""
import numpy as np
from gymnasium import spaces

from racevla.envs.go1_hurdle import Go1HurdleEnv
from racevla.envs.go1_running_sine import leg_ik, leg_fk
from racevla.envs.go1_scan_gait import HOME_FOOT_XY, MAX_TERRAIN_TERM
from racevla.envs.go1_walking_sine import GATE_RAMP

LOOK_AHEAD = (0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32)      # m ahead of the swing foot's reference position that are checked for a rise
CLEARANCE = 0.06                                        # m over the highest ground ahead (the real foot lags the reference: the first version with 4 cm hit a 10 cm bar with the calf while the foot was 12 cm up)
MAX_FOOT_RISE = 0.175                                   # m above the home foot height (the calf joint limit is reached at about 0.195 m)


class Go1StepOverEnv(Go1HurdleEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)
        h = self.home_joints.reshape(4, 3); self._home_x, self._home_z = leg_fk(h[:, 1], h[:, 2])

    def step(self, action): return super().step(np.append(np.asarray(action, np.float64).ravel()[:12], -1.0))        # the trigger is never used

    def _stepover_dz(self, dx, dz_base):
        R = self.data.xmat[1].reshape(3, 3); yaw = float(np.arctan2(R[1, 0], R[0, 0])); c, s = np.cos(yaw), np.sin(yaw); bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        fx = HOME_FOOT_XY[:, 0] + dx; fy = HOME_FOOT_XY[:, 1]; px, py = bx + c * fx - s * fy, by + s * fx + c * fy
        g0 = float(self.ground_height_at(bx, by)); here = self.ground_height_at(px, py); ahead = np.max([self.ground_height_at(px + k * c, py + k * s) for k in LOOK_AHEAD], axis=0)
        ph = self.foot_phases(); swing = ph < self.swing_fraction; sph = np.clip(ph / self.swing_fraction, 0.0, 1.0); plateau = np.clip(np.sin(np.pi * sph) / 0.7, 0.0, 1.0)      # reaches full height early in the swing and holds it (the arc peaks only in the middle)
        rise = np.maximum(ahead - here, 0.0); swing_h = np.where(swing, np.maximum(dz_base, (rise + CLEARANCE) * plateau), 0.0)
        return np.clip((here - g0) + swing_h, -MAX_TERRAIN_TERM, MAX_FOOT_RISE)

    def _reference_offset(self):
        if self.jump is not None: return super()._reference_offset()
        self._update_gait(); self.gate = float(np.clip(self.gate + (GATE_RAMP if self._moving() else -GATE_RAMP), 0.0, 1.0))
        if self.gate == 0.0: return 0.0
        dx, dz = self._foot_reference(); dz = self._stepover_dz(dx, dz); off = np.zeros(12); h = self.home_joints.reshape(4, 3)
        for i in range(4):
            th, ca = leg_ik(self._home_x[i] + dx[i], self._home_z[i] + dz[i]); off[3 * i + 1], off[3 * i + 2] = th - h[i, 1], ca - h[i, 2]
        return self.gate * off
