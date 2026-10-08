"""Go1ScanGaitEnv: Phase 6, Skill B option A -- the scan-aware reference gait. The reference leg motion of option B (sine swing arc + backward stance slide) gets a TERRAIN-FOLLOWING term from a height map:
for every foot the reference target is lifted by the terrain height under (stance) or in front of (swing) the foot's reference position, relative to the ground under the body, so that a foot is placed on top of a step instead of into its riser.
 - stance foot: dz += h(foot reference position) - h(body)                                   (the foot stays on the ground, whatever its height)
 - swing foot:  dz += max over the next LOOK_AHEAD metres of h - h(body)                      (the foot rises BEFORE the step edge: a clearance envelope) + EDGE_MARGIN when a rise is ahead
Leg angles come from the EXACT two-link inverse kinematics (the linear model of the walking gait is only valid for small displacements, a 15 cm step is not small). Flags: terrain_ref (the terrain term), exact_ik (exact IK instead of the linear model).
The height map is the same function that produces the scan (ground_height_at); a camera would supply an estimate of it. The 18-number scan stays in the observation. With terrain_ref=False, exact_ik=False the class is Go1ScanWalkEnv."""
import numpy as np

from racevla.envs.go1_scan_walk import Go1ScanWalkEnv
from racevla.envs.go1_walking_clock import SWING_FRACTION
from racevla.envs.go1_running_sine import leg_ik, leg_fk
from racevla.envs.go1_walking_sine import _J_INV
from racevla.envs.go1_walking_sine import GATE_RAMP

HOME_FOOT_XY = np.array([[0.188, -0.127], [0.188, 0.127], [-0.188, -0.127], [-0.188, 0.127]])      # FR, FL, RR, RL: foot position in the body frame at the home pose (x forward, y left)
LOOK_AHEAD = (0.04, 0.08, 0.12)         # m ahead of the swing foot's reference position that are checked for a rise
EDGE_MARGIN = 0.02                      # m of extra clearance over a rise ahead
MAX_TERRAIN_TERM = 0.25                 # m, limit of the terrain term
CLEARANCE = 0.04                        # m, clearance over a rise ahead (mode 'max')
MAX_FOOT_RISE = 0.175                   # m, highest reference foot height above the home foot height (the calf joint limit is reached at about 0.195 m)


class Go1ScanGaitEnv(Go1ScanWalkEnv):
    def __init__(self, terrain_ref=True, exact_ik=True, ref_mode="max", **kwargs):
        super().__init__(**kwargs)
        self.terrain_ref, self.exact_ik, self.ref_mode = bool(terrain_ref), bool(exact_ik), ref_mode      # ref_mode 'additive' = first version (terrain term ON TOP of the swing arc), 'max' = foot height = max(arc, rise ahead + clearance), capped
        h = self.home_joints.reshape(4, 3); self._home_x, self._home_z = leg_fk(h[:, 1], h[:, 2])

    def _terrain_term(self, dx):
        """Per foot: how much the reference foot height is raised (m), from the height map (see module docstring)."""
        R = self.data.xmat[1].reshape(3, 3); yaw = float(np.arctan2(R[1, 0], R[0, 0])); c, s = np.cos(yaw), np.sin(yaw); bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        fx = HOME_FOOT_XY[:, 0] + dx; fy = HOME_FOOT_XY[:, 1]
        px, py = bx + c * fx - s * fy, by + s * fx + c * fy; g0 = float(self.ground_height_at(bx, by)); here = self.ground_height_at(px, py)
        ahead = np.max([self.ground_height_at(px + k * c, py + k * s) for k in LOOK_AHEAD], axis=0)
        swing = self.foot_phases() < getattr(self, "swing_fraction", SWING_FRACTION)
        rise_ahead = np.maximum(ahead - here, 0.0)
        g = np.where(swing, np.maximum(here, ahead) - g0 + np.where(rise_ahead > 0.01, EDGE_MARGIN, 0.0), here - g0)
        return np.clip(g, -MAX_TERRAIN_TERM, MAX_TERRAIN_TERM)

    def _terrain_dz_max(self, dx, dz_base):
        """Version 2: dz = (ground under the foot's reference position - ground under the body) + swing height, where the swing height is max(normal arc, (rise ahead + CLEARANCE) * arc shape) -- NOT their sum -- and the total is capped at MAX_FOOT_RISE."""
        R = self.data.xmat[1].reshape(3, 3); yaw = float(np.arctan2(R[1, 0], R[0, 0])); c, s = np.cos(yaw), np.sin(yaw); bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        fx = HOME_FOOT_XY[:, 0] + dx; fy = HOME_FOOT_XY[:, 1]; px, py = bx + c * fx - s * fy, by + s * fx + c * fy
        g0 = float(self.ground_height_at(bx, by)); here = self.ground_height_at(px, py); ahead = np.max([self.ground_height_at(px + k * c, py + k * s) for k in LOOK_AHEAD], axis=0)
        swing = self.foot_phases() < getattr(self, "swing_fraction", SWING_FRACTION); shape = dz_base / max(self.lift, 1e-6); rise = np.maximum(ahead - here, 0.0)
        swing_h = np.where(swing, np.maximum(dz_base, (rise + CLEARANCE) * shape), 0.0)
        return np.clip((here - g0) + swing_h, -MAX_TERRAIN_TERM, MAX_FOOT_RISE)

    def _reference_offset(self):
        self.gate = float(np.clip(self.gate + (GATE_RAMP if self._moving() else -GATE_RAMP), 0.0, 1.0))
        if self.gate == 0.0: return 0.0
        dx, dz = self._foot_reference()
        if self.terrain_ref: dz = (dz + self._terrain_term(dx)) if self.ref_mode == "additive" else self._terrain_dz_max(dx, dz)
        off = np.zeros(12); h = self.home_joints.reshape(4, 3)
        for i in range(4):
            if self.exact_ik:
                th, ca = leg_ik(self._home_x[i] + dx[i], self._home_z[i] + dz[i]); off[3 * i + 1], off[3 * i + 2] = th - h[i, 1], ca - h[i, 2]
            else: off[3 * i + 1], off[3 * i + 2] = _J_INV @ np.array([dx[i], dz[i]])
        return self.gate * off
