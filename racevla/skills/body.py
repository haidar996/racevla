"""SwitchBody: ONE simulated Go1 whose 'skill settings' can be changed while the episode runs (Phase 7, skill switching).
Why: each trained skill expects its own simulator settings, not only its own network. Stand = home pose + 0.5 rad x action, no reference leg motion. Walk = the walking sine reference (fixed 2.5 Hz, 8 cm lift, linear leg
kinematics) + 0.25 rad x action. Run = the speed-dependent running reference (gait table, exact leg kinematics, push-off) + 0.25 rad x action. set_mode() switches between these three and nothing else changes: the robot, its joint
angles, its velocity and the gait clock simply carry on, so a switch is a real hand-over. TerrainSwitchBody (bottom of the file) adds the terrain scene, the height scan and the two terrain skills' modes.
Commands are NOT resampled by the environment: the supervisor sets body.command = [vx, wz] itself. The fall rule is the walking one (tilt > 60 deg or trunk touching the floor), the same for every mode, so that falls are comparable."""
import numpy as np

from racevla.envs.go1_walking import Go1WalkingEnv
from racevla.envs.go1_walking_clock import GAIT_FREQUENCY, SWING_FRACTION
from racevla.envs.go1_walking_sine import Go1WalkingSineEnv, LIFT_HEIGHT, RESIDUAL_SCALE, GATE_RAMP, _J_INV
from racevla.envs.go1_running_sine import Go1RunningSineEnv, leg_ik, leg_fk
from racevla.envs.go1_hurdle import Go1HurdleEnv, SCAN_DX, SCAN_DY
from racevla.envs.go1_stepover import Go1StepOverEnv
from racevla.envs.terrain import TerrainMixin
from racevla.envs.race import RACE_XML, NCOL_RACE, X1_RACE
from racevla.envs.action import ACTION_SCALE

MODES = ("stand", "walk", "run", "terrain", "stepover")      # terrain = the walking reference (its own trained network); stepover = the running reference + a terrain-following foot lift


class SwitchBody(Go1RunningSineEnv):
    def __init__(self, gait_table, **kwargs):
        super().__init__(push=False, gait={"table": gait_table}, **kwargs)
        self.mode = "walk"; self.set_mode("stand")

    def set_mode(self, mode):
        assert mode in MODES, mode
        self.mode = mode
        self.action_scale = ACTION_SCALE if mode == "stand" else RESIDUAL_SCALE        # 0.5 rad for standing, 0.25 rad residual for walking / running
        if mode in ("walk", "terrain"):                                                 # the walking reference: fixed gait, no speed dependence
            self.frequency, self.swing_fraction, self.lift, self.thrust = GAIT_FREQUENCY, SWING_FRACTION, LIFT_HEIGHT, 0.0
        if mode == "stand": self.gate = 0.0                                             # no reference leg motion while standing

    def _sample_command(self): pass                                                     # the supervisor owns the command

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.command = np.zeros(2); self.gate = 0.0; return obs, info

    def step(self, action):
        if self.mode in ("run", "stepover"): self._update_gait()                                      # frequency / swing / lift / push-off from the gait table
        if self._moving(): self.phase = (self.phase + self.frequency * 0.02) % 1.0     # the clock only ticks while asked to move
        return Go1WalkingEnv.step(self, action)

    def _reference_offset(self):
        if self.mode == "stand": self.gate = 0.0; return 0.0
        if self.mode == "run": return Go1RunningSineEnv._reference_offset(self)
        self.gate = float(np.clip(self.gate + (GATE_RAMP if self._moving() else -GATE_RAMP), 0.0, 1.0))      # walk: exactly Go1WalkingSineEnv's reference
        if self.gate == 0.0: return 0.0
        dx, dz = Go1WalkingSineEnv._foot_reference(self); off = np.zeros(12)
        for i in range(4): off[3 * i + 1], off[3 * i + 2] = _J_INV @ np.array([dx[i], dz[i]])
        return self.gate * off

    def tilt_deg(self): return float(np.degrees(np.arccos(np.clip(self.data.xmat[1].reshape(3, 3)[2, 2], -1.0, 1.0))))


class _SkillExtras:
    """What the two terrain skills need on top of SwitchBody: heading_hold() (they were trained with turn command = -2 x yaw error, clipped to +-0.5), scan() (the 24-number far height scan of the step-over skill, same function as
    Go1HurdleEnv.scan, noise 0.5 cm as in training), hit_hurdle() and the step-over reference (Go1StepOverEnv's foot-lift rule, used in mode 'stepover'). Needs ground_height_at(x, y)."""
    def _init_extras(self, scan_noise):
        self.scan_noise = float(scan_noise); gx, gy = np.meshgrid(SCAN_DX, SCAN_DY, indexing="ij"); self._offsets = np.stack([gx.ravel(), gy.ravel()], axis=1)
        m = self.model; self.body_geoms = {g for g in range(m.ngeom) if (lambda n: n == "trunk" or n.endswith("hip") or n.endswith("thigh"))(m.body(m.geom_bodyid[g]).name)}      # a 'hit' = trunk, hip or thigh touches the bar
        self.level_h = 0.0

    def heading_hold(self):
        R = self.data.xmat[1].reshape(3, 3); return float(np.clip(-2.0 * np.arctan2(R[1, 0], R[0, 0]), -0.5, 0.5))

    def scan(self): return Go1HurdleEnv.scan(self)

    def hit_hurdle(self): return Go1HurdleEnv._hit_hurdle(self)

    def _reference_offset(self):
        if self.mode != "stepover": return super()._reference_offset()
        self._update_gait(); self.gate = float(np.clip(self.gate + (GATE_RAMP if self._moving() else -GATE_RAMP), 0.0, 1.0))
        if self.gate == 0.0: return 0.0
        dx, dz = self._foot_reference(); dz = Go1StepOverEnv._stepover_dz(self, dx, dz); off = np.zeros(12); h = self.home_joints.reshape(4, 3)
        for i in range(4):
            th, ca = leg_ik(self._home_x[i] + dx[i], self._home_z[i] + dz[i]); off[3 * i + 1], off[3 * i + 2] = th - h[i, 1], ca - h[i, 2]
        return self.gate * off


class FlatSwitchBody(_SkillExtras, SwitchBody):
    """All five modes on the INFINITE flat floor (no height field): the scan reads zeros. For hand-over tests that need room (running covers 10 m in 5 s)."""
    def __init__(self, gait_table, scan_noise=0.005, **kwargs):
        super().__init__(gait_table, **kwargs); self._init_extras(scan_noise)

    def ground_height_at(self, x, y): return np.zeros(np.shape(x))


class TerrainSwitchBody(TerrainMixin, _SkillExtras, SwitchBody):
    """SwitchBody on the terrain course (racevla/envs/terrain.py, 10 m x 3 m): reset(options={'terrain': (kind, level), 'terrain_seed': s, 'yaw': 0.0}); the flat course is ('rough', 0.0)."""
    def __init__(self, gait_table, scan_noise=0.005, **kwargs):
        super().__init__(gait_table, **kwargs); self._init_extras(scan_noise)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options); kind, level = self.terrain; self.level_h = float(level) if kind == "hurdle" else 0.0; return obs, info


class RaceSwitchBody(TerrainSwitchBody):
    """TerrainSwitchBody on the 46 m race course (racevla/envs/race.py). reset(options={'height_field': (H, z0, ('race', 0.0)), 'yaw': 0.0})."""
    scene_path, ncol, x1 = RACE_XML, NCOL_RACE, X1_RACE
