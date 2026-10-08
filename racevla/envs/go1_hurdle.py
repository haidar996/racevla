"""Go1HurdleEnv: Phase 7, learned jump over a hurdle while running. The running environment (Go1RunningSineEnv: speed-dependent gait table with push-off, exact leg IK) on the terrain course with a HURDLE (a bar across the whole width at x = 5 m, 8 cm thick, 10-25 cm high), plus:
 * 13 actions = 12 joint corrections (scale 0.25 rad while running, 0.5 rad during a jump) + 1 TRIGGER: when it is above 0 (and a bar is within 1.6 m ahead, no jump is running and the 1 s cool-down is over) the SCRIPTED JUMP (racevla/envs/jump_primitive.py, parameters of the nearest starting speed from
   outputs/analysis/jump/feasibility.json) takes over the reference leg motion until it is done (crouch / thrust / flight / landing / hold), then the running gait reference comes back.
 * observation 75 = 49 (45 robot, 2 command, 2 clock) + 24 HEIGHT SCAN + 2 JUMP numbers. The scan looks far ahead: rows 0.2 ... 1.6 m ahead (8, each row = the HIGHEST ground in a 0.2 m cell) x columns -0.15 / 0 / +0.15 m (3), heading-aligned, heights relative to the ground under the base, x4, noise 0.5 cm (training).
   The jump numbers: (1 while a jump runs else 0, phase code: crouch 0.25, thrust 0.5, flight 0.75, landing 1.0, none 0).
 * episodes (400 steps = 8 s): the robot starts at x = 0 heading +x from rest (a start with an initial speed confused the running policy: 3 of 12 flat episodes stalled; from rest 11 of 12 succeed) (command ~ U(1.3, 1.7) m/s, heading hold on the turn command). 20 % of the episodes have no hurdle (the robot must NOT jump for nothing: it would lose speed).
   Success = 2 s (100 steps) after the robot passed the bar's zone (x > 5.45 m; at x = 8.95 m at the latest, the end of the course) it is upright (tilt < 35 deg) and its forward speed is >= 50 % of the command (truncation); the episode lasts at most 450 steps. The episode ends in failure when the robot falls or when the trunk, a hip or a thigh touches the bar itself (floor contact at x from 4.96 to 5.12 m); lower legs (calf, foot) may brush it.
 * reward = the walking reward (tracking, smoothness; the vertical-speed penalty is cut to -0.1 because a jump needs vertical speed; NO flight bonus) + 10 once for crossing the hurdle (x > 5.45 m) - 5 for hitting it.
 Curriculum: heights HURDLE_LEVELS[0 .. top]; set_top(index) / pop_stats() like the terrain environments (stats keyed ('hurdle', height), flat episodes ('rough', 0.0))."""
import json
import numpy as np
from gymnasium import spaces

from racevla.envs.go1_running_sine import Go1RunningSineEnv
from racevla.envs.go1_walking import Go1WalkingEnv, W_VZ
from racevla.envs.jump_primitive import JumpPrimitive
from racevla.envs.terrain import TerrainMixin, HURDLE_LEVELS, X_HURDLE, HURDLE_THICK
from racevla.robots.go1 import ROOT

SCAN_DX, SCAN_DY, SCAN_SCALE = np.arange(0.2, 1.61, 0.2), np.array([-0.15, 0.0, 0.15]), 4.0
SCAN_DIM = len(SCAN_DX) * len(SCAN_DY)          # 24
OBS_DIM = 49 + SCAN_DIM + 2                     # 75
ACT_DIM = 13
RUN_SCALE, JUMP_SCALE = 0.25, 0.5
MAX_STEPS, SUCCESS_AFTER, X_EDGE = 450, 100, 8.95        # success: 100 steps (2 s) after crossing the bar's zone (or at x = 8.95, the end of the course) the robot is upright and runs at >= 50 % of the commanded speed
HIT_ZONE = (X_HURDLE - 0.04, X_HURDLE + HURDLE_THICK + 0.04)            # a 'hit' = a non-foot part touches the floor INSIDE the bar's own footprint (a calf brushing the flat ground elsewhere is normal while running and must not count)
CROSSED_X = X_HURDLE + HURDLE_THICK + 0.37
RUN_W_VZ = -0.1
COOLDOWN = 50
TRIGGER_RANGE = 1.6                             # m: the trigger is ignored unless the bar is at most this far ahead (the far end of the scan)
CROSS_BONUS, HIT_PENALTY = 10.0, -5.0
PHASE_CODE = {"crouch": 0.25, "thrust": 0.5, "flight": 0.75, "land": 1.0, "hold": 1.0}


class Go1HurdleEnv(TerrainMixin, Go1RunningSineEnv):
    def __init__(self, speed_range=(1.3, 1.7), scan_noise=0.005, p_flat=0.2, **kwargs):
        table = json.load(open(ROOT / "models/running_gait_table.json"))
        super().__init__(push=False, gait={"table": table}, **kwargs)
        self.speed_range, self.scan_noise, self.p_flat = tuple(speed_range), float(scan_noise), float(p_flat)
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32); self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        best = json.load(open(ROOT / "models/jump_primitive_hurdle.json"))["best"][0]["params"]; jp = {k: v for k, v in best.items() if k != "d"}           # the jump found by the crossing search (almost no crouch, strong backward push)
        self.jump_params = {"1.5": {"params": jp}, "2.0": {"params": jp}, "2.5": {"params": jp}}; self.crossed_step = None
        gx, gy = np.meshgrid(SCAN_DX, SCAN_DY, indexing="ij"); self._offsets = np.stack([gx.ravel(), gy.ravel()], axis=1)
        m = self.model; self.body_geoms = {g for g in range(m.ngeom) if (lambda n: n == 'trunk' or n.endswith('hip') or n.endswith('thigh'))(m.body(m.geom_bodyid[g]).name)}          # a 'hit' = trunk, hip or thigh touches the bar; the lower legs (calf, foot) may brush it
        self.top, self.stats, self.jump, self.cooldown, self.current, self.crossed, self.level_h = 0, {}, None, 0, ("rough", 0.0), False, 0.0

    # ---- curriculum ---------------------------------------------------------------------------------------------
    def set_top(self, top): self.top = int(min(max(top, 0), len(HURDLE_LEVELS) - 1))

    def pop_stats(self):
        s, self.stats = self.stats, {}; return s

    def _draw_terrain(self, rng):
        if rng.random() < self.p_flat: return ("rough", 0.0)
        idx = self.top if (rng.random() < 0.5 or self.top == 0) else int(rng.integers(0, self.top + 1)); return ("hurdle", HURDLE_LEVELS[idx])

    # ---- command / observation -----------------------------------------------------------------------------------
    def _sample_command(self): self.command = np.array([self.np_random.uniform(*self.speed_range), 0.0])

    def _hold(self):
        R = self.data.xmat[1].reshape(3, 3); self.command[1] = float(np.clip(-2.0 * np.arctan2(R[1, 0], R[0, 0]), -0.5, 0.5))

    def scan(self):
        R = self.data.xmat[1].reshape(3, 3); yaw = float(np.arctan2(R[1, 0], R[0, 0])); c, s = np.cos(yaw), np.sin(yaw); bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        sub = np.linspace(-0.1, 0.1, 9)[:, None]                                       # each row is a 0.2 m cell: the scan reports the HIGHEST ground in the cell (max pooling), so that an 8 cm bar between two rows is not missed
        ahead = self._offsets[None, :, 0] + sub; side = np.broadcast_to(self._offsets[None, :, 1], ahead.shape)
        px = bx + c * ahead - s * side; py = by + s * ahead + c * side
        rel = self.ground_height_at(px, py).max(axis=0) - float(self.ground_height_at(bx, by))
        if self.scan_noise > 0: rel = rel + self.np_random.normal(0.0, self.scan_noise, rel.shape)
        return (SCAN_SCALE * rel).astype(np.float32)

    def _full_obs(self, obs):
        obs = np.array(obs, np.float32); obs[45:47] = self.command
        jump = np.array([1.0 if self.jump is not None else 0.0, PHASE_CODE.get(self.jump.phase, 0.0) if self.jump is not None else 0.0], np.float32)
        return np.concatenate([obs, self.scan(), jump]).astype(np.float32)

    # ---- reset / step -------------------------------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        options = dict(options or {}); rng = np.random.default_rng(seed) if seed is not None else (self.np_random if self.np_random is not None else np.random.default_rng())
        if options.get("terrain") is None: options["terrain"] = self._draw_terrain(rng)
        options.setdefault("yaw", 0.0)
        self.jump, self.cooldown, self.crossed, self.crossed_step, self.action_scale = None, 0, False, None, RUN_SCALE
        obs, info = super().reset(seed=seed, options=options); self.current = tuple(options["terrain"]); self.level_h = 0.0 if self.current[0] == "rough" else float(self.current[1])
        self._hold(); info["terrain"] = self.current; return self._full_obs(obs), info

    def _reference_offset(self):
        if self.jump is not None:
            tgt = self.jump.target(self._foot_contacts()); return (tgt - self.home_joints.reshape(4, 3)).ravel()          # absolute jump targets (offsets from home)
        return super()._reference_offset()

    def _compute_reward(self, action):
        _, t = Go1WalkingEnv._compute_reward(self, action); total = t["r_raw"] + (RUN_W_VZ - W_VZ) * t["p_vz"]; t["r_raw"] = float(total); return float(max(total, 0.0)), t

    def _hit_hurdle(self):
        d = self.data
        for c in d.contact[:d.ncon]:
            if c.dist >= 0.0 or self.floor_id not in (c.geom1, c.geom2): continue
            other = c.geom2 if c.geom1 == self.floor_id else c.geom1
            if other in self.body_geoms and HIT_ZONE[0] <= c.pos[0] <= HIT_ZONE[1] and self.level_h > 0.0: return True
        return False

    def step(self, action):
        a = np.asarray(action, np.float64).ravel()
        bar_ahead = self.level_h > 0.0 and -0.1 <= X_HURDLE - float(self.data.qpos[0]) <= TRIGGER_RANGE              # the trigger only counts while a bar is within the scan's range (otherwise random triggers on the empty course wreck the run)
        if self.jump is None and self.cooldown <= 0 and a[12] > 0.0 and bar_ahead:
            p = min((1.5, 2.0, 2.5), key=lambda v: abs(v - float(self.command[0]))); self.jump = JumpPrimitive(self.jump_params[str(p)]["params"], self.home_joints)
        self.action_scale = JUMP_SCALE if self.jump is not None else RUN_SCALE
        obs, reward, terminated, truncated, info = super().step(np.clip(a[:12], -1.0, 1.0).astype(np.float32))
        if self.jump is not None and self.jump.done: self.jump, self.cooldown = None, COOLDOWN
        elif self.cooldown > 0: self.cooldown -= 1
        hit = (not terminated) and self._hit_hurdle()
        if hit: terminated, reward = True, reward + HIT_PENALTY; info["termination_reason"] = "hit_hurdle"
        x = float(self.data.qpos[0])
        if (not terminated) and self.crossed_step is None and x > CROSSED_X:
            self.crossed_step = self.step_count                                                  # (also for the flat episodes: the same position counts as 'passed')
            if self.level_h > 0.0: self.crossed = True; reward += CROSS_BONUS
        R = self.data.xmat[1].reshape(3, 3); tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))); v_fwd = float((R.T @ self.data.qvel[:3])[0])
        ready = self.crossed_step is not None and (self.step_count - self.crossed_step >= SUCCESS_AFTER or x >= X_EDGE)
        success = (not terminated) and ready and tilt < 35.0 and v_fwd >= 0.5 * float(self.command[0])
        if (not terminated) and (ready or x >= X_EDGE or self.step_count >= MAX_STEPS): truncated = True
        if not (terminated or truncated): self._hold()
        if terminated or truncated:
            s = self.stats.setdefault(self.current, [0, 0]); s[0] += 1; s[1] += int(success); info["success"] = bool(success); info["terrain"] = self.current
        info.update(jump_active=float(self.jump is not None), crossed=float(self.crossed))
        return self._full_obs(obs), reward, terminated, truncated, info
