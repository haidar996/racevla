"""Go1TerrainWalkEnv: Phase 6 blind terrain walking (version 1) = option-B walking environment (Go1WalkingSineEnv: sine reference + learned residual, 49-D observation) on the terrain course of racevla/envs/terrain.py.
Every episode: a random terrain type and level (curriculum: per type, levels up to TOP[type] are unlocked), the robot starts at x = 0 heading along +x; forward command vx ~ U(vx_range) (resampled every 200 steps), turn command = heading hold
(-2 x yaw error, clipped to +-0.5 rad/s, refreshed every step so the policy sees it), no stand commands, no pushes. Reward and fall rule = the walking ones. The episode ENDS WITH SUCCESS (truncation, the value is bootstrapped) when x reaches X_GOAL.
Terrain type mix: flat 20 %, rough 20 %, slope_up 20 %, slope_down 10 %, stairs_up 15 %, stairs_down 15 % ('flat' = rough with level 0). Within a type: 50 % the top unlocked level, 50 % uniform over the unlocked levels.
set_top(dict) sets the unlocked levels (index into terrain.LEVELS[type]), pop_stats() returns and clears {(type, level): [episodes, successes]}."""
import numpy as np

from racevla.envs.go1_walking_sine import Go1WalkingSineEnv
from racevla.envs.terrain import TerrainMixin, LEVELS, X_GOAL

TYPE_PROBS = {"flat": 0.20, "rough": 0.20, "slope_up": 0.20, "slope_down": 0.10, "stairs_up": 0.15, "stairs_down": 0.15}
CAP = {"rough": 8, "slope_up": 4, "slope_down": 4, "stairs_up": 1, "stairs_down": 2}      # highest level index that can be unlocked (Skill A: rough to 10 cm, slopes to 15 deg, stairs up to 4 cm, stairs down to 6 cm)
HOLD_GAIN, WZ_MAX = 2.0, 0.5


class Go1TerrainWalkEnv(TerrainMixin, Go1WalkingSineEnv):
    def __init__(self, vx_range=(0.4, 0.8), lift=None, **kwargs):
        super().__init__(push=False, vx_range=vx_range, **kwargs)
        if lift is not None: self.lift = float(lift)                  # peak foot lift of the reference gait (default 0.08 m)
        self.top = {k: 0 for k in LEVELS}; self.stats = {}; self.current = ("rough", 0.0); self.type_probs, self.cap = dict(TYPE_PROBS), dict(CAP)      # instance copies, so a subclass can change the mix and the ceilings

    # ---- curriculum ---------------------------------------------------------------------------------------------
    def set_top(self, top): self.top = {k: int(min(top.get(k, 0), self.cap.get(k, 0))) for k in LEVELS}

    def pop_stats(self):
        s, self.stats = self.stats, {}; return s

    def _draw_terrain(self, rng):
        names = list(self.type_probs); kind = names[rng.choice(len(names), p=list(self.type_probs.values()))]
        key = "rough" if kind == "flat" else kind; top = self.top[key] if kind != "flat" else 0
        idx = top if (rng.random() < 0.5 or top == 0) else int(rng.integers(0, top + 1))
        return (key, LEVELS[key][idx]) if kind != "flat" else ("rough", 0.0)

    # ---- command ------------------------------------------------------------------------------------------------
    def _sample_command(self): self.command = np.array([self.np_random.uniform(*self.vx_range), 0.0])

    def _hold(self):
        R = self.data.xmat[1].reshape(3, 3); yaw = float(np.arctan2(R[1, 0], R[0, 0])); self.command[1] = float(np.clip(-HOLD_GAIN * yaw, -WZ_MAX, WZ_MAX))

    def reset(self, seed=None, options=None):
        options = dict(options or {}); rng = np.random.default_rng(seed) if seed is not None else (self.np_random if self.np_random is not None else np.random.default_rng())
        if options.get("terrain") is None: options["terrain"] = self._draw_terrain(rng)
        if options.get("yaw") is None: options["yaw"] = float(rng.uniform(-0.1, 0.1))
        obs, info = super().reset(seed=seed, options=options); self.current = tuple(options["terrain"]); self._hold(); obs[45:47] = self.command
        info["terrain"] = self.current; return obs, info

    # ---- step ---------------------------------------------------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        success = (not terminated) and float(self.data.qpos[0]) >= X_GOAL
        if success: truncated = True
        if not (terminated or truncated): self._hold(); obs[45:47] = self.command
        if terminated or truncated:
            s = self.stats.setdefault(self.current, [0, 0]); s[0] += 1; s[1] += int(success)
            info["success"] = bool(success); info["terrain"] = self.current
        return obs, reward, terminated, truncated, info
