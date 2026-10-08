"""Phase 6: checks for Go1TerrainWalkEnv (racevla/envs/go1_terrain_walk.py). Run: python scripts/phase6_terrain/04_test_terrain_walk_env.py"""
import sys, time, pathlib, warnings; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from racevla.envs.go1_terrain_walk import *
from racevla.envs.terrain import LEVELS, X_GOAL
from racevla.envs.wrappers import FixedObsNormalize

env = Go1TerrainWalkEnv()
print("1) terrain draw: types follow the mix, levels follow the unlocked tops (all tops 0 -> lowest levels only)")
rng = np.random.default_rng(0); draws = [env._draw_terrain(rng) for _ in range(4000)]
from collections import Counter
cnt = Counter(k for k, lv in draws); print("   share by kind (flat counted inside rough):", {k: round(v / 4000, 2) for k, v in cnt.items()})
assert abs(cnt["rough"] / 4000 - 0.40) < 0.04 and abs(cnt["slope_up"] / 4000 - 0.20) < 0.04 and abs(cnt["stairs_down"] / 4000 - 0.15) < 0.04
assert all(lv == LEVELS[k][0] for k, lv in draws), "with all tops at 0 only the first level of each type may appear"
env.set_top({"rough": 8, "slope_up": 4, "slope_down": 4, "stairs_up": 4, "stairs_down": 6}); assert env.top["stairs_up"] == 1 and env.top["stairs_down"] == 2, env.top
draws = [env._draw_terrain(rng) for _ in range(4000)]
mx = {k: max(LEVELS[k].index(lv) for kk, lv in draws if kk == k) for k in LEVELS}; print("   highest level drawn per type with tops set (stairs capped):", mx); assert mx["rough"] == 8 and mx["slope_up"] == 4 and mx["stairs_up"] == 1 and mx["stairs_down"] == 2
print("2) reset: heading along +x, forward command in 0.4-0.8, turn command = heading hold, observation is 49-D with the command inside")
for s in range(10):
    obs, info = env.reset(seed=s); R = env.data.xmat[1].reshape(3, 3); assert obs.shape == (49,) and 0.4 <= env.command[0] <= 0.8 and abs(np.arctan2(R[1, 0], R[0, 0])) < 0.5
    assert np.allclose(obs[-4:-2], env.command) and "terrain" in info
print("   ok")
print("3) turn command follows the heading error during an episode (rotate the robot by hand: the command must push back)")
env.reset(seed=3, options={"terrain": ("rough", 0.0), "yaw": 0.0}); env.step(np.zeros(12, np.float32)); c0 = env.command[1]
import mujoco; q = env.data.qpos[3:7].copy(); qn = np.zeros(4); mujoco.mju_mulQuat(qn, np.array([np.cos(0.2), 0, 0, np.sin(0.2)]), q); env.data.qpos[3:7] = qn; mujoco.mj_forward(env.model, env.data)
obs, *_ = env.step(np.zeros(12, np.float32)); print(f"   heading error +0.4 rad -> turn command {env.command[1]:+.2f} (was {c0:+.2f})"); assert env.command[1] < -0.3 and np.allclose(obs[-4:-2], env.command)
print("4) success: reaching x >= 6 ends the episode as a TRUNCATION (not a fall) and is counted; a fall is counted as a failure")
env.reset(seed=4, options={"terrain": ("rough", 0.0), "yaw": 0.0}); env.pop_stats(); env.data.qpos[0] = X_GOAL + 0.01; mujoco.mj_forward(env.model, env.data)
_, r, term, trunc, info = env.step(np.zeros(12, np.float32)); print("   terminated", term, "truncated", trunc, "success", info.get("success"), "stats", env.stats); assert trunc and not term and info["success"] and env.stats[("rough", 0.0)] == [1, 1]
env.reset(seed=5, options={"terrain": ("slope_up", 9), "yaw": 0.0}); env.pop_stats(); env.data.qpos[1] = 1.45; mujoco.mj_forward(env.model, env.data)
_, r, term, trunc, info = env.step(np.zeros(12, np.float32)); assert term and info["success"] is False and env.stats[("slope_up", 9)] == [1, 0]; print("   left-course failure counted: ok"); assert env.pop_stats() and env.stats == {}
print("5) the flat-floor walking environment still behaves as before (same observation for the same seed)")
from racevla.envs.go1_walking_sine import Go1WalkingSineEnv
a, _ = Go1WalkingSineEnv(push=False).reset(seed=9); b, _ = Go1WalkingSineEnv(push=False).reset(seed=9); assert np.array_equal(a, b); print("   ok")
print("6) Stable-Baselines3 check_env on the wrapped environment")
from stable_baselines3.common.env_checker import check_env
warnings.simplefilter("error"); check_env(FixedObsNormalize(Go1TerrainWalkEnv()), warn=True); warnings.simplefilter("default"); print("   passes")
print("7) the packaged walking policy on this environment (flat + one terrain each, 4 episodes): sanity of the whole chain")
from stable_baselines3 import PPO
model = PPO.load(pathlib.Path(__file__).resolve().parents[2] / "models/walking_policy_sine_seed1.zip", device="cpu"); e2 = FixedObsNormalize(Go1TerrainWalkEnv()); res = {}
for terrain in [("rough", 0.0), ("rough", 0.06), ("slope_up", 9), ("stairs_down", 0.04)]:
    ok = 0
    for s in range(4):
        obs, _ = e2.reset(seed=s, options={"terrain": terrain})
        for t in range(1000):
            obs, r, term, trunc, info = e2.step(model.predict(obs, deterministic=True)[0])
            if term or trunc: break
        ok += int(info.get("success", False))
    res[terrain] = ok
print("   successes of 4:", res); assert res[("rough", 0.0)] >= 3
print("\nTERRAIN WALK ENV CHECKS PASSED")
