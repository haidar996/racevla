"""Phase 4, step 1: checks for Go1WalkingEnv (racevla/envs/go1_walking.py). Run: python scripts/phase4_walking/01_test_walking_env.py"""
import sys, time, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from racevla.envs.go1_walking import (Go1WalkingEnv, WALK_OBS_DIM, CMD_VX_RANGE, CMD_WZ_RANGE, COMMAND_INTERVAL_STEPS, STAND_PROBABILITY, ALIVE_BONUS)
from racevla.envs.wrappers import FixedObsNormalize

env = Go1WalkingEnv(push=False)
print("1) observation: 47 numbers, the last two are the command")
obs, info = env.reset(seed=1)
assert obs.shape == (WALK_OBS_DIM,) == (47,) and env.observation_space.shape == (47,)
assert np.allclose(obs[-2:], env.command) and np.allclose(obs[-2:], [info["cmd_vx"], info["cmd_wz"]])
print(f"   shape {obs.shape}, command in obs {obs[-2:].round(3)}")

print("2) command schedule: resampled every", COMMAND_INTERVAL_STEPS, "steps, inside the ranges, ~", STAND_PROBABILITY, "of them zero")
env.reset(seed=2); seen = []; last = env.command.copy(); changes = []
for t in range(1, 1001):
    obs, r, term, trunc, info = env.step(np.zeros(12, np.float32))
    if not np.allclose(env.command, last): changes.append(t); last = env.command.copy()
    seen.append(env.command.copy())
assert all(c % COMMAND_INTERVAL_STEPS == 0 for c in changes), changes
vx = np.array([c[0] for c in seen]); wz = np.array([c[1] for c in seen])
assert vx.min() >= CMD_VX_RANGE[0] and vx.max() <= CMD_VX_RANGE[1] and wz.min() >= CMD_WZ_RANGE[0] and wz.max() <= CMD_WZ_RANGE[1]
samples = []
for _ in range(4000): env._sample_command(); samples.append(env.command.copy())
zero_frac = np.mean([np.all(s == 0) for s in samples]); assert abs(zero_frac - STAND_PROBABILITY) < 0.03
print(f"   changes at steps {changes}; 4000 sampled commands: {100 * zero_frac:.1f} % zero, vx in [{min(s[0] for s in samples):.2f}, {max(s[0] for s in samples):.2f}], wz in [{min(s[1] for s in samples):.2f}, {max(s[1] for s in samples):.2f}]")

print("3) reward ordering with the robot just standing (zero action): right command scores higher than a wrong one")
def mean_reward(cmd, n=100, seed=3):
    env.reset(seed=seed); env.command = np.array(cmd, float); rs = []
    for t in range(n):
        _, r, term, trunc, info = env.step(np.zeros(12, np.float32)); rs.append((r, info["r_lin_track"], info["r_yaw_track"]))
    return np.mean(rs, axis=0)
stand, walk, turn = mean_reward([0.0, 0.0]), mean_reward([0.5, 0.0]), mean_reward([0.0, 0.5])
print(f"   command (0, 0):   reward {stand[0]:.3f}  speed-tracking {stand[1]:.2f}  turn-tracking {stand[2]:.2f}")
print(f"   command (0.5, 0): reward {walk[0]:.3f}  speed-tracking {walk[1]:.2f}  turn-tracking {walk[2]:.2f}")
print(f"   command (0, 0.5): reward {turn[0]:.3f}  speed-tracking {turn[1]:.2f}  turn-tracking {turn[2]:.2f}")
assert stand[0] > walk[0] and stand[0] > turn[0] and stand[1] > 0.9 and stand[2] > 0.9 and walk[1] < 0.5 and turn[2] < 0.5

print("4) fall rule: standing never falls; random actions end with 'tilt' or 'trunk_contact' only")
env.reset(seed=4); ended = None
for t in range(1000):
    _, r, term, trunc, info = env.step(np.zeros(12, np.float32))
    if term: ended = info["termination_reason"]; break
assert ended is None, f"standing fell: {ended}"
rng = np.random.default_rng(0); reasons = {}
for ep in range(30):
    env.reset(seed=100 + ep)
    for t in range(1000):
        _, r, term, trunc, info = env.step(rng.uniform(-1, 1, 12).astype(np.float32))
        if term: reasons[info["termination_reason"]] = reasons.get(info["termination_reason"], 0) + 1; break
        if trunc: reasons["survived"] = reasons.get("survived", 0) + 1; break
print("   random-action episodes ended with:", reasons)
assert set(reasons) <= {"tilt", "trunk_contact", "survived"}, reasons

print("5) feet-air-time bonus: appears when the robot is asked to move and lifts feet")
rng = np.random.default_rng(1); airs, seen_air = [], False
env.reset(seed=5); env.command = np.array([0.4, 0.0])
for t in range(400):
    _, r, term, trunc, info = env.step(rng.uniform(-1, 1, 12).astype(np.float32)); airs.append(info["r_air"])
    if term or trunc: break
print(f"   {len(airs)} steps, touchdowns with a nonzero air bonus: {int(np.sum(np.abs(airs) > 0))}, max air time seen {env.air_time.max():.2f} s")
assert np.sum(np.abs(airs) > 0) > 0

print("6) reward is never negative; same seed -> same observation; wrapper scales only the first 45 numbers")
assert min(mean_reward([0.6, 0.5], n=50)) >= 0
a1, _ = env.reset(seed=7); a2, _ = Go1WalkingEnv(push=False).reset(seed=7); assert np.array_equal(a1, a2)
wrapped = FixedObsNormalize(Go1WalkingEnv(push=False)); w, _ = wrapped.reset(seed=7); assert w.shape == (47,) and np.allclose(w[-2:], a1[-2:])
print("   ok")

print("7) Stable-Baselines3 check_env on the wrapped environment")
from stable_baselines3.common.env_checker import check_env
import warnings; warnings.simplefilter("error"); check_env(FixedObsNormalize(Go1WalkingEnv(push=True)), warn=True); warnings.simplefilter("default"); print("   passes with no warnings")

print("8) speed (random actions, auto-reset)")
env2 = Go1WalkingEnv(push=True); env2.reset(seed=0); n = 3000; t0 = time.time(); rng = np.random.default_rng(2)
for i in range(n):
    _, r, term, trunc, info = env2.step(rng.uniform(-1, 1, 12).astype(np.float32))
    if term or trunc: env2.reset()
print(f"   {n / (time.time() - t0):.0f} policy steps/s")
print("\nWALKING ENV CHECKS PASSED")
