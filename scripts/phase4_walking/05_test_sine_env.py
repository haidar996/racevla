"""Phase 4, option B: checks for Go1WalkingSineEnv (racevla/envs/go1_walking_sine.py). Run: python scripts/phase4_walking/05_test_sine_env.py"""
import sys, time, pathlib, warnings; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from racevla.envs.go1_walking_sine import Go1WalkingSineEnv, LIFT_HEIGHT, RESIDUAL_SCALE, STRIDE_GAIN, STRIDE_MIN
from racevla.envs.wrappers import FixedObsNormalize

env = Go1WalkingSineEnv(push=False)
print("1) basics: 49-D observation like option A, residual scale", RESIDUAL_SCALE, "rad, reference is zero when told to stand")
obs, _ = env.reset(seed=1); assert obs.shape == (49,) and env.action_scale == RESIDUAL_SCALE
env.command = np.array([0.0, 0.0]); [env.step(np.zeros(12, np.float32)) for _ in range(30)]; assert np.all(env._reference_offset() == 0.0)
env.command = np.array([0.4, 0.0]); off = [env._reference_offset() for _ in range(30)]; assert np.abs(off[-1]).max() > 0.1 and off[0] is not None
print("   ok")

print("2) reference foot path: swing lifts to", LIFT_HEIGHT, "m and moves forward, stance slides backward; half-stride follows the commanded speed")
env.reset(seed=2); env.command = np.array([0.6, 0.0]); env.gate = 1.0
dxs, dzs = [], []
for k in range(200):
    env.phase = k / 200.0; dx, dz = env._foot_reference(); dxs.append(dx[0]); dzs.append(dz[0])
dxs, dzs = np.array(dxs), np.array(dzs)
print(f"   at 0.6 m/s: half-stride {dxs.max():.3f} m (expected {STRIDE_GAIN * 0.6 + STRIDE_MIN:.3f}), peak lift {dzs.max():.3f} m, lift only during swing: {bool(np.all(dzs[85:] == 0.0))}")
assert abs(dxs.max() - (STRIDE_GAIN * 0.6 + STRIDE_MIN)) < 1e-3 and abs(dzs.max() - LIFT_HEIGHT) < 1e-3 and np.all(dzs[85:] == 0.0)
assert np.abs(np.diff(dxs)).max() < 0.005, "the foot path must be continuous"
env.command = np.array([0.4, 0.4]); env.phase = 0.1; dx, dz = env._foot_reference(); print(f"   turning left: right-side foot stride {abs(dx[0]):.3f} m > left-side {abs(dx[1]):.3f} m"); assert abs(dx[0]) > abs(dx[1])

print("3) does the built-in reference ALONE (policy output = 0) make the real robot step? (command forward 0.4 m/s, 12 s, 3 starts)")
res = []
for seed in range(3):
    env.reset(seed=10 + seed); env.command = np.array([0.4, 0.0]); vx, hs, nd, fell = [], [], [], False
    for t in range(600):
        env.command = np.array([0.4, 0.0]); _, r, term, trunc, info = env.step(np.zeros(12, np.float32))
        if t >= 100: vx.append(info["vx"]); hs.append(env._foot_heights().max()); nd.append(env._foot_contacts().sum())
        if term: fell = True; break
    res.append((np.mean(vx) if vx else np.nan, np.percentile(hs, 95) if hs else np.nan, np.mean(nd) if nd else np.nan, fell))
    print(f"   start {seed}: mean vx {res[-1][0]:.2f} m/s | foot lift (95th pct) {100 * res[-1][1]:.1f} cm | feet down {res[-1][2]:.2f} | fell: {fell}")
assert all(np.isfinite(r[1]) for r in res)

print("4) reward is option-B style (no schedule terms), never negative; SB3 environment check")
_, r, term, trunc, info = env.step(np.zeros(12, np.float32)); assert "r_sched" not in info and r >= 0
from stable_baselines3.common.env_checker import check_env
warnings.simplefilter("error"); check_env(FixedObsNormalize(Go1WalkingSineEnv(push=True)), warn=True); warnings.simplefilter("default"); print("   passes")
print("\nSINE ENV CHECKS PASSED")
