"""Phase 7, recovery skill: checks for Go1RecoveryEnv (racevla/envs/go1_recovery.py). Run: python scripts/phase7_skills/01_test_recovery_env.py"""
import sys, time, pathlib, warnings; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from racevla.envs.go1_recovery import *
from racevla.envs.go1_recovery import CLASSES
from racevla.envs.wrappers import FixedObsNormalize

env = Go1RecoveryEnv()
print("1) basics: 45-D observation, action scale", RECOVERY_ACTION_SCALE, "rad, episode", MAX_STEPS, "steps")
obs, info = env.reset(seed=1); assert obs.shape == (45,) and env.action_scale == RECOVERY_ACTION_SCALE and env.action_space.shape == (12,); print("   ok")

print("2) starts: right side up only (tilt after the settle <= 50 deg), no self-penetration, per class (60 starts each)")
for cls in (0, 1, 2):
    tilts, hs, eps, t0 = [], [], [], time.time()
    for s in range(60):
        _, i = env.reset(seed=100 + s, options={"class": cls}); d = env.data
        tilts.append(np.degrees(i["start_tilt"])); hs.append(d.qpos[2]); eps.append(np.linalg.norm(d.qpos[7:] - env.home_joints))
        assert i["start_class"] == cls and i["start_tilt"] <= START_TILT_MAX + 1e-9 and not env._self_penetration() and d.xmat[1].reshape(3, 3)[2, 2] > 0.6
    print(f"   class {cls}: tilt mean {np.mean(tilts):4.1f} max {np.max(tilts):4.1f} deg | body height {np.mean(hs):.3f} m (min {np.min(hs):.3f}, max {np.max(hs):.3f}) | e_pose {np.mean(eps):.2f} (max {np.max(eps):.2f}) | {(time.time() - t0) / 60:.2f} s per reset")
print("   rejected starts so far:", env.rejected)

print("3) reproducible: same seed -> same start")
a, _ = env.reset(seed=7, options={"class": 2}); b, _ = Go1RecoveryEnv().reset(seed=7, options={"class": 2}); assert np.array_equal(a, b); print("   ok")

print("4) the action range reaches the sampled start poses: |start joints - home| <= 1.5 rad for every joint")
mx = 0.0
for cls in (1, 2):
    for s in range(40): env.reset(seed=300 + s, options={"class": cls}); mx = max(mx, np.abs(env.data.qpos[7:] - env.home_joints).max())
print(f"   largest joint distance from home among 80 starts: {mx:.2f} rad"); assert mx <= RECOVERY_ACTION_SCALE

print("5) reward: standing at home with zero action vs lying sprawled with zero action vs random actions")
def run(cls, seed, policy, n=100):
    env.reset(seed=seed, options={"class": cls}); rs, st = [], []
    for t in range(n):
        _, r, term, trunc, info = env.step(policy(t)); rs.append(r); st.append(info["stand_ok"])
        if term or trunc: break
    return np.mean(rs[-20:]), np.mean(st[-20:]), info
rng = np.random.default_rng(0); zero = lambda t: np.zeros(12, np.float32)
rs0, ss0, i0 = run(0, 11, zero); rs2, ss2, i2 = run(2, 11, zero); rr, sr, ir = run(2, 11, lambda t: rng.uniform(-1, 1, 12).astype(np.float32))
print(f"   standing (class 0, zero action): reward {rs0:.2f}, stand_ok {ss0:.2f}, tilt {np.degrees(i0['tilt']):.0f} deg, height {i0['height']:.3f}")
print(f"   sprawled (class 2, zero action): reward {rs2:.2f}, stand_ok {ss2:.2f}, tilt {np.degrees(i2['tilt']):.0f} deg, height {i2['height']:.3f}, e_pose {i2['e_pose']:.2f}")
print(f"   random actions from sprawl:      reward {rr:.2f}")
assert rs0 > 0.8 and rs2 < 0.5 and rs2 < rs0 - 0.3, "standing should pay clearly more than lying still"

print("5b) the reward is never negative while the episode continues, even under exploration noise (class 0 and class 2 starts, action noise std 0.1 -> 0.15 rad and std 0.37 -> 0.55 rad)")
for std in (0.1, 0.37):
    mins, lens = [], []
    for cls in (0, 2):
        for s_ in range(10):
            env.reset(seed=700 + s_, options={"class": cls}); n = 0
            for t in range(500):
                _, r, term, trunc, info = env.step(np.clip(rng.normal(0, std, 12), -1, 1).astype(np.float32)); n += 1
                if term or trunc: break
                mins.append(r)
            lens.append(n)
    print(f"   noise std {std}: min per-step reward {np.min(mins):.3f}, mean {np.mean(mins):.2f}, mean episode length {np.mean(lens):.0f}"); assert np.min(mins) >= 0.0

print("6) flipping over ends the episode with the penalty")
env.reset(seed=5, options={"class": 0}); d = env.data; d.qpos[3:7] = [np.cos(np.deg2rad(50)), np.sin(np.deg2rad(50)), 0, 0]      # quaternion of a 100 deg roll
import mujoco; mujoco.mj_forward(env.model, d); _, r, term, trunc, info = env.step(np.zeros(12, np.float32))
print(f"   body rolled 100 deg: terminated {term}, reason {info.get('termination_reason')}, reward {r:.1f}"); assert term and info["termination_reason"] == "flipped" and r < -4

print("7) the episode is truncated at", MAX_STEPS, "steps (not terminated)")
env.reset(seed=6, options={"class": 0}); n = 0
while True:
    _, r, term, trunc, info = env.step(np.zeros(12, np.float32)); n += 1
    if term or trunc: break
print(f"   ended after {n} steps, truncated {trunc}, terminated {term}"); assert trunc and not term and n == MAX_STEPS

print("8) Stable-Baselines3 check_env on the wrapped environment")
from stable_baselines3.common.env_checker import check_env
warnings.simplefilter("error"); check_env(FixedObsNormalize(Go1RecoveryEnv()), warn=True); warnings.simplefilter("default"); print("   passes")

print("9) speed (random actions, auto-reset)")
e2 = Go1RecoveryEnv(); e2.reset(seed=0); n = 3000; t0 = time.time(); rng = np.random.default_rng(2)
for i in range(n):
    _, r, term, trunc, info = e2.step(rng.uniform(-1, 1, 12).astype(np.float32))
    if term or trunc: e2.reset()
print(f"   {n / (time.time() - t0):.0f} policy steps/s")
print("\nRECOVERY ENV CHECKS PASSED")
