"""Test reset(): reproducibility, variety, valid starts, and how hard the starts are."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import mujoco, numpy as np
from racevla.envs.go1_standing import Go1StandingEnv
from racevla.envs.observation import OBS_SLICES as S
np.set_printoptions(precision=3, suppress=True)
env = Go1StandingEnv()

print("1) same seed -> identical start; different seed -> different start")
a, _ = env.reset(seed=7); b, _ = env.reset(seed=7); c, _ = env.reset(seed=8)
print("   seed 7 vs 7 identical:", np.array_equal(a, b), "| seed 7 vs 8 differ:", not np.allclose(a, c)); assert np.array_equal(a, b) and not np.allclose(a, c)

print("\n2) 2000 resets: validity and spread")
N = 2000; clearance, tilt, heights, npen = [], [], [], 0
for i in range(N):
    obs, info = env.reset(seed=1000 + i); d = env.data
    assert obs.shape == (45,) and np.isfinite(obs).all()
    clearance.append(env._lowest_foot_bottom()); heights.append(d.qpos[2])
    tilt.append(np.degrees(np.arccos(np.clip(-obs[S['gravity']][2], -1, 1))))
    npen += int(any(c.dist < -0.002 for c in d.contact[:d.ncon]))     # any real penetration at t=0
    assert np.all(d.qpos[7:] >= env.joint_lo - 1e-9) and np.all(d.qpos[7:] <= env.joint_hi + 1e-9)
print(f"   lowest foot above floor: min {min(clearance):.4f} max {max(clearance):.4f} m (want ~0.010)")
print(f"   base height: min {min(heights):.3f} max {max(heights):.3f} m | tilt: mean {np.mean(tilt):.1f} max {max(tilt):.1f} deg")
print(f"   starts with penetration: {npen}/{N}")
assert npen == 0 and min(clearance) > 0

print("\n3) how hard are the starts? hold the home pose with PD for 1 s, count falls (200 seeds)")
falls = 0; end_z = []
for i in range(200):
    env.reset(seed=5000 + i)
    for _ in range(500): env.pd.apply(env.data, env.home_joints); mujoco.mj_step(env.model, env.data)
    up = env.data.xmat[1].reshape(3, 3)[2, 2]; end_z.append(env.data.qpos[2]); falls += int(env.data.qpos[2] < 0.2 or up < 0.8)
print(f"   fell/collapsed: {falls}/200 | end height mean {np.mean(end_z):.3f} m")
print("\nRESET CHECKS PASSED")
