"""Phase 6, Skill B: checks for Go1ScanWalkEnv (racevla/envs/go1_scan_walk.py). Run: python scripts/phase6_terrain/09_test_scan_env.py"""
import sys, time, pathlib, warnings; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from racevla.envs.go1_scan_walk import *
from racevla.envs.wrappers import FixedObsNormalize

env = Go1ScanWalkEnv(scan_noise=0.0)
print("1) observation: 67 numbers = 49 (robot, command, clock) + 18 scan; lift default 12 cm; the command sits at [45:47] and is refreshed by the heading hold")
obs, info = env.reset(seed=1, options={"terrain": ("rough", 0.0), "yaw": 0.0}); assert obs.shape == (67,) and env.observation_space.shape == (67,) and env.lift == 0.12 and np.allclose(obs[45:47], env.command); print("   ok")
print("2) the scan on flat ground is zero; in front of a 10 cm step it shows +0.10 (x4 = 0.40) at the rows beyond the step, nothing at the rows before it")
assert np.abs(env.scan()).max() < 1e-6
env.reset(seed=2, options={"terrain": ("stairs_up", 0.10), "yaw": 0.0}); import mujoco                    # stairs start at x = 1 m; the scan reaches 0.9 m ahead, so from x = 0 nothing is visible yet
assert np.abs(env.scan()).max() < 1e-6; env.data.qpos[0] = 0.5; mujoco.mj_forward(env.model, env.data)
sc = env.scan().reshape(6, 3); print("   robot at x = 0.5 m; scan rows (0.15 .. 0.90 m ahead), middle column, in cm:", np.round(sc[:, 1] / SCAN_SCALE * 100, 1))
assert np.allclose(sc[:3, 1], 0, atol=1e-6) and np.allclose(sc[3:5, 1] / SCAN_SCALE, 0.10, atol=1e-3) and 0.10 < sc[5, 1] / SCAN_SCALE <= 0.20 + 1e-3      # the last row sits on the next riser (a 4 cm wide ramp in the height field)
print("3) stairs DOWN show negative heights; a slope up shows a ramp (the same in all three columns)")
env.reset(seed=3, options={"terrain": ("stairs_down", 0.10), "yaw": 0.0}); env.data.qpos[0] = 0.6; mujoco.mj_forward(env.model, env.data); d = env.scan().reshape(6, 3)[:, 1] / SCAN_SCALE * 100; print("   stairs down, robot at x = 0.6 m (cm):", np.round(d, 1)); assert d.min() < -9 and abs(d[0]) < 1e-3 and abs(d[1]) < 1e-3
env.reset(seed=4, options={"terrain": ("slope_up", 12), "yaw": 0.0}); env.data.qpos[0] = 1.2; mujoco.mj_forward(env.model, env.data); r = env.scan().reshape(6, 3); print("   slope up 12 deg, robot at x = 1.2 m, middle column (cm):", np.round(r[:, 1] / SCAN_SCALE * 100, 1)); assert np.all(np.diff(r[:, 1]) > 0) and np.allclose(r[:, 0], r[:, 2], atol=1e-6)
print("4) the scan follows the heading: rotate the robot by 90 deg and the 'ahead' rows look along the new heading (sideways in the world)")
env.reset(seed=5, options={"terrain": ("stairs_up", 0.10), "yaw": 0.0}); env.data.qpos[0] = 0.5; mujoco.mj_forward(env.model, env.data); a = env.scan()
q = env.data.qpos[3:7].copy(); qn = np.zeros(4); mujoco.mju_mulQuat(qn, np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]), q); env.data.qpos[3:7] = qn; mujoco.mj_forward(env.model, env.data); b = env.scan()
print(f"   heading 0: largest scan {np.abs(a).max() / SCAN_SCALE * 100:.1f} cm; heading 90 deg: {np.abs(b).max() / SCAN_SCALE * 100:.1f} cm"); assert np.abs(a).max() > 0.1 and np.abs(b).max() < 1e-6
print("5) noise: with scan_noise 0.5 cm the scan on flat ground has std about 0.5 cm x 4; episodes end on success / fall as before; stats work")
en = Go1ScanWalkEnv(scan_noise=0.005); en.reset(seed=6, options={"terrain": ("rough", 0.0), "yaw": 0.0}); v = np.concatenate([en.scan() for _ in range(300)]); print(f"   std of the noisy flat scan {v.std() / SCAN_SCALE * 100:.2f} cm"); assert 0.35 < v.std() / SCAN_SCALE * 100 < 0.65
print("6) caps and mix of Skill B: stairs up and down unlock up to 15 cm, rough 10 cm; stairs 50 % of the episodes")
e2 = Go1ScanWalkEnv(); e2.set_top({k: 99 for k in e2.top}); assert e2.top["stairs_up"] == 6 and e2.top["stairs_down"] == 6 and e2.top["rough"] == 8; rng = np.random.default_rng(0); dr = [e2._draw_terrain(rng)[0] for _ in range(4000)]
print("   share stairs:", round(np.mean([k.startswith("stairs") for k in dr]), 2)); assert abs(np.mean([k.startswith("stairs") for k in dr]) - 0.5) < 0.04
print("7) Stable-Baselines3 check_env")
from stable_baselines3.common.env_checker import check_env
warnings.simplefilter("error"); check_env(FixedObsNormalize(Go1ScanWalkEnv()), warn=True); warnings.simplefilter("default"); print("   passes")
print("8) speed")
e3 = Go1ScanWalkEnv(); e3.reset(seed=0); n = 1500; t0 = time.time(); rng = np.random.default_rng(1)
for i in range(n):
    _, r, term, trunc, info = e3.step(rng.uniform(-1, 1, 12).astype(np.float32))
    if term or trunc: e3.reset()
print(f"   {n / (time.time() - t0):.0f} policy steps/s")
print("\nSCAN ENV CHECKS PASSED")
