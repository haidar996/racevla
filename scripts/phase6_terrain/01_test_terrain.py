"""Phase 6, terrain: checks for racevla/envs/terrain.py. Run: python scripts/phase6_terrain/01_test_terrain.py"""
import sys, time, pathlib, warnings; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from racevla.envs.terrain import *
from racevla.envs.go1_walking import Go1WalkingEnv
from racevla.envs.go1_walking_sine import Go1WalkingSineEnv
from racevla.envs.go1_running_sine import Go1RunningSineEnv

TerrainSine = make_terrain_env(Go1WalkingSineEnv); env = TerrainSine(push=False)
print("1) the height field is written as intended: ray-cast ground height vs the generating function (4 kinds, several levels)")
checks = [("slope_up", 12, [(0.0, 0.0), (2.0, np.tan(np.deg2rad(12)) * 1.0), (5.0, np.tan(np.deg2rad(12)) * 4.0), (8.0, np.tan(np.deg2rad(12)) * 4.0)]),
          ("slope_down", 9, [(0.0, np.tan(np.deg2rad(9)) * 4.0), (3.0, np.tan(np.deg2rad(9)) * 2.0), (6.0, 0.0)]),
          ("stairs_up", 0.10, [(0.5, 0.0), (1.2, 0.10), (1.6, 0.20), (2.8, 0.50), (6.0, 0.50)]), ("stairs_down", 0.08, [(0.0, 0.40), (1.2, 0.32), (2.8, 0.0), (5.0, 0.0)])]
for kind, lvl, pts in checks:
    env.set_terrain(kind, lvl, 0)
    for x, want in pts:
        got = env.ground_z(x); assert abs(got - want) < 0.02, (kind, lvl, x, got, want)
    print(f"   {kind} {lvl}: ok ({len(pts)} points)")
env.set_terrain("rough", 0.05, 3); z = np.array([env.ground_z(x, y) for x in np.arange(1.5, 6.5, 0.07) for y in np.arange(-1.0, 1.0, 0.3)]); lead = np.array([env.ground_z(x) for x in np.arange(-0.9, 0.9, 0.2)])
print(f"   rough 5 cm: heights in the section {z.min() * 100:.1f}..{z.max() * 100:.1f} cm (std {z.std() * 100:.1f}); lead-in flat: {np.abs(lead).max() * 100:.2f} cm"); assert z.max() < 0.0501 and z.min() > -0.001 and z.std() > 0.01 and np.abs(lead).max() < 1e-3

print("2) changing the terrain between episodes works (the collision geometry follows the new data)")
for kind, lvl, x, want in [("stairs_up", 0.15, 3.0, 0.75), ("rough", 0.0, 3.0, 0.0), ("slope_up", 15, 5.5, np.tan(np.deg2rad(15)) * 4), ("rough", 0.0, 5.5, 0.0)]:
    env.set_terrain(kind, lvl, 1); assert abs(env.ground_z(x) - want) < 0.02, (kind, x, env.ground_z(x)); 
print("   ok")

print("3) the same terrain seed gives the same bumps, different seeds differ")
a = height_field("rough", 0.04, 5)[0]; b = height_field("rough", 0.04, 5)[0]; c = height_field("rough", 0.04, 6)[0]; assert np.array_equal(a, b) and not np.array_equal(a, c); print("   ok")

print("4) the robot starts on the lead-in with its feet on the ground (also on a raised start plateau) and stands for 2 s without falling (zero action)")
for kind, lvl in [("rough", 0.05), ("slope_down", 15), ("stairs_down", 0.15), ("stairs_up", 0.15), ("slope_up", 15)]:
    fell = 0
    for s in range(8):
        obs, info = env.reset(seed=s, options={"terrain": (kind, lvl), "yaw": 0.0}); env.command = np.zeros(2)
        low = env._lowest_foot_bottom() - env.ground_z(env.data.qpos[0], env.data.qpos[1]); assert -0.002 < low < 0.03, low
        for t in range(100):
            _, r, term, trunc, info = env.step(np.zeros(12, np.float32)); fell += int(term)
            if term: break
    print(f"   {kind} {lvl}: start foot height above ground {low * 100:.1f} cm, falls while standing: {fell}/8 starts"); assert fell <= 1

print("5) heading: options['yaw'] = 0 points the robot along +x; without it the yaw is random as before")
env.reset(seed=1, options={"yaw": 0.0}); R = env.data.xmat[1].reshape(3, 3); assert abs(np.arctan2(R[1, 0], R[0, 0])) < 0.35, R   # small roll/pitch noise leaks a little into the projected heading
yaws = []; [yaws.append(np.arctan2(*(lambda R: (R[1, 0], R[0, 0]))((env.reset(seed=s), env.data.xmat[1].reshape(3, 3))[1]))) for s in range(20)]; assert np.std(yaws) > 1.0; print("   ok")

print("6) leaving the course sideways ends the episode")
env.reset(seed=2, options={"terrain": ("rough", 0.0), "yaw": 0.0}); env.data.qpos[1] = 1.45; mujoco_forward = __import__("mujoco").mj_forward; mujoco_forward(env.model, env.data)
_, r, term, trunc, info = env.step(np.zeros(12, np.float32)); print("   terminated:", term, info.get("termination_reason")); assert term and info["termination_reason"] == "left_course"

print("7) the flat-floor environments are unchanged (default scene, random yaw stream, ground_height 0)")
f = Go1WalkingEnv(push=False); o1, _ = f.reset(seed=3); o2, _ = Go1WalkingEnv(push=False).reset(seed=3); assert np.array_equal(o1, o2) and f.ground_height == 0.0 and f.model.geom_type[f.floor_id] == 0; print("   ok")

print("8) the running environment works on terrain too, and the speed on terrain")
TR = make_terrain_env(Go1RunningSineEnv); e2 = TR(push=False); e2.reset(seed=0, options={"terrain": ("rough", 0.03), "yaw": 0.0}); n = 1500; t0 = time.time(); rng = np.random.default_rng(1)
for i in range(n):
    _, r, term, trunc, info = e2.step(rng.uniform(-1, 1, 12).astype(np.float32))
    if term or trunc: e2.reset(options={"terrain": ("rough", 0.03), "yaw": 0.0})
print(f"   {n / (time.time() - t0):.0f} policy steps/s on the height field (flat floor for comparison:", end=" ")
e3 = Go1RunningSineEnv(push=False); e3.reset(seed=0); t0 = time.time()
for i in range(n):
    _, r, term, trunc, info = e3.step(rng.uniform(-1, 1, 12).astype(np.float32))
    if term or trunc: e3.reset()
print(f"{n / (time.time() - t0):.0f} steps/s)")
print("\nTERRAIN CHECKS PASSED")
