"""Diagnosis of the Skill B pilot (stairs up of 8 cm and more stay at 0 %): (1) does the policy use the scan at all? (2) how high do the feet really go? (3) what happens at the first step?
Usage: python 10_diagnose_scan.py <checkpoint.zip>"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch; torch.set_num_threads(1)
from stable_baselines3 import PPO
from racevla.envs.go1_scan_walk import Go1ScanWalkEnv, SCAN_OBS_DIM
from racevla.envs.terrain import X_GOAL
from racevla.envs.wrappers import FixedObsNormalize

model = PPO.load(ROOT / sys.argv[1], device="cpu"); raw = Go1ScanWalkEnv(); env = FixedObsNormalize(raw); FOOT_R = raw.foot_radius


def episode(kind, level, seed, record=None):
    obs, _ = env.reset(seed=seed, options={"terrain": (kind, level), "terrain_seed": seed, "yaw": 0.0}); xm, front_peak, res = 0.0, 0.0, "stuck"
    for t in range(1000):
        a = model.predict(obs, deterministic=True)[0]
        if record is not None: record.append((obs.copy(), a.copy(), float(raw.data.qpos[0])))
        obs, r, term, trunc, info = env.step(a); x = float(raw.data.qpos[0]); xm = max(xm, x)
        if x < 1.05: front_peak = max(front_peak, float(max(raw.data.geom_xpos[raw.foot_ids[0], 2], raw.data.geom_xpos[raw.foot_ids[1], 2]) - FOOT_R))
        if info.get("success"): res = "success"; break
        if term: res = "left_course" if raw.termination_reason == "left_course" else "fell"; break
    return res, xm, front_peak


print("1) does the policy use the scan? Stairs up 8 cm, 6 episodes: compare the action with the real scan and with the scan set to zero (same everything else)")
rec = []
for s in range(6): episode("stairs_up", 0.08, 500 + s, rec)
obs = np.array([r[0] for r in rec]); xs = np.array([r[2] for r in rec]); a_real = np.array([r[1] for r in rec]); o0 = obs.copy(); o0[:, 49:] = 0.0
a_zero = np.array([model.predict(o, deterministic=True)[0] for o in o0]); near = (xs > 0.2) & (xs < 1.2)              # the step is within the scan range
d = np.abs(a_real - a_zero).mean(axis=1)
print(f"   mean |action| {np.abs(a_real).mean():.3f}; mean |action(real scan) - action(zero scan)|: everywhere {d.mean():.3f}, with the step within the scan range {d[near].mean():.3f}, on flat ground far from steps {d[~near].mean():.3f}")
print("2) how high do the feet really go? Peak foot height on flat ground (95th percentile of all four feet, cm) for the reference alone (action 0) at lift 8 / 12 cm and for the policy")
for lift in (0.08, 0.12):
    e = Go1ScanWalkEnv(lift=lift); e.reset(seed=1, options={"terrain": ("rough", 0.0), "yaw": 0.0}); h = []
    for t in range(500):
        e.step(np.zeros(12, np.float32))
        if t > 100: h.append(e._foot_heights())
    print(f"   reference only, lift {lift * 100:.0f} cm: p95 {np.percentile(np.concatenate(h), 95) * 100:.1f} cm, max {np.concatenate(h).max() * 100:.1f} cm")
raw.reset(seed=1, options={"terrain": ("rough", 0.0), "yaw": 0.0}); h = []; obs, _ = env.reset(seed=1, options={"terrain": ("rough", 0.0), "yaw": 0.0})
for t in range(500):
    obs, *_ = env.step(model.predict(obs, deterministic=True)[0])
    if t > 100: h.append(raw._foot_heights())
print(f"   the policy (lift 12 cm environment): p95 {np.percentile(np.concatenate(h), 95) * 100:.1f} cm, max {np.concatenate(h).max() * 100:.1f} cm")
print("3) what happens at the first step? 10 episodes each: outcome, mean furthest x (the steps start at x = 1 m), and the peak height of the FRONT feet while approaching (x < 1.05 m), in cm")
for h in (0.06, 0.08, 0.10, 0.12):
    res = [episode("stairs_up", h, 700 + s) for s in range(10)]; c = {k: sum(r[0] == k for r in res) for k in ("success", "fell", "stuck", "left_course")}
    print(f"   step {h * 100:.0f} cm: {c}, mean furthest x {np.mean([r[1] for r in res]):.2f} m, front-foot peak while approaching {np.mean([r[2] for r in res]) * 100:.1f} cm (step height {h * 100:.0f} cm)")
