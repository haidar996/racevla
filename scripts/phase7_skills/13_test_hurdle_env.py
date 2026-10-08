"""Checks for Go1HurdleEnv (racevla/envs/go1_hurdle.py). Run: python scripts/phase7_skills/13_test_hurdle_env.py"""
import sys, time, pathlib, warnings; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from racevla.envs.go1_hurdle import *
from racevla.envs.terrain import HURDLE_LEVELS
from racevla.envs.wrappers import FixedObsNormalize

env = Go1HurdleEnv(); print("1) spaces and reset: observation", OBS_DIM, "action", ACT_DIM, "; heading +x; starts from rest")
obs, info = env.reset(seed=1, options={"terrain": ("hurdle", 0.20)}); R = env.data.xmat[1].reshape(3, 3)
assert obs.shape == (75,) and env.action_space.shape == (13,) and abs(env.data.qvel[0]) < 0.3 and 1.3 <= env.command[0] <= 1.7; print("   ok, command", np.round(env.command, 2))
print("2) the scan sees the hurdle (8 cm thick, between rows): robot at x = 3.55 m, hurdle 20 cm at x = 5.0 -> the rows covering x = 4.95 .. 5.15 (1.4 and 1.6 m ahead) must show it")
env.reset(seed=2, options={"terrain": ("hurdle", 0.20)}); env.data.qpos[0] = 3.55; import mujoco; mujoco.mj_forward(env.model, env.data); sc = env.scan().reshape(8, 3)[:, 1] / SCAN_SCALE * 100
print("   scan, middle column (cm), rows 0.2 ... 1.6 m ahead:", np.round(sc, 1)); assert sc[:5].max() < 1.0 and sc[6:].max() > 10.0
print("3) the trigger starts the jump; the jump phases run; the robot takes off (feet leave the ground) and the jump ends")
from stable_baselines3 import PPO
model = PPO.load(pathlib.Path(__file__).resolve().parents[2] / "models/running_policy_2p5_seed0.zip", device="cpu")
en = FixedObsNormalize(env); took, ended, fell_n = 0, 0, 0; trig_x = [3.5 + 0.1 * i for i in range(9)]          # (the running policy expects the NORMALISED observation; the trigger counts only while the bar is within 1.6 m, i.e. from x = 3.4 m)
for x0 in trig_x:
    obs, _ = en.reset(seed=3, options={"terrain": ("hurdle", 0.03)}); phases = []; term = False; fired = False
    for t in range(300):
        trig = (not fired) and float(env.data.qpos[0]) >= x0; fired |= trig
        res = model.predict(obs[:49], deterministic=True)[0] if env.jump is None else np.zeros(12, np.float32)             # the running policy runs; during the jump the corrections are zero
        obs, r, term, trunc, info = en.step(np.concatenate([res, [1.0 if trig else -1.0]]).astype(np.float32)); phases.append(env.jump.phase if env.jump is not None else "-")
        if term or trunc: break
    seq = []; [seq.append(p) for p in phases if not seq or seq[-1] != p]
    took += int("flight" in seq); ended += int(seq[-1] == "-" and not term); fell_n += int(term)
    print(f"   trigger when x >= {x0:.1f} m: phases {seq}, ended: {info.get('termination_reason')}")
print(f"   took off in {took} of {len(trig_x)} trigger moments, jump ended and running resumed in {ended}"); assert took >= 4 and "crouch" in seq
print("   trigger outside the bar's range is ignored: trigger every step on the empty course -> no jump ever starts")
obs, _ = en.reset(seed=5, options={"terrain": ("rough", 0.0)}); started = False
for t in range(200):
    res = model.predict(obs[:49], deterministic=True)[0]; obs, r, term, trunc, info = en.step(np.concatenate([res, [1.0]]).astype(np.float32)); started |= env.jump is not None
    if term or trunc: break
assert not started; print("   ok")
print("4) running into the hurdle without jumping ends the episode with 'hit_hurdle' (hurdle 25 cm)")
reasons = []
for sd in range(4, 12):
    obs, _ = en.reset(seed=sd, options={"terrain": ("hurdle", 0.25)}); reason = None; ok = False
    for t in range(400):
        a = model.predict(obs[:49], deterministic=True)[0]; obs, r, term, trunc, info = en.step(np.concatenate([a, [-1.0]]).astype(np.float32))
        if term or trunc: reason = info.get("termination_reason"); ok = bool(info.get("success")); break
    reasons.append((reason, round(float(env.data.qpos[0]), 1), ok))
print("   (end reason, x, success) over 8 episodes:", reasons); assert not any(r[2] for r in reasons) and any(r[0] == "hit_hurdle" for r in reasons)
print("5) flat episodes (no hurdle): can end in success: 2 s after passing x = 5.45 m upright at >= 50 % of the speed (running policy, no jump)")
e2 = FixedObsNormalize(Go1HurdleEnv()); ok = 0
for s in range(4):
    obs, _ = e2.reset(seed=s, options={"terrain": ("rough", 0.0)})
    for t in range(400):
        a = model.predict(obs[:49], deterministic=True)[0]; obs, r, term, trunc, info = e2.step(np.concatenate([a, [-1.0]]).astype(np.float32))
        if term or trunc: break
    ok += int(info.get("success", False)); print(f"   episode {s}: steps {t + 1}, x {float(e2.unwrapped.data.qpos[0]):.1f}, success {info.get('success')}")
assert ok >= 3
print("6) curriculum helpers and the draw")
e3 = Go1HurdleEnv(); rng = np.random.default_rng(0); draws = [e3._draw_terrain(rng) for _ in range(2000)]; assert abs(np.mean([k == "rough" for k, _ in draws]) - 0.2) < 0.04 and all(h == 0.03 for k, h in draws if k == "hurdle")
e3.set_top(9); draws = [e3._draw_terrain(rng) for _ in range(2000)]; assert max(h for k, h in draws if k == "hurdle") == 0.25; print("   ok")
print("7) Stable-Baselines3 check_env")
from stable_baselines3.common.env_checker import check_env
warnings.simplefilter("error"); check_env(FixedObsNormalize(Go1HurdleEnv()), warn=True); warnings.simplefilter("default"); print("   passes")
print("\nHURDLE ENV CHECKS PASSED")
