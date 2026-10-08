"""Diagnosis for the hurdle pilot: how often does a HAND-CODED trigger (jump when the base is d metres before the hurdle) clear the hurdle with the real running policy (corrections zero during the jump, scripted jump as in the feasibility test)?
Outcomes per episode: hit (a non-foot part touched the bar zone), cleared (crossed without a hit), success (the environment's rule: x >= 7.5 m, upright, speed >= 50 % of the command), fell. Also the forward speed at x = 7.5 m.
Usage: python 15_hurdle_fixed_trigger.py   -> prints a table (heights 10 / 15 / 20 / 25 cm x trigger distances 0.3 ... 1.2 m, 6 episodes each)"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch; torch.set_num_threads(1)
from stable_baselines3 import PPO
from racevla.envs.go1_hurdle import Go1HurdleEnv
from racevla.envs.terrain import X_HURDLE
from racevla.envs.wrappers import FixedObsNormalize

model = PPO.load(ROOT / "models/running_policy_2p5_seed0.zip", device="cpu"); raw = Go1HurdleEnv(scan_noise=0.0); env = FixedObsNormalize(raw)
print("height | trigger distance | hit | cleared | success | fell | mean speed at the end of the episode (m/s)")
for h in (0.10, 0.15, 0.20, 0.25):
    for dist in (0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2):
        c = dict(hit=0, cleared=0, success=0, fell=0); sp = []
        for s in range(6):
            obs, _ = env.reset(seed=s, options={"terrain": ("hurdle", h)}); fired = False
            for t in range(400):
                trig = (not fired) and raw.data.qpos[0] >= X_HURDLE - dist
                res = model.predict(obs[:49], deterministic=True)[0] if raw.jump is None else np.zeros(12, np.float32)
                obs, r, term, trunc, info = env.step(np.concatenate([res, [1.0 if trig else -1.0]]).astype(np.float32)); fired |= trig
                if term or trunc: break
            reason = info.get("termination_reason"); c["hit"] += int(reason == "hit_hurdle"); c["fell"] += int(term and reason != "hit_hurdle"); c["cleared"] += int(info.get("crossed", 0) > 0 and reason != "hit_hurdle"); c["success"] += int(bool(info.get("success")))
            sp.append(float(raw.data.qvel[0]))
        print(f"{h * 100:4.0f} cm | {dist:4.2f} m | {c['hit']}/6 | {c['cleared']}/6 | {c['success']}/6 | {c['fell']}/6 | {np.mean(sp):.2f}", flush=True)
