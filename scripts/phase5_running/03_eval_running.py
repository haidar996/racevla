"""Phase 5: unified final evaluation of one running model. Usage: python 03_eval_running.py <run-name> <gait-table.json> [checkpoint file, default latest_model.zip]
Writes outputs/analysis/running/eval/<run>.json: (a) 50 fresh random-command episodes (seeds 5000..) with the commands of the stage, (b) steady-speed tests at 1.0 .. 2.5 m/s (4 episodes each, 8 s, held command):
falls, speed reached, flight share (all four feet off the floor), feet on floor per foot, slip."""
import sys, os, json, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
from stable_baselines3 import PPO
from racevla.envs.go1_running_sine import Go1RunningSineEnv, FLIGHT_MIN_CMD
from racevla.envs.wrappers import FixedObsNormalize

run, table = sys.argv[1], json.load(open(sys.argv[2])); ckpt = sys.argv[3] if len(sys.argv) > 3 else "latest_model.zip"
vmax = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5
model = PPO.load(ROOT / "outputs" / run / ckpt, device="cpu")
out_dir = ROOT / "outputs" / "analysis" / "running" / "eval"; out_dir.mkdir(parents=True, exist_ok=True)


def set_cmd(raw, obs, cmd):
    raw.command = np.array(cmd, float); obs[-4:-2] = raw.command; obs[-2:] = raw._clock()


res = {"run": run, "checkpoint": ckpt, "vmax": vmax, "steady": {}}
for v in (1.0, 1.5, 2.0, 2.5):
    vx, fl, nf, sl, fc, fell = [], [], [], [], [], 0
    for seed in range(4):
        env = FixedObsNormalize(Go1RunningSineEnv(push=False, gait={"table": table})); raw = env.unwrapped; obs, _ = env.reset(seed=6000 + seed); raw._sample_command = lambda: None; set_cmd(raw, obs, (v, 0.0))
        for t in range(400):
            obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); set_cmd(raw, obs, (v, 0.0))
            if term: fell += 1; break
            if t >= 100: c = raw._foot_contacts(); vx.append(info["vx"]); fl.append(float(c.sum() == 0)); nf.append(int(c.sum())); sl.append(info["p_slip"]); fc.append(c.astype(float))
    res["steady"][f"{v:.1f}"] = {"falls_of_4": fell, "vx": float(np.mean(vx)) if vx else None, "flight": float(np.mean(fl)) if fl else None, "feet_down": float(np.mean(nf)) if nf else None,
                                 "slip": float(np.mean(sl)) if sl else None, "on_floor": np.mean(fc, axis=0).round(3).tolist() if fc else None}

env = FixedObsNormalize(Go1RunningSineEnv(push=False, vx_range=(0.0, vmax), gait={"table": table})); raw = env.unwrapped
C, V, CW, W, FL, ND, SL, CON, fell = [], [], [], [], [], [], [], [], 0
for s in range(50):
    obs, _ = env.reset(seed=5000 + s)
    for t in range(1000):
        cv, cw = raw.command; obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); c = raw._foot_contacts()
        C.append(cv); V.append(info["vx"]); CW.append(cw); W.append(info["wz"]); FL.append(float(c.sum() == 0)); ND.append(c.sum()); SL.append(info["p_slip"]); CON.append(c)
        if term or trunc: fell += int(term); break
C, V, CW, W, FL, ND, CON = map(np.array, (C, V, CW, W, FL, ND, CON)); fm = C > 0.1; rm = C >= FLIGHT_MIN_CMD; tv = np.abs(CW) > 0.3; tm = np.abs(CW) > 0.1
res["random"] = {"episodes": 50, "falls": int(fell), "fwd_error": float(np.mean(np.abs(V[fm] - C[fm]))), "run_steps": int(rm.sum()),
                 "run_speed_ratio": float(V[rm].mean() / C[rm].mean()) if rm.any() else None, "run_fwd_error": float(np.mean(np.abs(V[rm] - C[rm]))) if rm.any() else None,
                 "flight_share_running": float(FL[rm].mean()) if rm.any() else None, "flight_share_all": float(FL.mean()),
                 "turn_error": float(np.mean(np.abs(W[tm] - CW[tm]))) if tm.any() else None, "feet_down": float(ND.mean()), "on_floor_running": CON[rm].mean(0).round(3).tolist() if rm.any() else None,
                 "slip": float(np.mean(SL))}
json.dump(res, open(out_dir / f"{run}.json", "w"), indent=1)
r = res["random"]; print(f"{run}: falls {r['falls']}/50, run speed ratio {r['run_speed_ratio']}, flight share at running commands {r['flight_share_running']}, fwd err {r['fwd_error']:.3f}", flush=True)
