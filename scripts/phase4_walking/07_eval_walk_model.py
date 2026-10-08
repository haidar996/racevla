"""Phase 4: unified final evaluation of one walking model, the same protocol for every option (A clock, B sine, C gait-reward).
Usage: python 07_eval_walk_model.py <run-name> <env: walk|clock|sine|gait> [checkpoint file, default model_1000000.zip]
Writes outputs/analysis/walk_eval/<run-name>.json (50 fresh random-command episodes + steady-command tests) and outputs/analysis/walk_eval/<run-name>_contact_sheet.png (10 side-view frames, 0.1 s apart)."""
import sys, os, json, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from stable_baselines3 import PPO
from racevla.envs.go1_walking import Go1WalkingEnv
from racevla.envs.go1_walking_clock import Go1WalkingClockEnv
from racevla.envs.go1_walking_sine import Go1WalkingSineEnv
from racevla.envs.go1_walking_gait import Go1WalkingGaitRewardEnv
from racevla.envs.wrappers import FixedObsNormalize

ENVS = {"walk": Go1WalkingEnv, "clock": Go1WalkingClockEnv, "sine": Go1WalkingSineEnv, "gait": Go1WalkingGaitRewardEnv}
run, env_name = sys.argv[1], sys.argv[2]; ckpt = sys.argv[3] if len(sys.argv) > 3 else "model_1000000.zip"
model = PPO.load(ROOT / "outputs" / run / "checkpoints" / ckpt, device="cpu"); is_clock = env_name in ("clock", "sine")
out_dir = ROOT / "outputs" / "analysis" / "walk_eval"; out_dir.mkdir(parents=True, exist_ok=True)
FEET = ["FR", "FL", "RR", "RL"]


def fixed_env(cmd, seed):
    env = FixedObsNormalize(ENVS[env_name](push=False)); raw = env.unwrapped; obs, _ = env.reset(seed=seed)
    raw._sample_command = lambda: None; set_cmd(raw, obs, cmd); return env, raw, obs


def set_cmd(raw, obs, cmd):
    raw.command = np.array(cmd, float)
    if is_clock: obs[-4:-2] = raw.command; obs[-2:] = raw._clock()          # observation = 45 robot + 2 command + 2 clock
    else: obs[-2:] = raw.command                                            # observation = 45 robot + 2 command


result = {"run": run, "env": env_name, "checkpoint": ckpt, "steady": {}}
for cmd in [(0.0, 0.0), (0.2, 0.0), (0.4, 0.0), (0.6, 0.0), (0.4, 0.4), (0.0, 0.5), (0.0, -0.5)]:
    vx, wz, fell, nf, sl, fl = [], [], 0, [], [], []
    for seed in range(4):
        env, raw, obs = fixed_env(cmd, 3000 + seed)
        for t in range(400):
            obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            if t >= 100: c = raw._foot_contacts(); vx.append(info["vx"]); wz.append(info["wz"]); nf.append(int(c.sum())); sl.append(info["p_slip"]); fl.append(c.astype(float))
            if term: fell += 1; break
    result["steady"][f"{cmd[0]:.1f},{cmd[1]:+.1f}"] = {"vx": float(np.mean(vx)), "wz": float(np.mean(wz)), "falls_of_4": fell, "feet_down": float(np.mean(nf)), "slip": float(np.mean(sl)), "on_floor": np.mean(fl, axis=0).round(3).tolist()}

env = FixedObsNormalize(ENVS[env_name](push=False)); raw = env.unwrapped
C, V, W, CW, nf, fell, rl, sl, heights = [], [], [], [], [], 0, [], [], []
for s in range(50):
    obs, _ = env.reset(seed=2000 + s)
    for t in range(1000):
        cv, cw = raw.command; obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
        C.append(cv); V.append(info["vx"]); CW.append(cw); W.append(info["wz"]); c = raw._foot_contacts(); nf.append(c.sum()); rl.append(c); sl.append(info["p_slip"])
        if hasattr(raw, "_foot_heights") and (cv > 0.1 or abs(cw) > 0.1): heights.append(raw._foot_heights())
        if term or trunc: fell += int(term); break
C, V, CW, W, nf, rl = map(np.array, (C, V, CW, W, nf, rl)); mv = C > 0.3; tv = np.abs(CW) > 0.3; fm = C > 0.1; tm = np.abs(CW) > 0.1
result["random"] = {"episodes": 50, "falls": int(fell), "fwd_error": float(np.mean(np.abs(V[fm] - C[fm]))), "speed_ratio": float(V[mv].mean() / C[mv].mean()), "fwd_corr": float(np.corrcoef(C, V)[0, 1]),
                    "turn_error": float(np.mean(np.abs(W[tm] - CW[tm]))), "turn_achieved": float(np.mean(W[tv] * np.sign(CW[tv]))), "turn_commanded": float(np.mean(np.abs(CW[tv]))), "turn_corr": float(np.corrcoef(CW, W)[0, 1]),
                    "feet_down": float(nf.mean()), "on_floor": rl.mean(0).round(3).tolist(), "slip": float(np.mean(sl)), "lift_cm_p95": float(100 * np.percentile(np.concatenate(heights), 95)) if heights else None}
json.dump(result, open(out_dir / f"{run}.json", "w"), indent=1)

from PIL import Image, ImageDraw
import mujoco
env, raw, obs = fixed_env((0.5, 0.0), 3000); rend = mujoco.Renderer(raw.model, 300, 400); cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; cam.distance = 1.3; cam.elevation = -8
frames = []
for t in range(330):
    obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
    if term: break
    if t >= 250 and (t - 250) % 5 == 0:
        R = raw.data.xmat[1].reshape(3, 3); cam.azimuth = np.degrees(np.arctan2(R[1, 0], R[0, 0])) + 90; cam.lookat[:] = raw.data.qpos[:3] + [0, 0, -0.05]
        rend.update_scene(raw.data, camera=cam); img = Image.fromarray(rend.render().copy()); c = raw._foot_contacts()
        ImageDraw.Draw(img).text((6, 4), f"{run}  t={t * 0.02:.2f}s  " + " ".join(f"{n}:{'down' if x else 'UP'}" for n, x in zip(FEET, c)), fill=(255, 255, 0)); frames.append(img)
if len(frames) >= 10:
    W_, H_ = frames[0].size; sheet = Image.new("RGB", (W_ * 2, H_ * 5))
    for i, f in enumerate(frames[:10]): sheet.paste(f, ((i % 2) * W_, (i // 2) * H_))
    sheet.save(out_dir / f"{run}_contact_sheet.png")
print(f"{run}: random-command falls {fell}/50, fwd error {result['random']['fwd_error']:.3f}, turn error {result['random']['turn_error']:.3f}, on floor {result['random']['on_floor']}", flush=True)
