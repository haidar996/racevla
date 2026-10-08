"""Viewer for the hurdle step-over skill (models/hurdle_stepover_seed0.zip): the robot runs at 1.3-1.7 m/s towards a bar across the course (x = 5 m) and steps over it; the bar height changes from case to case (3, 5, 7, 10, 12.5, 15 cm, two runs each),
in slow motion (default x2; --slow 1 = real time). The window shows the bar height, x, speed, the height of the four feet and the outcome (CROSSED, SUCCESS when it keeps running 2 s after the bar, HIT: trunk / hip / thigh touched the bar, FELL).
Usage: python 18_view_stepover.py [--slow 2] [--model path.zip] [--no-window]   (loops until the window is closed)"""
import sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch; torch.set_num_threads(1)
import mujoco, mujoco.viewer
from stable_baselines3 import PPO
from racevla.envs.go1_stepover import Go1StepOverEnv
from racevla.envs.wrappers import FixedObsNormalize

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
slow, model_path = float(arg("--slow", 2.0)), arg("--model", "models/hurdle_stepover_seed0.zip")
model = PPO.load(ROOT / model_path, device="cpu"); raw = Go1StepOverEnv(); env = FixedObsNormalize(raw)
CASES = [h for h in (0.03, 0.05, 0.07, 0.10, 0.125, 0.15) for _ in range(2)]


def run_case(viewer, h, seed):
    if viewer is not None:
        with viewer.lock(): raw.set_terrain("hurdle", h, seed)
        viewer.update_hfield(raw.hf_id)                    # outside the lock (it takes the lock itself)
    else: raw.set_terrain("hurdle", h, seed)
    obs, _ = env.reset(seed=seed, options={"terrain": ("hurdle", h), "terrain_seed": seed}); status = "running"
    for t in range(450):
        if viewer is not None and not viewer.is_running(): return None
        t0 = time.time(); obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); d = raw.data
        if info.get("crossed", 0) > 0: status = "CROSSED the bar, keeps running..."
        if info.get("success"): status = "SUCCESS: crossed and still running at >= 50 % of the speed"
        elif term: status = "HIT the bar (trunk / hip / thigh)" if info.get("termination_reason") == "hit_hurdle" else "FELL"
        elif trunc: status = "ended (not running fast enough after the bar)"
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; viewer.cam.trackbodyid = 1; viewer.cam.distance = 3.0; viewer.cam.elevation = -8; viewer.cam.azimuth = 90
            fh = " ".join(f"{max(0.0, float(x)) * 100:2.0f}" for x in raw._foot_heights())
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, f"step over a {h * 100:g} cm bar (bar at x = 5 m)   slow motion x{slow:g}",
                              f"x {d.qpos[0]:.2f} m   speed {info['vx']:.2f} m/s (command {raw.command[0]:.2f})   foot heights FR FL RR RL: {fh} cm\n{status}"))
            viewer.sync(); time.sleep(max(0.0, 0.02 * slow - (time.time() - t0)))
        if term or trunc:
            if viewer is not None: time.sleep(2.0)
            return status
    return status


if "--no-window" in sys.argv:
    for i, h in enumerate(CASES): print(f"{h * 100:g} cm: {run_case(None, h, 2000 + i)}", flush=True)
else:
    with mujoco.viewer.launch_passive(raw.model, raw.data) as v:
        ep = 0
        while v.is_running():
            for h in CASES:
                res = run_case(v, h, 2000 + ep); print(f"{h * 100:g} cm: {res}", flush=True); ep += 1
                if res is None: break
