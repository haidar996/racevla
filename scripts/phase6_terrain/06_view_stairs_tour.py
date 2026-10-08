"""Viewer tour of the terrain (default: the stairs; --terrains others: rough ground and slopes). For the stairs: the packaged terrain policy (models/terrain_policy_seed2.zip, forward command 0.6 m/s + heading hold) climbs UP stairs of 2, 4, 6, 8, 10, 12, 15 cm, then goes DOWN stairs of the same heights,
one episode per height, in one window (the height field is updated live). The window shows the case, x / y position, speed, feet and the outcome (SUCCESS: x >= 6 m; FELL; STUCK: less than 0.3 m of progress in 4 s; LEFT THE COURSE).
Usage: python 06_view_stairs_tour.py [--terrains stairs|others] [--model path.zip] [--speed 0.6] [--seed 0] [--no-window]   Loops until the window is closed."""
import sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch; torch.set_num_threads(1)
import mujoco, mujoco.viewer
from stable_baselines3 import PPO
from racevla.envs.go1_terrain_walk import Go1TerrainWalkEnv
from racevla.envs.terrain import X_GOAL
from racevla.envs.wrappers import FixedObsNormalize

arg = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default
model_path, speed, seed = arg("--model", "models/terrain_policy_seed2.zip"), float(arg("--speed", 0.6)), int(arg("--seed", 0))
model = PPO.load(ROOT / model_path, device="cpu"); raw = Go1TerrainWalkEnv(); env = FixedObsNormalize(raw); raw._sample_command = lambda: None
STAIRS = [(k, h) for k in ("stairs_up", "stairs_down") for h in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15)]
OTHERS = [("rough", h) for h in (0.02, 0.04, 0.06, 0.08, 0.10)] + [(k, a) for k in ("slope_up", "slope_down") for a in (3, 6, 9, 12, 15)]       # --terrains others: rough ground, slopes up and down
CASES = OTHERS if arg("--terrains", "stairs") == "others" else STAIRS
NAMES = {"stairs_up": "stairs UP", "stairs_down": "stairs DOWN", "rough": "ROUGH ground (random bumps up to", "slope_up": "slope UP", "slope_down": "slope DOWN"}
FEET = ["FR", "FL", "RR", "RL"]


def run_case(viewer, kind, h, ep):
    if viewer is not None:
        with viewer.lock(): raw.set_terrain(kind, h, seed + ep)
        viewer.update_hfield(raw.hf_id)                         # must be OUTSIDE the lock: it takes the lock itself and blocks until the picture is updated (inside the lock it deadlocks)
    else: raw.set_terrain(kind, h, seed + ep)
    print(f"starting {kind} {h:g}", flush=True)
    obs, _ = env.reset(seed=seed + ep, options={"terrain": (kind, h), "terrain_seed": seed + ep, "yaw": 0.0}); hist, xm, status = [], 0.0, ""      # (without options['terrain'] this environment draws a random terrain)
    what = f"{NAMES[kind]} {h * 100:.0f} cm)" if kind == "rough" else (f"{NAMES[kind]}  {h:g} deg" if kind.startswith("slope") else f"{NAMES[kind]}  {h * 100:.0f} cm per step")
    title = f"{what}   (policy {pathlib.Path(model_path).stem}, command {speed} m/s)"
    for t in range(1000):
        if viewer is not None and not viewer.is_running(): return "closed"
        t0 = time.time(); raw.command = np.array([speed, raw.command[1]]); obs[-4:-2] = raw.command
        obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); raw.command[0] = speed; obs[-4:-2] = raw.command
        x, y = float(raw.data.qpos[0]), float(raw.data.qpos[1]); xm = max(xm, x); hist.append(x); done = False
        if xm >= X_GOAL: status, done = "SUCCESS: crossed it (x >= 6 m)", True
        elif term: status, done = ("LEFT THE COURSE" if raw.termination_reason == "left_course" else "FELL OVER"), True
        elif t >= 200 and hist[-1] - hist[-200] < 0.3 and x > 0.5: status, done = "STUCK: no progress", True
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; viewer.cam.trackbodyid = 1; viewer.cam.distance = 2.2; viewer.cam.elevation = -12; viewer.cam.azimuth = 90
            if t % 5 == 0 or done:
                down = raw._foot_contacts(); feet = "  ".join(f"{n}:{'down' if c else 'UP'}" for n, c in zip(FEET, down))
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, title, f"x {x:.2f} m (goal 6)   y {y:+.2f} m   speed {info['vx']:.2f} m/s   time {t * 0.02:.1f} s\nfeet: {feet}\n{status}"))
            viewer.sync(); time.sleep(max(0.0, 0.02 - (time.time() - t0)))
        if done:
            if viewer is not None: time.sleep(2.0)
            return status
    return "TIMEOUT"


if "--no-window" in sys.argv:
    for i, (k, h) in enumerate(CASES): print(f"{k} {h:g}: {run_case(None, k, h, i)}", flush=True)
else:
    with mujoco.viewer.launch_passive(raw.model, raw.data) as v:
        ep = 0
        while v.is_running():
            for k, h in CASES:
                res = run_case(v, k, h, ep); print(f"{k} {h:g}: {res}", flush=True); ep += 1
                if res == "closed" or not v.is_running(): break
