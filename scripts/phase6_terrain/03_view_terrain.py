"""Viewer: a packaged policy crosses one terrain in the MuJoCo viewer (real time, camera follows the robot from the side, +x to the right). The window shows the terrain, the command, x and y position, speed and what happened.
Usage: python 03_view_terrain.py <policy: walking|running> <kind> <level> <speed> [--hold] [--seed 0] [--model path.zip] [--no-window]
  kind: rough | slope_up | slope_down | stairs_up | stairs_down;  level: cm for rough and stairs (e.g. 0.08 = 8 cm), degrees for slopes;  --hold: simple heading hold (turn command follows the heading error), without it the policy only gets a forward speed
Example: python 03_view_terrain.py walking stairs_up 0.08 0.6      After each crossing (success: x >= 6 m; fell; left the course; stuck for 20 s) it restarts on the same terrain."""
import sys, json, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch; torch.set_num_threads(1)
import mujoco, mujoco.viewer
from stable_baselines3 import PPO
from racevla.envs.terrain import make_terrain_env, X_GOAL
from racevla.envs.go1_walking_sine import Go1WalkingSineEnv
from racevla.envs.go1_running_sine import Go1RunningSineEnv
from racevla.envs.wrappers import FixedObsNormalize

args = [a for a in sys.argv[1:] if not a.startswith("--")]; args = [a for a in args if a not in (sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None, sys.argv[sys.argv.index("--seed") + 1] if "--seed" in sys.argv else None)]
family, kind, level, speed = args[0], args[1], float(args[2]), float(args[3]); hold = "--hold" in sys.argv
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
model_path = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else ("models/walking_policy_sine_seed1.zip" if family == "walking" else "models/running_policy_2p5_seed0.zip")      # --model: any walking-type policy (e.g. outputs/terr_2M_seed0/latest_model.zip)
model = PPO.load(ROOT / model_path, device="cpu")
if family == "walking": raw = make_terrain_env(Go1WalkingSineEnv)(push=False)
else: raw = make_terrain_env(Go1RunningSineEnv)(push=False, gait={"table": json.load(open(ROOT / "models/running_gait_table.json"))})
env = FixedObsNormalize(raw); raw._sample_command = lambda: None
raw.set_terrain(kind, level, seed); unit = "deg" if kind.startswith("slope") else "cm"; shown = level if unit == "deg" else level * 100
title = f"{family} policy ({pathlib.Path(model_path).parent.name}), {kind} {shown:g} {unit}, forward command {speed} m/s" + (" + heading hold" if hold else " (no steering)")
FEET = ["FR", "FL", "RR", "RL"]


def episode(viewer, ep):
    obs, _ = env.reset(seed=seed + ep, options={"yaw": 0.0}); xm = 0.0; hist = []
    for t in range(1000):
        if viewer is not None and not viewer.is_running(): return "closed"
        t0 = time.time(); yaw = float(np.arctan2(raw.data.xmat[1].reshape(3, 3)[1, 0], raw.data.xmat[1].reshape(3, 3)[0, 0]))
        raw.command = np.array([speed, float(np.clip(-2.0 * yaw, -0.5, 0.5)) if hold else 0.0]); obs[-4:-2] = raw.command; obs[-2:] = raw._clock()
        obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); x, y = float(raw.data.qpos[0]), float(raw.data.qpos[1]); xm = max(xm, x); hist.append(x)
        status = ""
        if xm >= X_GOAL: status = "SUCCESS: crossed (x >= 6 m)"
        elif term: status = f"FAILED: {'left the course sideways' if raw.termination_reason == 'left_course' else 'fell over'}"
        elif t >= 150 and hist[-1] - hist[-150] < 0.3 and x > 0.5: status = "stuck? (less than 0.3 m of progress in the last 3 s)"
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; viewer.cam.trackbodyid = 1; viewer.cam.distance = 2.2; viewer.cam.elevation = -12; viewer.cam.azimuth = 90
            if t % 5 == 0:
                down = raw._foot_contacts(); feet = "  ".join(f"{n}:{'down' if c else 'UP'}" for n, c in zip(FEET, down))
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, title, f"x {x:.2f} m (goal 6)   y {y:+.2f} m (course edge +-1.4)   speed {info['vx']:.2f} m/s   time {t * 0.02:.1f} s\nfeet: {feet}\n{status}"))
            viewer.sync(); time.sleep(max(0.0, 0.02 - (time.time() - t0)))
        if xm >= X_GOAL or term:
            if viewer is not None: time.sleep(1.5)
            return status
    return "TIMEOUT (20 s, did not reach 6 m)"


print(title, flush=True)
if "--no-window" in sys.argv:
    for ep in range(3): print(f"episode {ep}: {episode(None, ep)}", flush=True)
else:
    with mujoco.viewer.launch_passive(raw.model, raw.data) as v:
        ep = 0
        while v.is_running():
            res = episode(v, ep); print(f"episode {ep}: {res}", flush=True); ep += 1
            if res == "closed": break
