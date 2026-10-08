"""Viewer for a running model (Go1RunningSineEnv): plays a fixed script of commands in the MuJoCo viewer (real time). The window shows the command, the actual speeds, which feet touch the floor and a FLIGHT label
while all four feet are off the floor, plus the share of flight so far. Usage: python 06_view_running.py [model.zip] [--seed 3000] [--no-window]
Default model: models/running_policy_2p5_seed0.zip. Script (4 s each): walk 0.6, run 1.5, run 2.0, run 2.5, run 2.0 + turn 0.4, run 2.5, stand. Loops until the window is closed."""
import sys, json, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch; torch.set_num_threads(1)
import mujoco, mujoco.viewer
from stable_baselines3 import PPO
from racevla.envs.go1_running_sine import Go1RunningSineEnv
from racevla.envs.wrappers import FixedObsNormalize

args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = pathlib.Path(args[0]) if args else ROOT / "models/running_policy_2p5_seed0.zip"
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 3000
if "--seed" in sys.argv and args and args[0] == sys.argv[sys.argv.index("--seed") + 1]: path = ROOT / "models/running_policy_2p5_seed0.zip"
SCRIPT = [(0.6, 0.0, "walk 0.6"), (1.5, 0.0, "run 1.5"), (2.0, 0.0, "run 2.0"), (2.5, 0.0, "run 2.5"), (2.0, 0.4, "run 2.0 + turn 0.4"), (2.5, 0.0, "run 2.5"), (0.0, 0.0, "stand still")]
SECONDS = 4.0
table = json.load(open(ROOT / "outputs/analysis/running/gait_table.json"))
model = PPO.load(path, device="cpu")
raw_env = Go1RunningSineEnv(push=False, gait={"table": table}); env = FixedObsNormalize(raw_env)
raw_env._sample_command = lambda: None                         # the script sets the command; the environment must not resample it
FEET = ["FR", "FL", "RR", "RL"]


def set_command(obs, cmd):
    raw_env.command = np.array(cmd, float); obs[-4:-2] = raw_env.command; obs[-2:] = raw_env._clock()      # observation = 45 robot + 2 command + 2 clock


def run(viewer=None, max_steps=None):
    obs, _ = env.reset(seed=seed); step = 0; air = total = 0
    while viewer is None or viewer.is_running():
        for vx, wz, label in SCRIPT:
            for t in range(int(SECONDS * 50)):
                if viewer is not None and not viewer.is_running(): return
                t0 = time.time(); set_command(obs, (vx, wz))
                obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); set_command(obs, (vx, wz)); step += 1
                down = raw_env._foot_contacts()
                if vx >= 1.0: total += 1; air += int(down.sum() == 0)
                if viewer is not None and step % 5 == 0:
                    feet = "  ".join(f"{n}:{'down' if c else 'UP'}" for n, c in zip(FEET, down)); flight = "   *** FLIGHT ***" if down.sum() == 0 else ""
                    share = f"\nflight share while running (cmd >= 1): {100 * air / max(total, 1):.0f} %"
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, f"command: {label}{flight}", f"actual: vx {info['vx']:.2f} m/s, turn {info['wz']:.2f} rad/s\nfeet: {feet}{share}"))
                if viewer is not None: viewer.sync(); time.sleep(max(0.0, 0.02 - (time.time() - t0)))
                if term:
                    print(f"fell ({info['termination_reason']}) during '{label}'", flush=True); obs, _ = env.reset(seed=seed + step); break
            if max_steps and step >= max_steps: print(f"flight share while running: {100 * air / max(total, 1):.0f} %"); return
        print("--- script finished, repeating", flush=True)


print(f"model {path}", flush=True)
if "--no-window" in sys.argv: run(None, max_steps=1400); print("headless run ok")
else:
    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as v: run(v)
