"""Viewer for a walking model: plays a fixed script of commands in the MuJoCo viewer (real time); the window shows the current command, the actual speeds, which feet touch the floor and, for the
clock model, which feet the metronome says should be in the air. Works with both models (picks the environment from the model's observation size: 47 = Go1WalkingEnv, 49 = Go1WalkingClockEnv).
Usage: python 03_view_walk.py [model.zip] [--seed 3000] [--sine] [--no-window]
Default model: outputs/walk3_clock_1M_seed0/checkpoints/model_1000000.zip. Command script (4 s each): forward 0.4, forward 0.6, forward 0.4 + turn 0.4, turn in place 0.5, stand. Loops until the window is closed."""
import sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch; torch.set_num_threads(1)
import mujoco, mujoco.viewer
from stable_baselines3 import PPO
from racevla.envs.go1_walking import Go1WalkingEnv
from racevla.envs.go1_walking_clock import Go1WalkingClockEnv
from racevla.envs.go1_walking_sine import Go1WalkingSineEnv
from racevla.envs.wrappers import FixedObsNormalize

args = [a for a in sys.argv[1:] if not a.startswith("--")]
path = pathlib.Path(args[0]) if args else ROOT / "outputs/walk3_clock_1M_seed0/checkpoints/model_1000000.zip"
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 3000
SCRIPT = [(0.4, 0.0, "walk forward 0.4"), (0.6, 0.0, "walk forward 0.6"), (0.4, 0.4, "forward 0.4 + turn 0.4"), (0.0, 0.5, "turn in place 0.5"), (0.0, 0.0, "stand still")]
SECONDS = 4.0
model = PPO.load(path, device="cpu"); clock = model.observation_space.shape[0] == 49
sine = "--sine" in sys.argv                                    # option B models also have 49 inputs: tell the viewer with --sine
raw_env = (Go1WalkingSineEnv if sine else Go1WalkingClockEnv if clock else Go1WalkingEnv)(push=False); env = FixedObsNormalize(raw_env)
raw_env._sample_command = lambda: None                         # the script sets the command; the environment must not resample it
FEET = ["FR", "FL", "RR", "RL"]


def set_command(obs, cmd):
    raw_env.command = np.array(cmd, float)
    if clock: obs[-4:-2] = raw_env.command; obs[-2:] = raw_env._clock()       # observation = 45 robot + 2 command + 2 clock
    else: obs[-2:] = raw_env.command


def run(viewer=None, max_steps=None):
    obs, _ = env.reset(seed=seed); step = 0
    while viewer is None or viewer.is_running():
        for vx, wz, label in SCRIPT:
            for t in range(int(SECONDS * 50)):
                if viewer is not None and not viewer.is_running(): return
                t0 = time.time(); set_command(obs, (vx, wz))
                obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); set_command(obs, (vx, wz)); step += 1
                if viewer is not None and step % 5 == 0:
                    down = raw_env._foot_contacts(); feet = "  ".join(f"{n}:{'down' if c else 'UP'}" for n, c in zip(FEET, down))
                    sched = ("\nmetronome says in the air: " + (" ".join(n for n, s in zip(FEET, raw_env.desired_swing()) if s) or "(none)")) if clock and raw_env._moving() else ""
                    viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, f"command: {label}", f"actual: vx {info['vx']:.2f} m/s, turn {info['wz']:.2f} rad/s\nfeet: {feet}{sched}"))
                if viewer is not None: viewer.sync(); time.sleep(max(0.0, 0.02 - (time.time() - t0)))
                if term:
                    print(f"fell ({info['termination_reason']}) during '{label}'", flush=True); obs, _ = env.reset(seed=seed + step); break
            if max_steps and step >= max_steps: return
        print("--- script finished, repeating", flush=True)


print(f"model {path.name}: {'clock model (49 inputs)' if clock else 'walking model (47 inputs)'}", flush=True)
if "--no-window" in sys.argv: run(None, max_steps=1000); print("headless run ok")
else:
    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as v: run(v)
