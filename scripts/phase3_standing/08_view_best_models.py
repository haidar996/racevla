"""Short demo: plays the three final standing models one after another in the MuJoCo viewer (real time), each for SECONDS seconds on the SAME start state
and the SAME pushes, so they can be compared by eye. The model name is shown in the top-left of the window. Loops until the window is closed.
Usage: python 08_view_best_models.py [--seed 4001] [--seconds 12] [--no-window]     (--no-window = headless check: prints the result of each model and exits)
TD3 models get the 10-step zero-action warm-up (see models/standing_policy_final.md); the PPO model does not need it."""
import argparse, sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch; torch.set_num_threads(1)
import mujoco, mujoco.viewer
from stable_baselines3 import PPO, TD3
from racevla.envs.go1_standing import Go1StandingEnv
from racevla.envs.wrappers import FixedObsNormalize

MODELS = [("TD3 seed 0  (primary, 0.2 s warm-up)", TD3, "standing_policy_td3_seed0.zip", 10),
          ("TD3 seed 1  (backup, 0.2 s warm-up)", TD3, "standing_policy_td3_seed1_backup.zip", 10),
          ("PPO wide-push seed 2  (no warm-up)", PPO, "standing_policy_ppo_wide_seed2_nowarmup.zip", 0)]
ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=4001); ap.add_argument("--seconds", type=float, default=12.0); ap.add_argument("--no-window", action="store_true")
a = ap.parse_args(); STEPS = int(a.seconds * 50)

raw_env = Go1StandingEnv(push=True); env = FixedObsNormalize(raw_env)        # the window shares raw_env.data, so it shows every step
loaded = [(label, cls.load(ROOT / "models" / f, device="cpu"), k) for label, cls, f, k in MODELS]


def play(label, model, k, viewer=None):
    obs, _ = env.reset(seed=a.seed); ret, pushes, reason = 0.0, [], "survived"
    if viewer: viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, label, f"seed {a.seed}  |  pushes shown in the terminal"))
    for t in range(STEPS):
        if viewer is not None and not viewer.is_running(): return None
        t0 = time.time()
        action = np.zeros(12, np.float32) if t < k else model.predict(obs, deterministic=True)[0]
        obs, r, term, trunc, info = env.step(action); ret += r
        if info["push_kick"] > 0: pushes.append(round(info["push_kick"], 2)); print(f"   push {info['push_kick']:.2f} m/s at {t / 50:.1f} s", flush=True)
        if viewer is not None: viewer.sync(); time.sleep(max(0.0, 0.02 - (time.time() - t0)))
        if term: reason = info["termination_reason"]; break
    return f"{label}: {'FELL (' + reason + ') at ' + format((t + 1) / 50, '.1f') + ' s' if reason != 'survived' else 'stayed up'}, return {ret:.0f}, pushes {pushes}"


if a.no_window:
    for label, model, k in loaded: print(play(label, model, k))
else:
    with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as v:
        while v.is_running():
            for label, model, k in loaded:
                print(f"--- now playing: {label}", flush=True); res = play(label, model, k, v)
                if res is None: break
                print(res, flush=True); time.sleep(1.5)
