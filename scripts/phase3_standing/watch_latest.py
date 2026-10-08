"""Viewer process: plays push episodes with the newest checkpoint of a training run in the MuJoCo viewer (real time).
Usage: python watch_latest.py <run_dir> [checkpoint file, default latest_model.zip] [--raw] [--algo ppo|sac|td3]
Pass --raw for a model trained WITHOUT FixedObsNormalize (every run before the seed sweep, e.g.
ppo_push_500k_logstd-1, ppo_push_1M_scale0.5_logstd-0.8, models/standing_recovery_ppo_final.zip).
Feeding a normalized-trained model raw (unscaled) observations, or vice versa, makes it act on
inputs it never saw in training and its behavior becomes meaningless -- this must match training.
--algo picks which SB3 class to load with (default ppo); must match what actually trained the checkpoint."""
import sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch; torch.set_num_threads(1)                      # leave the CPU threads to the training processes
import mujoco.viewer
from stable_baselines3 import PPO, SAC, TD3
from racevla.envs.go1_standing import Go1StandingEnv, MAX_EPISODE_STEPS
from racevla.envs.wrappers import FixedObsNormalize

ALGO_CLASSES = {"ppo": PPO, "sac": SAC, "td3": TD3}
args = [a for a in sys.argv[1:] if not a.startswith("--")]
run_dir = pathlib.Path(args[0]); ckpt = run_dir / (args[1] if len(args) > 1 else "latest_model.zip")   # e.g. best_model.zip
raw = "--raw" in sys.argv
algo_name = sys.argv[sys.argv.index("--algo") + 1] if "--algo" in sys.argv else "ppo"
AlgoClass = ALGO_CLASSES[algo_name]
EVAL_SEEDS = range(1000, 1010)                              # same episodes the evaluation callback scores


def load():
    """Load the checkpoint; None if the file is missing or half-written."""
    try: return AlgoClass.load(ckpt, device="cpu"), ckpt.stat().st_mtime
    except Exception as e: print("could not load checkpoint:", e, flush=True); return None


while not ckpt.exists(): time.sleep(0.5)                    # training saves the first checkpoint before it starts
model, mtime = load()
raw_env = Go1StandingEnv(push=True)                      # kept unwrapped: the viewer needs its .model/.data directly
env = raw_env if raw else FixedObsNormalize(raw_env)      # what the policy actually sees
print(f"observation mode: {'RAW (unnormalized)' if raw else 'normalized (FixedObsNormalize)'}", flush=True)
with mujoco.viewer.launch_passive(raw_env.model, raw_env.data) as v:   # the window shares raw_env.data: it shows every step
    episode = 0
    while v.is_running():
        if ckpt.stat().st_mtime != mtime and (new := load()): model, mtime = new    # pick up a newer policy between episodes
        obs, _ = env.reset(seed=EVAL_SEEDS[episode % len(EVAL_SEEDS)]); ret = 0.0; reason = "survived"
        for t in range(MAX_EPISODE_STEPS):
            if not v.is_running(): break
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action); ret += r
            if info["push_kick"] > 0: print(f"   push {info['push_kick']:.2f} m/s at step {t}", flush=True)
            v.sync(); time.sleep(max(0.0, 0.02 - (time.time() - t0)))     # 0.02 s per policy step = real time
            if term: reason = info["termination_reason"]
            if term or trunc: break
        print(f"policy from {model.num_timesteps:>7d} training steps | episode {episode}: {t + 1} steps, return {ret:.1f}, {reason}", flush=True)
        episode += 1; time.sleep(0.5)
