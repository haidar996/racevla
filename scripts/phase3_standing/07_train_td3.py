"""Phase 3: TD3 on the push task -- same environment, evaluation protocol, replay settings and output format as 04_train_sac.py (the stable 'v2' SAC recipe:
1M buffer, learning rate decayed to 0, 5-step returns, 4 gradient steps per vec-env step), so the algorithms are compared on equal terms. Differences are TD3's own:
deterministic actor + Gaussian exploration noise, twin critics, delayed actor updates, target-policy smoothing. Checkpoint saved at every evaluation.
Usage: python 07_train_td3.py [--steps 1000000] [--eval-every 50000] [--run-name td3_push_1M_seed0] [--seed 0] [--noise-sigma 0.1] [--no-viewer]
Writes to outputs/<run-name>/: latest_model.zip, best_model.zip, checkpoints/, evaluations.npz, eval_curve.csv"""
import argparse, os, subprocess, sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from racevla.envs.go1_standing import Go1StandingEnv, MAX_EPISODE_STEPS
from racevla.envs.wrappers import FixedObsNormalize

# --- shared with PPO, for a fair comparison ------------------------------------------------------
N_ENVS = 4
EVAL_SEEDS = list(range(1000, 1030))   # 30 fixed episodes, same ones PPO was scored on
SAVE_LATEST_EVERY = 10_000             # timesteps between checkpoints for the viewer

# --- TD3-specific (replay settings identical to the SAC v2 recipe) ----------------------------------
LEARNING_RATE_START = 3e-4   # linearly decayed to 0, as in SAC v2 (SB3's TD3 default would be 1e-3 constant)
BUFFER_SIZE = 1_000_000  # whole step budget, nothing dropped
N_STEPS = 5              # 5-step returns, as in SAC v2
LEARNING_STARTS = 5_000  # random-action steps before training starts
BATCH_SIZE = 256
TAU = 0.005
TRAIN_FREQ = 1
GRADIENT_STEPS = 4       # matched to N_ENVS, as in SAC
POLICY_DELAY = 2         # actor and target networks update every 2nd critic update (TD3 default)
TARGET_POLICY_NOISE = 0.2    # smoothing noise on the target action, in [-1, 1] action units (TD3 default)
TARGET_NOISE_CLIP = 0.5
NET_ARCH = [256, 256]    # same as SAC
NOISE_SIGMA_DEFAULT = 0.1    # exploration noise std in [-1, 1] action units = 0.05 rad at the 0.5 rad action scale


def lr_schedule(progress_remaining): return LEARNING_RATE_START * progress_remaining   # SB3 passes 1.0 -> 0.0 over training


def make_env(max_steps=None):
    """max_steps caps the episode length (training only; evaluation always uses the full 1000). A cap counts as a time-out, so SB3 still bootstraps through it."""
    env = FixedObsNormalize(Go1StandingEnv(push=True))
    return TimeLimit(env, max_steps) if max_steps else env


class FixedSeedCycle(gym.Wrapper):
    """reset() with no seed walks through EVAL_SEEDS, so every evaluation sees the same fixed start states and pushes."""
    def __init__(self, env): super().__init__(env); self.i = 0
    def reset(self, *, seed=None, options=None):
        if seed is None: seed = EVAL_SEEDS[self.i % len(EVAL_SEEDS)]; self.i += 1
        return self.env.reset(seed=seed, options=options)


class FixedSeedEvalCallback(EvalCallback):
    """Fixed-seed evaluation as before, and (v3+) also keeps the model from EVERY evaluation in ckpt_dir/model_<steps>.zip, so the
    final pick can be made on a large held-out seed set (05_eval_many_seeds.py) instead of trusting the 30-episode 'best_model.zip'."""
    def __init__(self, *args, ckpt_dir=None, **kwargs): super().__init__(*args, **kwargs); self.ckpt_dir = ckpt_dir

    def _on_step(self) -> bool:
        is_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if is_eval: self.eval_env.envs[0].i = 0            # episode k of every evaluation uses EVAL_SEEDS[k]
        result = super()._on_step()
        if is_eval and self.ckpt_dir is not None: self.model.save(self.ckpt_dir / f"model_{self.num_timesteps}.zip")
        return result


class SaveLatest(BaseCallback):
    """Writes latest_model.zip every `every` timesteps (atomically), for the viewer process to pick up."""
    def __init__(self, path, every): super().__init__(); self.path, self.every, self.last = path, every, 0
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last >= self.every: save_atomic(self.model, self.path); self.last = self.num_timesteps
        return True


def save_atomic(model, path):
    tmp = path.with_name("latest_tmp.zip"); model.save(tmp); os.replace(tmp, path)   # rename is atomic: no half-written file


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000); ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--run-name", default="td3_push_1M"); ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="training seed -- network init, action noise, env resets")
    ap.add_argument("--noise-sigma", type=float, default=NOISE_SIGMA_DEFAULT, help="std of the Gaussian exploration noise added to the actor output")
    ap.add_argument("--train-ep-steps", type=int, default=None, help="cap TRAINING episodes at this many steps; default None = full 1000")
    a = ap.parse_args()
    torch.set_num_threads(1)                                # leave the other CPU threads to the 4 parallel simulators
    run_dir = ROOT / "outputs" / a.run_name
    if (run_dir / "eval_curve.csv").exists(): sys.exit(f"{run_dir} already has results; pick another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)     # one model per evaluation, ~3.6 MB each

    env = make_vec_env(lambda: make_env(a.train_ep_steps), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=a.seed)
    eval_env = DummyVecEnv([lambda: FixedSeedCycle(Monitor(make_env()))])
    noise = NormalActionNoise(mean=np.zeros(12), sigma=a.noise_sigma * np.ones(12))     # SB3 vectorises it across the 4 environments
    model = TD3("MlpPolicy", env, learning_rate=lr_schedule, buffer_size=BUFFER_SIZE, learning_starts=LEARNING_STARTS, n_steps=N_STEPS,
                batch_size=BATCH_SIZE, tau=TAU, train_freq=TRAIN_FREQ, gradient_steps=GRADIENT_STEPS, action_noise=noise, policy_delay=POLICY_DELAY,
                target_policy_noise=TARGET_POLICY_NOISE, target_noise_clip=TARGET_NOISE_CLIP, seed=a.seed, device="cpu", verbose=1,
                policy_kwargs={"net_arch": NET_ARCH})
    latest = run_dir / "latest_model.zip"
    save_atomic(model, latest)                              # untrained policy, so the viewer has something to show right away

    viewer = None
    if not a.no_viewer:
        viewer = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).parent / "watch_latest.py"),
                                   str(run_dir), "latest_model.zip", "--algo", "td3"],
                                  stdout=open(run_dir / "viewer.log", "w"), stderr=subprocess.STDOUT,
                                  preexec_fn=lambda: os.nice(15))    # low priority: training wins any CPU contention
    callbacks = [SaveLatest(latest, SAVE_LATEST_EVERY),
                 FixedSeedEvalCallback(eval_env, n_eval_episodes=len(EVAL_SEEDS), eval_freq=a.eval_every // N_ENVS, deterministic=True,
                                       best_model_save_path=str(run_dir), log_path=str(run_dir), ckpt_dir=ckpt_dir)]   # eval_freq counts vec-env steps

    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=callbacks)
    save_atomic(model, latest)
    print(f"\ntrained {a.steps} steps in {(time.time() - t0) / 60:.1f} min")

    ev = np.load(run_dir / "evaluations.npz")               # timesteps (n_evals,), results / ep_lengths (n_evals, 30)
    with open(run_dir / "eval_curve.csv", "w") as f:
        f.write(f"timesteps,mean_return,mean_length,fell_out_of_{len(EVAL_SEEDS)}\n")
        for ts, ret, ln in zip(ev["timesteps"], ev["results"], ev["ep_lengths"]):
            f.write(f"{ts},{ret.mean():.1f},{ln.mean():.1f},{int((ln < MAX_EPISODE_STEPS).sum())}\n")   # length < 1000 means it fell
    print((run_dir / "eval_curve.csv").read_text())
    env.close(); eval_env.close()
