"""Phase 3: SAC on the push task -- same environment, evaluation protocol and output format as
03_train_ppo.py (see outputs/report/sac_plan.pdf for the full reasoning behind every hyperparameter
below). Periodic evaluation on fixed seeds, with an optional live MuJoCo viewer in a second process.
Usage: python 04_train_sac.py [--steps 1000000] [--eval-every 50000] [--run-name sac_push_1M] [--no-viewer]
Writes to outputs/<run-name>/: latest_model.zip, best_model.zip, evaluations.npz, eval_curve.csv, viewer.log"""
import argparse, os, subprocess, sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import torch
from stable_baselines3 import SAC
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

# --- SAC-specific, see outputs/report/sac_plan.pdf for the reasoning behind each --------------------
LEARNING_RATE_START = 3e-4   # v2: linearly decayed to 0 (v1 was constant and oscillated between 7% and 100% falls after 500k)
BUFFER_SIZE = 1_000_000  # v2: equals the whole step budget, so nothing is ever dropped (v1: 300k) -- keeps rare fall transitions forever
N_STEPS = 5              # v2: 5-step returns (0.1 s): a fall within 5 steps is seen directly instead of via slow 1-step bootstrapping
LEARNING_STARTS = 5_000  # random-policy steps collected before training starts
BATCH_SIZE = 256         # SAC default; cheap to sample from a replay buffer, unlike PPO's fresh rollouts
TAU = 0.005              # soft target-network update rate
TRAIN_FREQ = 1           # a gradient update opportunity every environment step
GRADIENT_STEPS = 4       # matched to N_ENVS, so updates keep pace with experience collection
ENT_COEF = "auto"        # automatic entropy tuning, replaces PPO's hand-tuned log_std_init
TARGET_UPDATE_INTERVAL = 1
USE_SDE = False          # matches every PPO run (no state-dependent exploration)
NET_ARCH = [256, 256]    # SAC's own default; deliberately not shrunk to PPO's 64x64 (see sac_plan.pdf)


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
    ap.add_argument("--run-name", default="sac_push_1M"); ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="training seed -- network init, action noise, env resets")
    ap.add_argument("--train-ep-steps", type=int, default=None, help="v3: cap TRAINING episodes at this many steps (e.g. 250) so episode starts are a bigger share of the replay buffer; default None = full 1000 like v2")
    a = ap.parse_args()
    torch.set_num_threads(1)                                # leave the other CPU threads to the 4 parallel simulators
    run_dir = ROOT / "outputs" / a.run_name
    if (run_dir / "eval_curve.csv").exists(): sys.exit(f"{run_dir} already has results; pick another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)     # one model per evaluation, ~3.6 MB each

    env = make_vec_env(lambda: make_env(a.train_ep_steps), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=a.seed)
    eval_env = DummyVecEnv([lambda: FixedSeedCycle(Monitor(make_env()))])
    model = SAC("MlpPolicy", env, learning_rate=lr_schedule, buffer_size=BUFFER_SIZE, learning_starts=LEARNING_STARTS, n_steps=N_STEPS,
                batch_size=BATCH_SIZE, tau=TAU, train_freq=TRAIN_FREQ, gradient_steps=GRADIENT_STEPS, ent_coef=ENT_COEF,
                target_update_interval=TARGET_UPDATE_INTERVAL, use_sde=USE_SDE, seed=a.seed, device="cpu", verbose=1,
                policy_kwargs={"net_arch": NET_ARCH})
    latest = run_dir / "latest_model.zip"
    save_atomic(model, latest)                              # untrained policy, so the viewer has something to show right away

    viewer = None
    if not a.no_viewer:
        viewer = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).parent / "watch_latest.py"),
                                   str(run_dir), "latest_model.zip", "--algo", "sac"],
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
