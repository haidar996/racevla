"""Phase 3: PPO on the push task, with periodic evaluation on fixed seeds and a live MuJoCo viewer in a second process.
Usage: python 03_train_ppo.py [--steps 500000] [--eval-every 25000] [--run-name ppo_push_500k] [--no-viewer]
Writes to outputs/<run-name>/: latest_model.zip, best_model.zip, evaluations.npz, eval_curve.csv, train.log, viewer.log"""
import argparse, os, subprocess, sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from racevla.envs.go1_standing import Go1StandingEnv, MAX_EPISODE_STEPS
from racevla.envs.wrappers import FixedObsNormalize

N_ENVS = 4
N_STEPS = 2048                 # per env per update -> 8192 samples per PPO update (was 1024/4096, doubled again; paired with a 2M-step run to compensate for fewer, larger updates)
CLIP_RANGE = 0.1                # PPO update ratio clip (was 0.2), more conservative updates
CLIP_RANGE_VF = 0.2             # NEW: same idea applied to the critic (was unconstrained) -- looser than CLIP_RANGE since
                                 # episode returns swing from ~5 (instant fall) to ~1000 (full survival), so the value
                                 # function needs more room to move per update than the policy does
GAE_LAMBDA = 0.9                # NEW: was SB3 default 0.95 -- trades a little bias for lower-variance advantage estimates
LOG_STD_INIT = -0.8            # exploration noise std = exp(-0.8) = 0.449  (was -1.0 / 0.368 in run 2, -2.0 / 0.135 in run 1)
N_EPOCHS = 5                   # passes over each batch before discarding it (was SB3 default 10 -- fewer, gentler updates)
LR_INITIAL = 3e-4              # decays linearly to 0 by the last training step (was constant)
EVAL_SEEDS = list(range(1000, 1030))   # 30 fixed episodes (was 10), so one fall swings the score ~3.3 pts instead of 10
SAVE_LATEST_EVERY = 10_000     # timesteps between checkpoints for the viewer


def make_env(push_max=None):
    """push_max (m/s) widens the TRAINING push range to (0.4, push_max); evaluation always uses the standard 0.4-1.2 so results stay comparable."""
    return FixedObsNormalize(Go1StandingEnv(push=True, push_kick_range=(0.4, push_max) if push_max else None))


def linear_schedule(initial_value):
    """SB3 calls this with progress_remaining going from 1.0 (start) to 0.0 (end of training) -- so the
    learning rate reaches exactly 0 on the very last step, regardless of how many total steps are set."""
    return lambda progress_remaining: progress_remaining * initial_value


class FixedSeedCycle(gym.Wrapper):
    """reset() with no seed walks through EVAL_SEEDS, so every evaluation sees the same 10 start states and pushes."""
    def __init__(self, env): super().__init__(env); self.i = 0
    def reset(self, *, seed=None, options=None):
        if seed is None: seed = EVAL_SEEDS[self.i % len(EVAL_SEEDS)]; self.i += 1
        return self.env.reset(seed=seed, options=options)


class FixedSeedEvalCallback(EvalCallback):
    """Fixed-seed evaluation, and (if ckpt_dir is given) the model from EVERY evaluation saved as ckpt_dir/model_<steps>.zip for later selection on many seeds."""
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
    ap.add_argument("--steps", type=int, default=500_000); ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--run-name", default="ppo_push_500k"); ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--seed", type=int, default=0, help="training seed -- network init, action noise, env resets")
    ap.add_argument("--push-max", type=float, default=None, help="widen the TRAINING push range to 0.4..this (m/s); default = standard 0.4..1.2")
    ap.add_argument("--n-steps", type=int, default=N_STEPS); ap.add_argument("--n-epochs", type=int, default=N_EPOCHS)
    ap.add_argument("--clip-range", type=float, default=CLIP_RANGE); ap.add_argument("--clip-range-vf", type=float, default=CLIP_RANGE_VF, help="negative = no value clipping")
    ap.add_argument("--gae-lambda", type=float, default=GAE_LAMBDA); ap.add_argument("--lr", choices=["linear", "const"], default="linear")
    a = ap.parse_args()
    torch.set_num_threads(1)                                # tiny 64x64 networks: extra threads only fight the simulators for cores
    run_dir = ROOT / "outputs" / a.run_name
    if (run_dir / "eval_curve.csv").exists(): sys.exit(f"{run_dir} already has results; pick another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)     # one model per evaluation

    env = make_vec_env(lambda: make_env(a.push_max), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=a.seed)
    eval_env = DummyVecEnv([lambda: FixedSeedCycle(Monitor(make_env()))])
    model = PPO("MlpPolicy", env, n_steps=a.n_steps, n_epochs=a.n_epochs, learning_rate=linear_schedule(LR_INITIAL) if a.lr == "linear" else LR_INITIAL,
                clip_range=a.clip_range, clip_range_vf=a.clip_range_vf if a.clip_range_vf >= 0 else None, gae_lambda=a.gae_lambda,
                seed=a.seed, device="cpu", verbose=1, policy_kwargs={"log_std_init": LOG_STD_INIT})
    latest = run_dir / "latest_model.zip"
    save_atomic(model, latest)                              # untrained policy, so the viewer has something to show right away

    viewer = None
    if not a.no_viewer:
        viewer = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).parent / "watch_latest.py"), str(run_dir)],
                                  stdout=open(run_dir / "viewer.log", "w"), stderr=subprocess.STDOUT,
                                  preexec_fn=lambda: os.nice(15))    # low priority: training wins any CPU contention
    callbacks = [SaveLatest(latest, SAVE_LATEST_EVERY),
                 FixedSeedEvalCallback(eval_env, n_eval_episodes=len(EVAL_SEEDS), eval_freq=a.eval_every // N_ENVS, deterministic=True,
                                       best_model_save_path=str(run_dir), log_path=str(run_dir), ckpt_dir=ckpt_dir)]   # eval_freq counts vec-env steps

    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=callbacks)
    save_atomic(model, latest)
    print(f"\ntrained {a.steps} steps in {(time.time() - t0) / 60:.1f} min")

    ev = np.load(run_dir / "evaluations.npz")               # timesteps (n_evals,), results / ep_lengths (n_evals, 10)
    with open(run_dir / "eval_curve.csv", "w") as f:
        f.write(f"timesteps,mean_return,mean_length,fell_out_of_{len(EVAL_SEEDS)}\n")
        for ts, ret, ln in zip(ev["timesteps"], ev["results"], ev["ep_lengths"]):
            f.write(f"{ts},{ret.mean():.1f},{ln.mean():.1f},{int((ln < MAX_EPISODE_STEPS).sum())}\n")   # length < 1000 means it fell
    print((run_dir / "eval_curve.csv").read_text())
    env.close(); eval_env.close()
