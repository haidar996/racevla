"""Phase 7: PPO for the recovery skill (stand up from a very random, always right-side-up start). Environment: Go1RecoveryEnv (racevla/envs/go1_recovery.py): action range +-1.5 rad around home, 45-D observation
(with the previous action), 10 s episodes, reward = progress terms - smoothness / torque penalties.
Curriculum: the share of the three start classes (0 near standing, 1 crouched, 2 sprawled) moves linearly from (0.5, 0.35, 0.15) to (0.2, 0.4, 0.4) over the first half of training.
PPO: n_steps 512, 10 epochs, clip 0.2, gae 0.95, lr 3e-4 constant, 4 envs, log_std_init -2.3 (std 0.1 = 0.15 rad of noise on the +-1.5 rad range; -1.0 made the robot topple within 40 steps from pure noise).
Every --eval-every steps: 8 fixed episodes per start class (seeds 1000..), scored on SUCCESS (stand_ok during the last 2 s), flips, time to stand (first step from which stand_ok holds for 1 s), return.
Usage: python 02_train_recovery_ppo.py --run-name rec1_4M_seed0 --seed 0 --steps 4000000
Writes outputs/<run-name>/: eval_recovery.csv, checkpoints/model_<steps>.zip, best_model.zip (highest mean success, ties by return), latest_model.zip"""
import argparse, sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from racevla.envs.go1_recovery import Go1RecoveryEnv, MAX_STEPS
from racevla.envs.wrappers import FixedObsNormalize

N_ENVS = 4
EVAL_SEEDS = list(range(1000, 1008))                       # 8 episodes per start class per evaluation
N_STEPS, N_EPOCHS, CLIP_RANGE, GAE_LAMBDA, LEARNING_RATE, LOG_STD_INIT = 512, 10, 0.2, 0.95, 3e-4, -2.3
PROBS_START, PROBS_END = np.array([0.5, 0.35, 0.15]), np.array([0.2, 0.4, 0.4])
STAND_HOLD = 50                                            # steps (1 s) of stand_ok in a row that count as "stood up"; success = stand_ok during the last 100 steps (2 s)


def make_env(): return FixedObsNormalize(Go1RecoveryEnv())




def evaluate(model, env, seeds):
    raw = env.unwrapped; out = {}
    for cls in (0, 1, 2):
        succ, flips, t_stand, rets, jerk = 0, 0, [], [], []
        for s in seeds:
            obs, _ = env.reset(seed=s, options={"class": cls}); ok, ret, term, prev = [], 0.0, False, None
            for t in range(MAX_STEPS):
                a = model.predict(obs, deterministic=True)[0]; obs, r, term, trunc, info = env.step(a); ret += r; ok.append(info["stand_ok"])
                if term or trunc: break
            ok = np.array(ok); n = len(ok)
            succ += int(n == MAX_STEPS and ok[-100:].all()); flips += int(term)
            run = np.convolve(ok, np.ones(STAND_HOLD), mode="valid") if n >= STAND_HOLD else np.array([])
            first = np.argmax(run >= STAND_HOLD) if len(run) and (run >= STAND_HOLD).any() else None
            if first is not None: t_stand.append(first * 0.02)
            rets.append(ret)
        out[cls] = {"success": succ / len(seeds), "flips": flips / len(seeds), "t_stand": float(np.mean(t_stand)) if t_stand else float("nan"), "stood": len(t_stand) / len(seeds), "return": float(np.mean(rets))}
    return out


class RecoveryCallback(BaseCallback):
    def __init__(self, eval_env, run_dir, every, total):
        super().__init__(); self.eval_env, self.run_dir, self.every, self.total, self.last, self.last_cur, self.best = eval_env, run_dir, every, total, 0, -1, (-1.0, -np.inf)
        self.csv = run_dir / "eval_recovery.csv"
        self.csv.write_text("timesteps," + ",".join(f"{k}_c{c}" for c in (0, 1, 2) for k in ("success", "flips", "stood", "t_stand", "return")) + ",mean_success,probs\n")
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    def _probs(self):
        p = min(1.0, self.num_timesteps / (0.5 * self.total)); return (1 - p) * PROBS_START + p * PROBS_END

    def _on_step(self) -> bool:
        if self.num_timesteps // 50_000 != self.last_cur:
            self.last_cur = self.num_timesteps // 50_000; self.training_env.env_method("set_class_probs", self._probs())
        if self.num_timesteps - self.last < self.every: return True
        self.last = self.num_timesteps; m = evaluate(self.model, self.eval_env, EVAL_SEEDS); ms = float(np.mean([m[c]["success"] for c in (0, 1, 2)])); mr = float(np.mean([m[c]["return"] for c in (0, 1, 2)]))
        with open(self.csv, "a") as f:
            f.write(f"{self.num_timesteps}," + ",".join(f"{m[c][k]:.3f}" for c in (0, 1, 2) for k in ("success", "flips", "stood", "t_stand", "return")) + f",{ms:.3f},{'/'.join(f'{x:.2f}' for x in self._probs())}\n")
        self.model.save(self.run_dir / "checkpoints" / f"model_{self.num_timesteps}.zip")
        if (ms, mr) > self.best: self.best = (ms, mr); self.model.save(self.run_dir / "best_model.zip")
        print(f"EVAL {self.num_timesteps}: " + " | ".join(f"class {c}: success {m[c]['success']:.2f} flips {m[c]['flips']:.2f} t_stand {m[c]['t_stand']:.1f}s ret {m[c]['return']:.0f}" for c in (0, 1, 2)) + f" | probs {self._probs().round(2)}", flush=True)
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4_000_000); ap.add_argument("--eval-every", type=int, default=200_000)
    ap.add_argument("--run-name", required=True); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(1)
    run_dir = ROOT / "outputs" / a.run_name
    if (run_dir / "eval_recovery.csv").exists(): sys.exit(f"{run_dir} already has results; pick another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=a.seed)
    env.env_method("set_class_probs", PROBS_START)
    model = PPO("MlpPolicy", env, n_steps=N_STEPS, n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE, clip_range=CLIP_RANGE, gae_lambda=GAE_LAMBDA, seed=a.seed, device="cpu", verbose=1, policy_kwargs={"log_std_init": LOG_STD_INIT})
    t0 = time.time(); model.learn(total_timesteps=a.steps, callback=RecoveryCallback(make_env(), run_dir, a.eval_every, a.steps))
    model.save(run_dir / "latest_model.zip"); print(f"\ntrained {a.steps} steps in {(time.time() - t0) / 60:.1f} min"); env.close()
