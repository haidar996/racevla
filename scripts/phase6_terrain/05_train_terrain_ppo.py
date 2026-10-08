"""Phase 6: PPO for blind terrain walking (Skill A) with ONE mixed training: every episode a random terrain type and level (racevla/envs/go1_terrain_walk.py), curriculum per type.
Warm start from the packaged walking policy (--init); PPO settings as in the walking and running runs (n_steps 512, 10 epochs, clip 0.2, gae 0.95, lr 3e-4, 4 envs), action noise reset to log_std -0.8.
Curriculum: every 25k steps the training-episode statistics of each type's top unlocked level are checked; with at least 15 episodes and >= 80 % success the next level of that type is unlocked (up to CAP).
Evaluation every --eval-every steps (and once before training) on FIXED terrains, 5 episodes each, deterministic policy, commands vx ~ U(0.4, 0.8) with heading hold: flat, rough 6 cm, rough 10 cm, slope up 6 / 9 / 15 deg,
slope down 15 deg, stairs up 4 cm, stairs down 6 cm. Success = x reaches 6 m.
Usage: python 05_train_terrain_ppo.py --run-name terr_pilot_seed0 --seed 0 --steps 500000 --init models/walking_policy_sine_seed1.zip
Writes outputs/<run>/: eval_terrain.csv, curriculum.log, checkpoints/model_<steps>.zip, latest_model.zip"""
import argparse, sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from racevla.envs.go1_terrain_walk import Go1TerrainWalkEnv, CAP
from racevla.envs.terrain import LEVELS
from racevla.envs.wrappers import FixedObsNormalize

N_ENVS, N_STEPS, N_EPOCHS, CLIP_RANGE, GAE_LAMBDA, LEARNING_RATE, LOG_STD_INIT = 4, 512, 10, 0.2, 0.95, 3e-4, -0.8
EVAL_TERRAINS = [("flat", ("rough", 0.0)), ("rough 6cm", ("rough", 0.06)), ("rough 10cm", ("rough", 0.10)), ("slope up 6", ("slope_up", 6)), ("slope up 9", ("slope_up", 9)), ("slope up 15", ("slope_up", 15)),
                 ("slope down 15", ("slope_down", 15)), ("stairs up 4cm", ("stairs_up", 0.04)), ("stairs down 6cm", ("stairs_down", 0.06))]
UNLOCK_MIN_EPISODES, UNLOCK_SUCCESS = 15, 0.8


def make_env(lift=None): return FixedObsNormalize(Go1TerrainWalkEnv(lift=lift))        # lift = peak foot lift of the reference gait in m (None = the default 0.08)


def evaluate(model, env, n_ep=5):
    out = {}
    for name, terrain in EVAL_TERRAINS:
        ok, tm = 0, []
        for s in range(n_ep):
            obs, _ = env.reset(seed=1000 + s, options={"terrain": terrain, "terrain_seed": 1000 + s})
            for t in range(1000):
                obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
                if term or trunc: break
            if info.get("success"): ok += 1; tm.append((t + 1) * 0.02)
        out[name] = ok / n_ep
    return out


class TerrainCallback(BaseCallback):
    def __init__(self, eval_env, run_dir, every):
        super().__init__(); self.eval_env, self.run_dir, self.every, self.last_eval, self.last_cur, self.acc = eval_env, run_dir, every, -1, -1, {}
        self.csv = run_dir / "eval_terrain.csv"; self.csv.write_text("timesteps," + ",".join(n for n, _ in EVAL_TERRAINS) + ",mean,tops\n"); self.top = {k: 0 for k in LEVELS}
        (run_dir / "checkpoints").mkdir(exist_ok=True); self.log = open(run_dir / "curriculum.log", "a")

    def _write_eval(self, label):
        m = evaluate(self.model, self.eval_env); mean = float(np.mean(list(m.values())))
        with open(self.csv, "a") as f: f.write(f"{label}," + ",".join(f"{m[n]:.2f}" for n, _ in EVAL_TERRAINS) + f",{mean:.3f},{'/'.join(str(self.top[k]) for k in LEVELS)}\n")
        print(f"EVAL {label}: " + " | ".join(f"{n} {m[n]:.1f}" for n, _ in EVAL_TERRAINS) + f" | mean {mean:.2f} | unlocked tops {self.top}", flush=True)

    def _on_training_start(self):
        self._write_eval("0")

    def _on_step(self) -> bool:
        if self.num_timesteps // 25_000 != self.last_cur:
            self.last_cur = self.num_timesteps // 25_000
            for st in self.training_env.env_method("pop_stats"):
                for key, (n, s) in st.items(): a = self.acc.setdefault(key, [0, 0]); a[0] += n; a[1] += s
            changed = False
            for kind in LEVELS:
                idx = self.top[kind]; key = (kind, LEVELS[kind][idx]); n, s = self.acc.get(key, [0, 0])
                if idx < CAP[kind] and n >= UNLOCK_MIN_EPISODES and s / n >= UNLOCK_SUCCESS:
                    self.top[kind] += 1; changed = True; self.acc.pop((kind, LEVELS[kind][self.top[kind]]), None)
                    msg = f"{self.num_timesteps}: {kind} level {key[1]} succeeded {s}/{n} -> unlocked {LEVELS[kind][self.top[kind]]}"; print("CURRICULUM", msg, flush=True); self.log.write(msg + "\n"); self.log.flush()
            if changed: self.training_env.env_method("set_top", self.top)
        if self.num_timesteps // self.every != self.last_eval and self.num_timesteps >= self.every:
            self.last_eval = self.num_timesteps // self.every; self._write_eval(str(self.num_timesteps))
            self.model.save(self.run_dir / "checkpoints" / f"model_{self.num_timesteps}.zip")
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500_000); ap.add_argument("--eval-every", type=int, default=100_000)
    ap.add_argument("--run-name", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--init", required=True); ap.add_argument("--lift", type=float, default=None)
    a = ap.parse_args(); torch.set_num_threads(1)
    run_dir = ROOT / "outputs" / a.run_name
    if (run_dir / "eval_terrain.csv").exists(): sys.exit(f"{run_dir} already has results; pick another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(lambda: make_env(a.lift), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=a.seed)
    model = PPO.load(ROOT / a.init if not pathlib.Path(a.init).is_absolute() else a.init, env=env, device="cpu", seed=a.seed, custom_objects={"n_steps": N_STEPS, "n_epochs": N_EPOCHS, "learning_rate": LEARNING_RATE, "clip_range": lambda _: CLIP_RANGE})
    with torch.no_grad(): model.policy.log_std.fill_(LOG_STD_INIT)
    t0 = time.time(); model.learn(total_timesteps=a.steps, callback=TerrainCallback(make_env(a.lift), run_dir, a.eval_every), reset_num_timesteps=True)
    model.save(run_dir / "latest_model.zip"); print(f"\ntrained {a.steps} steps in {(time.time() - t0) / 60:.1f} min"); env.close()
