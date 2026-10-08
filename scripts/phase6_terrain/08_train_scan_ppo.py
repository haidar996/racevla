"""Phase 6, Skill B: PPO for blind-plus-scan terrain walking: Go1ScanWalkEnv (racevla/envs/go1_scan_walk.py): 67-D observation = the terrain walking observation + an 18-number height scan, reference foot lift 12 cm, terrain mix with 50 % stairs,
stairs up AND down unlocked up to 15 cm, rough ground to 10 cm, slopes to 15 deg. Warm start from a Skill-A style model (--init, 49-D observation): the first layer of the policy and value networks gets 18 new input columns with ZERO weights, so the extended model
starts out acting exactly like the old one (checked numerically at start-up) and learns to use the scan. PPO settings as in the earlier runs (4 envs, n_steps 512, 10 epochs, clip 0.2, gae 0.95, lr 3e-4), action noise reset to log_std -0.8.
Curriculum and evaluation as in 05_train_terrain_ppo.py, with the fixed evaluation terrains extended to the tall stairs: flat, rough 10 cm, slope up 15, slope down 15, stairs up 6 / 8 / 10 / 12 / 15 cm, stairs down 6 / 10 / 12 / 15 cm (5 episodes each, deterministic policy).
Usage: python 08_train_scan_ppo.py [--gait-ref] --run-name scan_pilot_seed2 --seed 2 --steps 1000000 --init outputs/terr_lift12_seed2/latest_model.zip [--eval-every 250000]
Writes outputs/<run>/: eval_scan.csv, curriculum.log, checkpoints/model_<steps>.zip, latest_model.zip"""
import argparse, sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from racevla.envs.go1_scan_walk import Go1ScanWalkEnv, SCAN_CAP, SCAN_OBS_DIM
from racevla.envs.go1_scan_gait import Go1ScanGaitEnv
from racevla.envs.terrain import LEVELS
from racevla.envs.wrappers import FixedObsNormalize

N_ENVS, N_STEPS, N_EPOCHS, CLIP_RANGE, GAE_LAMBDA, LEARNING_RATE, LOG_STD_INIT = 4, 512, 10, 0.2, 0.95, 3e-4, -0.8
EVAL_TERRAINS = [("flat", ("rough", 0.0)), ("rough 10cm", ("rough", 0.10)), ("slope up 15", ("slope_up", 15)), ("slope down 15", ("slope_down", 15)), ("stairs up 6cm", ("stairs_up", 0.06)), ("stairs up 8cm", ("stairs_up", 0.08)),
                 ("stairs up 10cm", ("stairs_up", 0.10)), ("stairs up 12cm", ("stairs_up", 0.12)), ("stairs up 15cm", ("stairs_up", 0.15)), ("stairs down 6cm", ("stairs_down", 0.06)), ("stairs down 10cm", ("stairs_down", 0.10)),
                 ("stairs down 12cm", ("stairs_down", 0.12)), ("stairs down 15cm", ("stairs_down", 0.15))]
UNLOCK_MIN_EPISODES, UNLOCK_SUCCESS = 15, 0.8


def make_env(gait_ref=False): return FixedObsNormalize(Go1ScanGaitEnv(terrain_ref=True, exact_ik=True, ref_mode="max") if gait_ref else Go1ScanWalkEnv())      # gait_ref: the scan-aware reference gait (racevla/envs/go1_scan_gait.py)


def extended_model(old_path, env, seed):
    """New PPO on the 67-D environment with the weights of the 49-D model; the 18 new input columns of the first layers are zero."""
    old = PPO.load(old_path, device="cpu"); n_old = old.observation_space.shape[0]
    new = PPO("MlpPolicy", env, n_steps=N_STEPS, n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE, clip_range=CLIP_RANGE, gae_lambda=GAE_LAMBDA, seed=seed, device="cpu", verbose=1, policy_kwargs={"log_std_init": LOG_STD_INIT})
    sd_old, sd_new = old.policy.state_dict(), new.policy.state_dict()
    for k, v in sd_old.items():
        if sd_new[k].shape == v.shape: sd_new[k] = v.clone()
        else: pad = torch.zeros_like(sd_new[k]); pad[:, :n_old] = v; sd_new[k] = pad
    new.policy.load_state_dict(sd_new)
    rng = np.random.default_rng(0); x = rng.normal(0, 1, (64, n_old)).astype(np.float32); x67 = np.concatenate([x, rng.normal(0, 1, (64, SCAN_OBS_DIM - n_old)).astype(np.float32)], axis=1)
    a_old, a_new = old.predict(x, deterministic=True)[0], new.predict(x67, deterministic=True)[0]
    assert np.abs(a_old - a_new).max() < 1e-5, "the extended model must act like the old one whatever the scan says"
    print(f"extended model: {n_old} -> {SCAN_OBS_DIM} inputs, max action difference to the old model {np.abs(a_old - a_new).max():.2e} (scan values ignored)", flush=True)
    return new


def evaluate(model, env, n_ep=5):
    out = {}
    for name, terrain in EVAL_TERRAINS:
        ok = 0
        for s in range(n_ep):
            obs, _ = env.reset(seed=1000 + s, options={"terrain": terrain, "terrain_seed": 1000 + s})
            for t in range(1000):
                obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
                if term or trunc: break
            ok += int(bool(info.get("success", False)))
        out[name] = ok / n_ep
    return out


class ScanCallback(BaseCallback):
    def __init__(self, eval_env, run_dir, every):
        super().__init__(); self.eval_env, self.run_dir, self.every, self.last_eval, self.last_cur, self.acc = eval_env, run_dir, every, -1, -1, {}
        self.csv = run_dir / "eval_scan.csv"; self.csv.write_text("timesteps," + ",".join(n for n, _ in EVAL_TERRAINS) + ",mean,tops\n"); self.top = {k: 0 for k in LEVELS}
        (run_dir / "checkpoints").mkdir(exist_ok=True); self.log = open(run_dir / "curriculum.log", "a")

    def _write_eval(self, label):
        m = evaluate(self.model, self.eval_env); mean = float(np.mean(list(m.values())))
        with open(self.csv, "a") as f: f.write(f"{label}," + ",".join(f"{m[n]:.2f}" for n, _ in EVAL_TERRAINS) + f",{mean:.3f},{'/'.join(str(self.top[k]) for k in LEVELS)}\n")
        print(f"EVAL {label}: " + " | ".join(f"{n} {m[n]:.1f}" for n, _ in EVAL_TERRAINS) + f" | mean {mean:.2f} | unlocked tops {self.top}", flush=True)

    def _on_training_start(self): self._write_eval("0")

    def _on_step(self) -> bool:
        if self.num_timesteps // 25_000 != self.last_cur:
            self.last_cur = self.num_timesteps // 25_000
            for st in self.training_env.env_method("pop_stats"):
                for key, (n, s) in st.items(): a = self.acc.setdefault(key, [0, 0]); a[0] += n; a[1] += s
            changed = False
            for kind in LEVELS:
                idx = self.top[kind]; key = (kind, LEVELS[kind][idx]); n, s = self.acc.get(key, [0, 0])
                if idx < SCAN_CAP[kind] and n >= UNLOCK_MIN_EPISODES and s / n >= UNLOCK_SUCCESS:
                    self.top[kind] += 1; changed = True; self.acc.pop((kind, LEVELS[kind][self.top[kind]]), None)
                    msg = f"{self.num_timesteps}: {kind} level {key[1]} succeeded {s}/{n} -> unlocked {LEVELS[kind][self.top[kind]]}"; print("CURRICULUM", msg, flush=True); self.log.write(msg + "\n"); self.log.flush()
            if changed: self.training_env.env_method("set_top", self.top)
        if self.num_timesteps // self.every != self.last_eval and self.num_timesteps >= self.every:
            self.last_eval = self.num_timesteps // self.every; self._write_eval(str(self.num_timesteps)); self.model.save(self.run_dir / "checkpoints" / f"model_{self.num_timesteps}.zip")
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000); ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--run-name", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--init", required=True); ap.add_argument("--gait-ref", action="store_true", help="use the scan-aware terrain-following reference gait")
    a = ap.parse_args(); torch.set_num_threads(1)
    run_dir = ROOT / "outputs" / a.run_name
    if (run_dir / "eval_scan.csv").exists(): sys.exit(f"{run_dir} already has results; pick another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(lambda: make_env(a.gait_ref), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=a.seed)
    model = extended_model(ROOT / a.init if not pathlib.Path(a.init).is_absolute() else a.init, env, a.seed)
    t0 = time.time(); model.learn(total_timesteps=a.steps, callback=ScanCallback(make_env(a.gait_ref), run_dir, a.eval_every), reset_num_timesteps=True)
    model.save(run_dir / "latest_model.zip"); print(f"\ntrained {a.steps} steps in {(time.time() - t0) / 60:.1f} min"); env.close()
