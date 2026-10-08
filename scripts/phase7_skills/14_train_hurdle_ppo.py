"""Phase 7: PPO for the learned hurdle jump while running: Go1HurdleEnv (racevla/envs/go1_hurdle.py): 75-D observation (running obs + 24-number far height scan + 2 jump numbers), 13 actions (12 corrections + the jump TRIGGER), hurdles 3-25 cm across the whole course (the curriculum starts at 3 cm),
command ~ U(1.3, 1.7) m/s, success = 2 s after passing the bar's zone the robot is upright and runs at >= 50 % of the commanded speed. Warm start from the packaged running policy (49 -> 75 inputs with zero columns, 12 -> 13 outputs; the new output (the trigger) has zero weights and bias -1, i.e. 'no jump');
the old behaviour is checked at start-up. PPO as before (4 envs, n_steps 512, 10 epochs, clip 0.2, gae 0.95, lr 3e-4), action noise reset to log_std -0.8 on all 13 outputs.
Curriculum: hurdle heights HURDLE_LEVELS[0..top]; the next height is unlocked when the top height succeeded in >= 70 % of >= 15 training episodes of the last 25k steps (checked every 25k steps). --stepover: Go1StepOverEnv (high-stepping reference, 12 actions, no jump). 20 % of the episodes have no hurdle.
Evaluation every --eval-every steps (and once at the start): 5 episodes each, deterministic: no hurdle, 3, 7, 10, 15, 20, 25 cm; reports success and 'cleared' (crossed the bar without hitting it).
Usage: python 14_train_hurdle_ppo.py --run-name hurdle_pilot --seed 0 --steps 1000000 --init models/running_policy_2p5_seed0.zip
Writes outputs/<run>/: eval_hurdle.csv, curriculum.log, checkpoints/model_<steps>.zip, latest_model.zip"""
import argparse, sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from racevla.envs.go1_hurdle import Go1HurdleEnv, OBS_DIM
from racevla.envs.go1_stepover import Go1StepOverEnv
from racevla.envs.terrain import HURDLE_LEVELS
from racevla.envs.wrappers import FixedObsNormalize

N_ENVS, N_STEPS, N_EPOCHS, CLIP_RANGE, GAE_LAMBDA, LEARNING_RATE, LOG_STD_INIT = 4, 512, 10, 0.2, 0.95, 3e-4, -0.8
EVAL_TERRAINS = [("no hurdle", ("rough", 0.0)), ("3 cm", ("hurdle", 0.03)), ("5 cm", ("hurdle", 0.05)), ("7 cm", ("hurdle", 0.07)), ("10 cm", ("hurdle", 0.10)), ("12.5 cm", ("hurdle", 0.125)), ("15 cm", ("hurdle", 0.15)), ("20 cm", ("hurdle", 0.20))]
UNLOCK_MIN_EPISODES, UNLOCK_SUCCESS = 15, 0.7
RESIDUAL_LOG_STD = -1.8


STEPOVER = False                                          # set from --stepover (step-over reference gait, 12 actions, no jump trigger)


def make_env(stepover=False): return FixedObsNormalize(Go1StepOverEnv() if stepover else Go1HurdleEnv())


def extended_model(old_path, env, seed):
    old = PPO.load(old_path, device="cpu"); n_old = old.observation_space.shape[0]
    new = PPO("MlpPolicy", env, n_steps=N_STEPS, n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE, clip_range=CLIP_RANGE, gae_lambda=GAE_LAMBDA, seed=seed, device="cpu", verbose=1, policy_kwargs={"log_std_init": LOG_STD_INIT})
    sd_old, sd_new = old.policy.state_dict(), new.policy.state_dict()
    for k, v in sd_old.items():
        if sd_new[k].shape == v.shape: sd_new[k] = v.clone()
        elif v.dim() == 2 and v.shape[0] == sd_new[k].shape[0]: pad = torch.zeros_like(sd_new[k]); pad[:, :v.shape[1]] = v; sd_new[k] = pad              # first layers: new input columns are zero
        elif v.dim() == 2: pad = torch.zeros_like(sd_new[k]); pad[:v.shape[0]] = v; sd_new[k] = pad                                                         # action layer: new output row is zero
        else: pad = torch.full_like(sd_new[k], -1.0); pad[:v.shape[0]] = v; sd_new[k] = pad                                                                 # action bias: new output starts at -1 = no jump
    new.policy.load_state_dict(sd_new)
    with torch.no_grad(): new.policy.log_std.fill_(LOG_STD_INIT); new.policy.log_std[:12] = RESIDUAL_LOG_STD              # less noise on the 12 corrections (a noisy run fails the flat course), the trigger keeps std 0.45
    rng = np.random.default_rng(0); x = rng.normal(0, 1, (64, n_old)).astype(np.float32); x75 = np.concatenate([x, rng.normal(0, 1, (64, OBS_DIM - n_old)).astype(np.float32)], axis=1)
    a_old, a_new = old.predict(x, deterministic=True)[0], new.predict(x75, deterministic=True)[0]
    n_act = a_new.shape[1]; assert np.abs(a_old - a_new[:, :12]).max() < 1e-5 and (n_act == 12 or (a_new[:, 12] < 0).all()), "the extended model must act like the old one and never trigger a jump at the start"
    print(f"extended model: {n_old} -> {OBS_DIM} inputs, 12 -> {n_act} outputs, max difference to the old model {np.abs(a_old - a_new[:, :12]).max():.2e}" + ("" if n_act == 12 else f", trigger output {a_new[:, 12].mean():.2f} (no jump)"), flush=True)
    return new


def evaluate(model, env, n_ep=5):
    out = {}
    for name, terrain in EVAL_TERRAINS:
        ok = cleared = 0
        for s in range(n_ep):
            obs, _ = env.reset(seed=1000 + s, options={"terrain": terrain, "terrain_seed": 1000 + s})
            for t in range(400):
                obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
                if term or trunc: break
            ok += int(bool(info.get("success", False))); cleared += int(info.get("crossed", 0) > 0 and info.get("termination_reason") != "hit_hurdle")
        out[name] = (ok / n_ep, cleared / n_ep)
    return out


class HurdleCallback(BaseCallback):
    def __init__(self, eval_env, run_dir, every):
        super().__init__(); self.eval_env, self.run_dir, self.every, self.last_eval, self.last_cur, self.acc, self.top = eval_env, run_dir, every, -1, -1, {}, 0
        self.csv = run_dir / "eval_hurdle.csv"; self.csv.write_text("timesteps," + ",".join(f"{n} success,{n} cleared" for n, _ in EVAL_TERRAINS) + ",top\n")
        (run_dir / "checkpoints").mkdir(exist_ok=True); self.log = open(run_dir / "curriculum.log", "a")

    def _write_eval(self, label):
        m = evaluate(self.model, self.eval_env)
        with open(self.csv, "a") as f: f.write(f"{label}," + ",".join(f"{m[n][0]:.2f},{m[n][1]:.2f}" for n, _ in EVAL_TERRAINS) + f",{self.top}\n")
        print(f"EVAL {label}: " + " | ".join(f"{n}: success {m[n][0]:.1f} cleared {m[n][1]:.1f}" for n, _ in EVAL_TERRAINS) + f" | unlocked top {HURDLE_LEVELS[self.top]}", flush=True)

    def _on_training_start(self): self._write_eval("0")

    def _on_step(self) -> bool:
        if self.num_timesteps // 25_000 != self.last_cur:
            self.last_cur = self.num_timesteps // 25_000; self.acc = {}                               # window statistics: the unlock looks at the last 25k steps only
            for st in self.training_env.env_method("pop_stats"):
                for key, (n, s) in st.items(): a = self.acc.setdefault(key, [0, 0]); a[0] += n; a[1] += s
            key = ("hurdle", HURDLE_LEVELS[self.top]); n, s = self.acc.get(key, [0, 0])
            if self.top < len(HURDLE_LEVELS) - 1 and n >= UNLOCK_MIN_EPISODES and s / n >= UNLOCK_SUCCESS:
                self.top += 1; self.acc.pop(("hurdle", HURDLE_LEVELS[self.top]), None); self.training_env.env_method("set_top", self.top)
                msg = f"{self.num_timesteps}: hurdle {key[1]} succeeded {s}/{n} -> unlocked {HURDLE_LEVELS[self.top]}"; print("CURRICULUM", msg, flush=True); self.log.write(msg + "\n"); self.log.flush()
            self.log.write(f"{self.num_timesteps}: top {HURDLE_LEVELS[self.top]} stats " + ", ".join(f"{k[1]}:{v[1]}/{v[0]}" for k, v in sorted(self.acc.items(), key=lambda kv: kv[0][1])) + "\n"); self.log.flush()
        if self.num_timesteps // self.every != self.last_eval and self.num_timesteps >= self.every:
            self.last_eval = self.num_timesteps // self.every; self._write_eval(str(self.num_timesteps)); self.model.save(self.run_dir / "checkpoints" / f"model_{self.num_timesteps}.zip")
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000); ap.add_argument("--eval-every", type=int, default=250_000)
    ap.add_argument("--run-name", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--init", required=True); ap.add_argument("--stepover", action="store_true"); ap.add_argument("--unlock", type=float, default=UNLOCK_SUCCESS, help="success rate of the top height (last 25k steps) needed to unlock the next one")
    a = ap.parse_args(); torch.set_num_threads(1); UNLOCK_SUCCESS = a.unlock
    run_dir = ROOT / "outputs" / a.run_name
    if (run_dir / "eval_hurdle.csv").exists(): sys.exit(f"{run_dir} already has results; pick another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(lambda: make_env(a.stepover), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=a.seed)
    model = extended_model(ROOT / a.init if not pathlib.Path(a.init).is_absolute() else a.init, env, a.seed)
    t0 = time.time(); model.learn(total_timesteps=a.steps, callback=HurdleCallback(make_env(a.stepover), run_dir, a.eval_every), reset_num_timesteps=True)
    model.save(run_dir / "latest_model.zip"); print(f"\ntrained {a.steps} steps in {(time.time() - t0) / 60:.1f} min"); env.close()
