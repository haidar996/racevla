"""Phase 5: PPO on the running task (velocity commands up to --vx-max, flight phase wanted above 1 m/s), no pushes. Environment: Go1RunningSineEnv (racevla/envs/go1_running_sine.py): option-B reference leg
motion with a speed-dependent gait (table read from --gait-table JSON) + learned residual; walking reward with a weaker vertical-velocity penalty and a flight bonus.
Warm start: --init <model.zip> (the walking policy, or the previous speed stage). Same PPO settings as the walking runs (n_steps 512, 10 epochs, clip 0.2, gae 0.95, lr 3e-4 constant, 4 envs); the policy's
action noise is reset to log_std -0.8 so that it explores the new gait. Every --eval-every steps 16 fixed episodes (seeds 1000..) are scored: falls, speed error and speed reached AT RUNNING COMMANDS (>= 1 m/s),
flight share (all four feet off the floor), feet down, slip.
Usage: python 02_train_run_ppo.py --run-name run1_1p5_seed0 --seed 0 --vx-max 1.5 --init models/walking_policy_sine_seed1.zip --gait-table outputs/analysis/running/gait_table.json --steps 1000000
Writes outputs/<run-name>/: eval_run.csv, checkpoints/model_<steps>.zip, best_model.zip, latest_model.zip"""
import argparse, json, sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from racevla.envs.go1_running_sine import Go1RunningSineEnv, FLIGHT_MIN_CMD
from racevla.envs.go1_standing import MAX_EPISODE_STEPS
from racevla.envs.wrappers import FixedObsNormalize

N_ENVS = 4
EVAL_SEEDS = list(range(1000, 1016))
N_STEPS, N_EPOCHS, CLIP_RANGE, GAE_LAMBDA, LEARNING_RATE, LOG_STD_INIT = 512, 10, 0.2, 0.95, 3e-4, -0.8
FEET = ["FR", "FL", "RR", "RL"]


def make_env(vx_max, gait, wz_max=0.5):
    return FixedObsNormalize(Go1RunningSineEnv(push=False, vx_range=(0.0, vx_max), wz_range=(-wz_max, wz_max), gait=gait))


def evaluate(model, env, seeds):
    raw = env.unwrapped; rets, fell, ve, yaw, ratio_v, ratio_c, flight, down, slip, contacts = [], 0, [], [], [], [], [], [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s); ret = 0.0
        for t in range(MAX_EPISODE_STEPS):
            cv, cw = raw.command; obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); ret += r
            c = raw._foot_contacts(); down.append(int(c.sum())); slip.append(info["p_slip"])
            if cv > 0.1: ve.append(abs(info["vx"] - cv))
            if abs(cw) > 0.1: yaw.append(abs(info["wz"] - cw))
            if cv >= FLIGHT_MIN_CMD: flight.append(float(c.sum() == 0)); ratio_v.append(info["vx"]); ratio_c.append(cv); contacts.append(c.astype(float))
            if term or trunc: break
        rets.append(ret); fell += int(term)
    m = lambda x: float(np.mean(x)) if len(x) else float("nan")
    return {"return": m(rets), "fell": fell, "vel_err": m(ve), "yaw_err": m(yaw), "run_speed_ratio": (m(ratio_v) / m(ratio_c)) if ratio_c else float("nan"), "run_steps": len(ratio_c),
            "flight": m(flight), "feet_down": m(down), "slip": m(slip), "down": np.mean(contacts, axis=0) if contacts else np.full(4, np.nan)}


class RunEvalCallback(BaseCallback):
    def __init__(self, eval_env, run_dir, every):
        super().__init__(); self.eval_env, self.run_dir, self.every, self.last, self.best = eval_env, run_dir, every, 0, -np.inf
        self.csv = run_dir / "eval_run.csv"
        self.csv.write_text(f"timesteps,mean_return,fell_out_of_{len(EVAL_SEEDS)},speed_error,turn_error,run_speed_ratio,run_steps,flight_share,feet_down,foot_slip,down_FR,down_FL,down_RR,down_RL\n")
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last < self.every: return True
        self.last = self.num_timesteps; m = evaluate(self.model, self.eval_env, EVAL_SEEDS); d = m["down"]
        with open(self.csv, "a") as f:
            f.write(f"{self.num_timesteps},{m['return']:.1f},{m['fell']},{m['vel_err']:.3f},{m['yaw_err']:.3f},{m['run_speed_ratio']:.3f},{m['run_steps']},{m['flight']:.3f},{m['feet_down']:.2f},{m['slip']:.3f},"
                    f"{d[0]:.2f},{d[1]:.2f},{d[2]:.2f},{d[3]:.2f}\n")
        self.model.save(self.run_dir / "checkpoints" / f"model_{self.num_timesteps}.zip")
        if m["return"] > self.best: self.best = m["return"]; self.model.save(self.run_dir / "best_model.zip")
        print(f"EVAL {self.num_timesteps}: return {m['return']:.0f} fell {m['fell']}/{len(EVAL_SEEDS)} speed_err {m['vel_err']:.3f} turn_err {m['yaw_err']:.3f} run_speed_ratio {m['run_speed_ratio']:.2f} "
              f"flight {m['flight']:.2f} feet_down {m['feet_down']:.2f} slip {m['slip']:.2f} down {d.round(2)}", flush=True)
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000); ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--run-name", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vx-max", type=float, default=1.5); ap.add_argument("--init", required=True); ap.add_argument("--gait-table", required=True)
    a = ap.parse_args(); torch.set_num_threads(1)
    gait = {"table": json.load(open(a.gait_table))}
    run_dir = ROOT / "outputs" / a.run_name
    if (run_dir / "eval_run.csv").exists(): sys.exit(f"{run_dir} already has results; pick another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(lambda: make_env(a.vx_max, gait), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=a.seed)
    model = PPO.load(ROOT / a.init if not pathlib.Path(a.init).is_absolute() else a.init, env=env, device="cpu", seed=a.seed, custom_objects={"n_steps": N_STEPS, "n_epochs": N_EPOCHS, "learning_rate": LEARNING_RATE, "clip_range": lambda _: CLIP_RANGE})
    with torch.no_grad(): model.policy.log_std.fill_(LOG_STD_INIT)
    t0 = time.time(); model.learn(total_timesteps=a.steps, callback=RunEvalCallback(make_env(a.vx_max, gait), run_dir, a.eval_every), reset_num_timesteps=True)
    model.save(run_dir / "latest_model.zip"); print(f"\ntrained {a.steps} steps in {(time.time() - t0) / 60:.1f} min"); env.close()
