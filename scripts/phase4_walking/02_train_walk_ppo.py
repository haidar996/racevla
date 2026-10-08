"""Phase 4: PPO on the velocity-command walking task, no pushes. Same PPO settings as the standing runs that worked (original runs 3-5), plus a walking evaluation:
every --eval-every steps a fixed set of episodes (seeds 1000..) is played with the deterministic policy and scored on what matters for walking -- falls, speed error, turn-rate error, how fast
the robot REALLY moves compared with the command, how many feet are down, foot slip and (while moving) the share of time each foot is on the floor and how high the feet lift.
--env walk  = Go1WalkingEnv (racevla/envs/go1_walking.py)            reward v2: tracking + slip penalty + capped air-time bonus
--env clock = Go1WalkingClockEnv (racevla/envs/go1_walking_clock.py)  option A: gait clock + trot contact schedule + half-sine swing-height target
--env gait  = Go1WalkingGaitRewardEnv (racevla/envs/go1_walking_gait.py) option C: reward-only gait fixes (clearance, trot pairing, alternation, stuck-foot penalty), standard 47-D observation
--env sine  = Go1WalkingSineEnv (racevla/envs/go1_walking_sine.py)    option B: sine-shaped reference leg motion from the clock added to the joint targets + small learned residual
Usage: python 02_train_walk_ppo.py [--env clock] [--steps 1000000] [--eval-every 50000] [--run-name walk3_clock_1M_seed0] [--seed 0]
Writes to outputs/<run-name>/: eval_walk.csv, checkpoints/model_<steps>.zip (one per evaluation), best_model.zip (highest eval return), latest_model.zip"""
import argparse, os, sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from racevla.envs.go1_walking import Go1WalkingEnv, CMD_VX_RANGE, CMD_WZ_RANGE
from racevla.envs.go1_walking_clock import Go1WalkingClockEnv
from racevla.envs.go1_walking_sine import Go1WalkingSineEnv
from racevla.envs.go1_walking_gait import Go1WalkingGaitRewardEnv
from racevla.envs.go1_standing import MAX_EPISODE_STEPS
from racevla.envs.wrappers import FixedObsNormalize

N_ENVS = 4
EVAL_SEEDS = list(range(1000, 1016))      # 16 fixed episodes per evaluation (each with ~5 commands)
# PPO settings = the standing runs that worked (original runs 3-5), unchanged so that any difference comes from the task, not the algorithm
N_STEPS, N_EPOCHS, CLIP_RANGE, GAE_LAMBDA, LEARNING_RATE, LOG_STD_INIT = 512, 10, 0.2, 0.95, 3e-4, -0.8
ENVS = {"walk": Go1WalkingEnv, "clock": Go1WalkingClockEnv, "sine": Go1WalkingSineEnv, "gait": Go1WalkingGaitRewardEnv}
FEET = ["FR", "FL", "RR", "RL"]


def make_env(env_name="walk", vx_max=CMD_VX_RANGE[1], wz_max=CMD_WZ_RANGE[1]):
    return FixedObsNormalize(ENVS[env_name](push=False, vx_range=(0.0, vx_max), wz_range=(-wz_max, wz_max)))


def evaluate(model, env, seeds):
    """Play one episode per seed (deterministic policy). Returns a dict of walking metrics."""
    rets, lens, fell, vel_err, yaw_err, vx_ratio, feet_down, slip, contacts, heights = [], [], 0, [], [], [], [], [], [], []
    raw = env.unwrapped
    for s in seeds:
        obs, _ = env.reset(seed=s); ret = 0.0
        for t in range(MAX_EPISODE_STEPS):
            cmd_v, cmd_w = raw.command                                       # the command this step is scored against
            obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); ret += r
            if cmd_v > 0.1: vel_err.append(abs(info["vx"] - cmd_v))
            if cmd_v > 0.2: vx_ratio.append(info["vx"] / cmd_v)
            if abs(cmd_w) > 0.1: yaw_err.append(abs(info["wz"] - cmd_w))
            c = raw._foot_contacts(); feet_down.append(int(c.sum())); slip.append(info["p_slip"])
            if cmd_v > 0.1 or abs(cmd_w) > 0.1:                                # per-foot statistics only while asked to move
                contacts.append(c.astype(float))
                if hasattr(raw, "_foot_heights"): heights.append(raw._foot_heights())
            if term or trunc: break
        rets.append(ret); lens.append(t + 1); fell += int(term)
    m = lambda x: float(np.mean(x)) if len(x) else float("nan")
    down = np.mean(contacts, axis=0) if len(contacts) else np.full(4, np.nan)
    lift = float(np.percentile(np.concatenate(heights), 95)) * 100 if len(heights) else float("nan")      # cm: how high the feet get (95th percentile of all foot heights while moving)
    return {"return": m(rets), "length": m(lens), "fell": fell, "vel_err": m(vel_err), "yaw_err": m(yaw_err), "vx_ratio": m(vx_ratio), "feet_down": m(feet_down), "slip": m(slip), "down": down, "lift_cm": lift}


class WalkEvalCallback(BaseCallback):
    def __init__(self, eval_env, run_dir, every):
        super().__init__(); self.eval_env, self.run_dir, self.every, self.last, self.best = eval_env, run_dir, every, 0, -np.inf
        self.csv = run_dir / "eval_walk.csv"
        self.csv.write_text(f"timesteps,mean_return,mean_length,fell_out_of_{len(EVAL_SEEDS)},speed_error,turn_error,speed_ratio,feet_down,foot_slip,down_FR,down_FL,down_RR,down_RL,lift_cm\n")
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last < self.every: return True
        self.last = self.num_timesteps; m = evaluate(self.model, self.eval_env, EVAL_SEEDS); d = m["down"]
        with open(self.csv, "a") as f:
            f.write(f"{self.num_timesteps},{m['return']:.1f},{m['length']:.1f},{m['fell']},{m['vel_err']:.3f},{m['yaw_err']:.3f},{m['vx_ratio']:.3f},{m['feet_down']:.2f},{m['slip']:.3f},"
                    f"{d[0]:.2f},{d[1]:.2f},{d[2]:.2f},{d[3]:.2f},{m['lift_cm']:.1f}\n")
        self.model.save(self.run_dir / "checkpoints" / f"model_{self.num_timesteps}.zip")
        if m["return"] > self.best: self.best = m["return"]; self.model.save(self.run_dir / "best_model.zip")
        print(f"EVAL {self.num_timesteps}: return {m['return']:.0f} len {m['length']:.0f} fell {m['fell']}/{len(EVAL_SEEDS)} speed_err {m['vel_err']:.3f} turn_err {m['yaw_err']:.3f} speed_ratio {m['vx_ratio']:.2f} "
              f"feet_down {m['feet_down']:.2f} slip {m['slip']:.2f} down(FR FL RR RL) {d.round(2)} lift {m['lift_cm']:.1f} cm", flush=True)
        return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", choices=list(ENVS), default="walk")
    ap.add_argument("--steps", type=int, default=1_000_000); ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--run-name", default="walk_ppo_1M_seed0"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vx-max", type=float, default=CMD_VX_RANGE[1], help="largest commanded forward speed in TRAINING (evaluation uses the same)")
    ap.add_argument("--wz-max", type=float, default=CMD_WZ_RANGE[1])
    a = ap.parse_args(); torch.set_num_threads(1)
    run_dir = ROOT / "outputs" / a.run_name
    if (run_dir / "eval_walk.csv").exists(): sys.exit(f"{run_dir} already has results; pick another --run-name")
    run_dir.mkdir(parents=True, exist_ok=True)
    env = make_vec_env(lambda: make_env(a.env, a.vx_max, a.wz_max), n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=a.seed)
    model = PPO("MlpPolicy", env, n_steps=N_STEPS, n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE, clip_range=CLIP_RANGE, gae_lambda=GAE_LAMBDA,
                seed=a.seed, device="cpu", verbose=1, policy_kwargs={"log_std_init": LOG_STD_INIT})
    t0 = time.time(); model.learn(total_timesteps=a.steps, callback=WalkEvalCallback(make_env(a.env, a.vx_max, a.wz_max), run_dir, a.eval_every))
    model.save(run_dir / "latest_model.zip"); print(f"\ntrained {a.steps} steps in {(time.time() - t0) / 60:.1f} min"); env.close()
