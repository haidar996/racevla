"""Phase 3: PPO smoke test on the push task, with small initial exploration noise (log_std_init = -2).
Also scores a do-nothing (zero action) baseline on the same pushes. Still a plumbing check, not a result."""
import sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from racevla.envs.go1_standing import Go1StandingEnv, MAX_EPISODE_STEPS

N_ENVS = 4
TOTAL_STEPS = 20_000
N_STEPS = 512
LOG_STD_INIT = -2.0            # exploration noise std = exp(-2) = 0.135  (SB3 default is 0 -> std 1)
EVAL_SEEDS = range(1000, 1010) # fixed seeds -> the same start states and the same pushes for every policy we score


def evaluate(act_fn, name):
    """Run 10 full push episodes with act_fn(obs) -> action; report mean length, mean return, how many fell."""
    env = Go1StandingEnv(push=True); lens, rets, falls = [], [], 0
    for s in EVAL_SEEDS:
        obs, _ = env.reset(seed=s); ret = 0.0
        for t in range(MAX_EPISODE_STEPS):
            obs, r, term, trunc, _ = env.step(act_fn(obs)); ret += r
            if term or trunc: break
        lens.append(t + 1); rets.append(ret); falls += term
    print(f"{name:26s} mean length {np.mean(lens):6.1f}/{MAX_EPISODE_STEPS}   mean return {np.mean(rets):6.1f}   fell {falls}/{len(EVAL_SEEDS)}")


if __name__ == "__main__":
    env = make_vec_env(Go1StandingEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=0, env_kwargs={"push": True})
    model = PPO("MlpPolicy", env, n_steps=N_STEPS, seed=0, device="cpu", verbose=1, policy_kwargs={"log_std_init": LOG_STD_INIT})

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS)
    print(f"\ntrained {TOTAL_STEPS} steps in {time.time() - t0:.1f} s")
    env.close()

    print("\nevaluation on 10 fixed push episodes (deterministic policy):")
    evaluate(lambda obs: np.zeros(12), "zero action (baseline)")
    evaluate(lambda obs: model.predict(obs, deterministic=True)[0], "PPO after 20k steps")
