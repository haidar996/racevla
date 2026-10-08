"""Phase 3, first run: a PPO smoke test. Checks the training loop works end to end. NOT a real result."""
import sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from racevla.envs.go1_standing import Go1StandingEnv

N_ENVS = 4                     # one simulator per CPU thread
TOTAL_STEPS = 20_000           # summed over all envs
N_STEPS = 512                  # steps each env collects before one PPO update -> 4 * 512 = 2048 per update


if __name__ == "__main__":     # required: SubprocVecEnv starts child processes that re-import this file
    env = make_vec_env(Go1StandingEnv, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, seed=0)   # also wraps each env in Monitor
    model = PPO("MlpPolicy", env, n_steps=N_STEPS, seed=0, device="cpu", verbose=1)         # default 64x64 tanh actor and critic
    print(model.policy)

    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS)
    dt = time.time() - t0
    print(f"\ntrained {TOTAL_STEPS} steps in {dt:.1f} s  ({TOTAL_STEPS / dt:.0f} steps/s including learning)")

    # act deterministically for one 6 s episode and report what happened
    test_env = Go1StandingEnv()
    obs, _ = test_env.reset(seed=123); total, reason = 0.0, None
    for t in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = test_env.step(action); total += r
        if term or trunc: reason = info.get("termination_reason", "time limit"); break
    print(f"one deterministic test episode: {t + 1} steps, return {total:.1f}, ended: {reason or 'survived 300 steps'}")
    env.close()
