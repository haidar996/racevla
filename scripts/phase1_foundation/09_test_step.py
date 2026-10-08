"""Test step(): timing, previous-action wiring, truncation, Gymnasium API compliance, speed."""
import sys, time; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import numpy as np
from gymnasium.utils.env_checker import check_env
from racevla.envs.go1_standing import Go1StandingEnv, DECIMATION, MAX_EPISODE_STEPS
from racevla.envs.observation import OBS_SLICES as S
np.set_printoptions(precision=3, suppress=True)
env = Go1StandingEnv()

print("1) simulated time per step")
env.reset(seed=0); t0 = env.data.time; env.step(np.zeros(12))
print(f"   {env.data.time - t0:.3f} s per step (want {DECIMATION * env.model.opt.timestep:.3f}) -> policy rate {1/(env.data.time - t0):.0f} Hz")

print("2) the action you send comes back as 'previous action' in the next observation (and is clipped)")
env.reset(seed=0); a = np.linspace(-2, 2, 12); obs, *_ = env.step(a)
print("   sent   :", a); print("   in obs :", obs[S['prev_action']]); assert np.allclose(obs[S['prev_action']], np.clip(a, -1, 1), atol=1e-6)

print("3) truncation at exactly MAX_EPISODE_STEPS, terminated stays False (placeholder)")
env.reset(seed=1); n = 0
while True:
    obs, r, term, trunc, info = env.step(np.zeros(12)); n += 1
    if term or trunc: break
print(f"   episode ended after {n} steps (want {MAX_EPISODE_STEPS}) truncated={trunc} terminated={term} | base z {info['base_height']:.3f}"); assert n == MAX_EPISODE_STEPS and trunc

print("4) Gymnasium's official API checker")
check_env(Go1StandingEnv(), skip_render_check=True); print("   check_env passed")

print("5) speed (this is the number that decides your training time)")
env.reset(seed=2); rng = np.random.default_rng(0); N = 3000; t = time.time()
for i in range(N):
    _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 12))
    if term or trunc: env.reset()
dt = time.time() - t
print(f"   {N/dt:.0f} policy steps/s = {N*DECIMATION/dt:.0f} physics steps/s  (1M policy steps would take ~{1e6/(N/dt)/60:.0f} min)")
print("\nSTEP CHECKS PASSED")
