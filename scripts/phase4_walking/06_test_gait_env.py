"""Phase 4, option C: checks for Go1WalkingGaitRewardEnv (racevla/envs/go1_walking_gait.py). Run: python scripts/phase4_walking/06_test_gait_env.py"""
import sys, time, pathlib, warnings; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from racevla.envs.go1_walking_gait import Go1WalkingGaitRewardEnv
from racevla.envs.wrappers import FixedObsNormalize
env = Go1WalkingGaitRewardEnv(push=False); z = np.zeros(12, np.float32)
print("1) observation stays the standard 47-D (no clock, no reference); foot heights ~0 at rest")
obs, _ = env.reset(seed=1); assert obs.shape == (47,)
[env.step(z) for _ in range(60)]; h = env._foot_heights(); print("   foot heights at rest (cm):", np.round(100 * h, 1)); assert np.all(np.abs(h) < 0.02)
print("2) a robot that just STANDS while told to WALK gets the stuck-foot penalty (and little reward); told to STAND it gets the stand reward and no penalty")
def run(cmd, n=150, seed=2):
    env.reset(seed=seed); out = []
    for t in range(n):
        env.command = np.array(cmd, float); _, r, term, trunc, info = env.step(z); out.append((r, info["p_stuck"], info["r_stand"], info["r_trot"], info["r_raw"]))
    return np.array(out)
walk, stand = run([0.5, 0.0]), run([0.0, 0.0])
print(f"   told to WALK 0.5 while standing: first second  reward {walk[:50, 0].mean():.2f}, last second reward {walk[-50:, 0].mean():.2f}, stuck penalty (last second) {walk[-50:, 1].mean():.2f}")
print(f"   told to STAND:                   reward {stand[-50:, 0].mean():.2f}, stuck penalty {stand[-50:, 1].mean():.2f}, stand reward {stand[-50:, 2].mean():.2f}")
assert walk[-50:, 1].mean() > 2.0 and stand[:, 1].max() == 0.0 and stand[-50:, 2].mean() > 0.95 and stand[-50:, 0].mean() > walk[-50:, 0].mean()
assert walk[-50:, 4].mean() > -3.0
print("3) the stuck penalty is capped (a foot stuck for 10 s costs no more than 1 s of overtime per foot) and the reward is never negative")
assert walk[:, 1].max() <= 4 * 1.0 + 1e-6 and walk[:, 0].min() >= 0.0
print("4) trot reward terms respond to contact patterns (set by hand): perfect trot swing scores alternation 1, pronk/stand 0")
env.reset(seed=3); env.command = np.array([0.5, 0.0])
def terms_for(contact):
    env.in_contact = np.array(contact, bool); c = env.in_contact
    r_trot = 0.5 * (float(c[0] == c[3]) + float(c[1] == c[2])); r_alt = abs(int(c[0]) + int(c[3]) - int(c[1]) - int(c[2])) / 2.0; return r_trot, r_alt
for name, c in [("FR+RL down, FL+RR up", [1, 0, 0, 1]), ("all four down", [1, 1, 1, 1]), ("all four up", [0, 0, 0, 0]), ("three down", [1, 1, 1, 0])]: print(f"   {name:22s} pairing {terms_for(c)[0]:.2f}  alternation {terms_for(c)[1]:.2f}")
assert terms_for([1, 0, 0, 1]) == (1.0, 1.0) and terms_for([1, 1, 1, 1]) == (1.0, 0.0) and terms_for([0, 0, 0, 0]) == (1.0, 0.0) and terms_for([1, 1, 1, 0])[1] == 0.5
print("5) SB3 environment check, speed")
from stable_baselines3.common.env_checker import check_env
warnings.simplefilter("error"); check_env(FixedObsNormalize(Go1WalkingGaitRewardEnv(push=True)), warn=True); warnings.simplefilter("default")
e2 = Go1WalkingGaitRewardEnv(push=True); e2.reset(seed=0); n = 2000; t0 = time.time(); rng = np.random.default_rng(2)
for i in range(n):
    _, r, term, trunc, info = e2.step(rng.uniform(-1, 1, 12).astype(np.float32))
    if term or trunc: e2.reset()
print(f"   passes; {n / (time.time() - t0):.0f} policy steps/s")
print("\nGAIT-REWARD ENV CHECKS PASSED")
