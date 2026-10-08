"""Phase 4, option A: checks for Go1WalkingClockEnv (racevla/envs/go1_walking_clock.py). Run: python scripts/phase4_walking/04_test_clock_env.py"""
import sys, time, pathlib, warnings; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from racevla.envs.go1_walking_clock import Go1WalkingClockEnv, CLOCK_OBS_DIM, GAIT_FREQUENCY, SWING_FRACTION
from racevla.envs.wrappers import FixedObsNormalize

env = Go1WalkingClockEnv(push=False)
print("1) observation: 49 numbers = 45 robot + 2 command + 2 clock; the clock is a unit-length (sin, cos) pair while moving and (0, 0) when told to stand")
obs, _ = env.reset(seed=1); env.command = np.array([0.4, 0.0]); obs, *_ = env.step(np.zeros(12, np.float32))
assert obs.shape == (CLOCK_OBS_DIM,) == (49,) and env.observation_space.shape == (49,)
assert abs(np.hypot(*obs[-2:]) - 1.0) < 1e-5, obs[-2:]
env.command = np.array([0.0, 0.0]); obs, *_ = env.step(np.zeros(12, np.float32)); assert np.allclose(obs[-2:], 0.0)
print("   ok: moving ->", np.round(np.array([np.sin(2 * np.pi * env.phase), np.cos(2 * np.pi * env.phase)]), 2), "| standing -> [0 0]")

print("2) the metronome advances by", GAIT_FREQUENCY * 0.02, "per step while moving (a cycle every 20 steps) and is frozen while standing")
env.reset(seed=2); env.command = np.array([0.4, 0.2]); p0 = env.phase; phases = []
for t in range(40): env.command = np.array([0.4, 0.2]); env.step(np.zeros(12, np.float32)); phases.append(env.phase)
assert abs(((phases[19] - p0 + 0.5) % 1.0) - 0.5) < 1e-6, "one full cycle after 20 steps"
env.command = np.array([0.0, 0.0]); p = env.phase; [env.step(np.zeros(12, np.float32)) for _ in range(5)]; assert env.phase == p
print(f"   ok: phase {p0:.2f} -> {phases[0]:.2f} -> ... after 20 steps back to {phases[19]:.2f}")

print("3) trot schedule: FR and RL swing together, FL and RR swing together, the two pairs never swing at once, 40 % of the time a foot is in the air")
env.reset(seed=3); env.command = np.array([0.4, 0.0]); counts = np.zeros(4); both = 0; none = 0
for k in range(2000):
    env.phase = (k / 2000.0); s = env.desired_swing(); counts += s
    assert s[0] == s[3] and s[1] == s[2], s
    both += int(s[0] and s[1]); none += int(not s.any())
fr = counts / 2000
print(f"   swing share per foot (FR FL RR RL): {fr.round(3)} | both pairs in the air at once: {both} | all four down (double stance): {100 * none / 2000:.0f} % of the cycle")
assert np.allclose(fr, SWING_FRACTION, atol=0.01) and both == 0 and abs(none / 2000 - 0.2) < 0.01

print("4) reward with the robot just standing (zero action):")
def run(cmd, n=120, seed=4):
    env.reset(seed=seed); out = []
    for t in range(n):
        env.command = np.array(cmd, float); _, r, term, trunc, info = env.step(np.zeros(12, np.float32)); out.append((r, info["r_sched"], info["p_swing"]))
        if term: break
    return np.mean(out[40:], axis=0)
stand, move = run([0.0, 0.0]), run([0.5, 0.0])
print(f"   told to STAND:      reward {stand[0]:.2f}  schedule score {stand[1]:.2f} (all four feet down)  swing-height penalty {stand[2]:.4f}")
print(f"   told to WALK 0.5:   reward {move[0]:.2f}  schedule score {move[1]:.2f} (a standing robot matches only the stance feet)  swing-height penalty {move[2]:.4f}")
assert stand[1] > 0.95 and stand[2] == 0 and 0.4 < move[1] < 0.8 and move[2] > 0.0 and stand[0] > move[0]

print("5) reset: random start phase, same seed -> same observation, reward never negative, foot heights ~0 at rest")
ph = [(env.reset(seed=s), env.phase)[1] for s in range(20)]; assert len(set(np.round(ph, 3))) > 10
a, _ = env.reset(seed=7); b, _ = Go1WalkingClockEnv(push=False).reset(seed=7); assert np.array_equal(a, b)
env.reset(seed=8); [env.step(np.zeros(12, np.float32)) for _ in range(100)]; h = env._foot_heights(); print("   foot heights at rest (cm):", np.round(100 * h, 1)); assert np.all(np.abs(h) < 0.02)
rng = np.random.default_rng(0); env.reset(seed=9); rs = []
for t in range(200):
    _, r, term, trunc, info = env.step(rng.uniform(-1, 1, 12).astype(np.float32)); rs.append(r)
    if term: break
assert min(rs) >= 0

print("6) Stable-Baselines3 check_env on the wrapped environment")
from stable_baselines3.common.env_checker import check_env
warnings.simplefilter("error"); check_env(FixedObsNormalize(Go1WalkingClockEnv(push=True)), warn=True); warnings.simplefilter("default"); print("   passes")

print("7) speed (random actions, auto-reset)")
e2 = Go1WalkingClockEnv(push=True); e2.reset(seed=0); n = 3000; t0 = time.time(); rng = np.random.default_rng(2)
for i in range(n):
    _, r, term, trunc, info = e2.step(rng.uniform(-1, 1, 12).astype(np.float32))
    if term or trunc: e2.reset()
print(f"   {n / (time.time() - t0):.0f} policy steps/s")
print("\nCLOCK ENV CHECKS PASSED")
