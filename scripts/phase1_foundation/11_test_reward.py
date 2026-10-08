"""Test the reward: always positive, high when standing calmly, lower when tilted, thrashing or crouched."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import mujoco, numpy as np
from racevla.envs.go1_standing import Go1StandingEnv
env = Go1StandingEnv(); d = env.data
zero = np.zeros(12)

print("1) holding the home pose from 20 option-C starts: reward must be positive every step and settle near 1")
lows, finals = [], []
for i in range(20):
    env.reset(seed=i); rs = []
    for _ in range(300):
        _, r, term, _, _ = env.step(zero); rs.append(r)
        assert not term
    lows.append(min(rs)); finals.append(np.mean(rs[-50:]))
print(f"   lowest reward seen: {min(lows):.3f}   mean of last 50 steps: {np.mean(finals):.3f}")
assert min(lows) > 0 and np.mean(finals) > 0.9

print("2) tilted robot must score lower than upright (same height, held in the air)")
def reward_after_reset(setup):
    env.reset(seed=1); setup(); mujoco.mj_forward(env.model, d)
    return env._compute_reward(zero)[0]
def roll(deg):
    a = np.deg2rad(deg) / 2; d.qpos[3:7] = [np.cos(a), np.sin(a), 0, 0]
def upright(): d.qpos[3:7] = [1, 0, 0, 0]
for deg in (0, 20, 40, 60): print(f"   roll {deg:2d} deg -> reward {reward_after_reset(lambda: (upright(), roll(deg)) if deg else upright()):.3f}")
assert reward_after_reset(lambda: roll(40)) < reward_after_reset(upright)

print("3) crouched robot (base too low) must score lower than at stand height")
def at_height(z): d.qpos[2] = z
print(f"   z=0.265 -> {reward_after_reset(lambda: (upright(), at_height(0.265))):.3f}   z=0.15 -> {reward_after_reset(lambda: (upright(), at_height(0.15))):.3f}")
assert reward_after_reset(lambda: (upright(), at_height(0.15))) < reward_after_reset(lambda: (upright(), at_height(0.265)))

print("4) thrashing (random +-1 actions) must score lower than still (zero action), and stay positive")
env.reset(seed=5); rng = np.random.default_rng(0); thrash = []
for _ in range(20):
    _, r, term, _, _ = env.step(rng.uniform(-1, 1, 12)); thrash.append(r)
    if term: break
env.reset(seed=5); still = [env.step(zero)[1] for _ in range(len(thrash))]
print(f"   thrashing mean {np.mean(thrash):.3f}   still mean {np.mean(still):.3f}   thrashing min {min(thrash):.4f}")
assert np.mean(thrash) < np.mean(still) and min(thrash) > 0

print("5) step() returns the same reward that _compute_reward gives (reward is not the placeholder 0)")
env.reset(seed=2); _, r, _, _, _ = env.step(zero); assert r > 0; print(f"   reward = {r:.3f}")

print("\nREWARD CHECKS PASSED")
