"""Step 14: the reward breakdown in info, and a side-by-side of a calm stand vs a thrashing robot (for tuning)."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import numpy as np
from racevla.envs.go1_standing import Go1StandingEnv
env = Go1StandingEnv()
KEYS = ["r_upright", "r_height", "r_pose", "r_bracket", "r_action_rate", "r_joint_vel_sq", "r_discount"]

print("1) every step() must put all reward terms in info, and they must reproduce the reward")
env.reset(seed=0); _, r, _, _, info = env.step(np.zeros(12))
assert all(k in info for k in KEYS), [k for k in KEYS if k not in info]
assert abs(info["r_bracket"] * info["r_discount"] - r) < 1e-12
print("   keys present, bracket * discount == reward")

def run(policy, seeds=10, steps=200):
    """Mean of each term over steps 50..steps (skips the landing), averaged over several starts.
    If a policy falls before step 50 every time (e.g. a wide action range makes thrashing fatal fast),
    falls back to whatever steps it did survive, so the test still reports a number instead of crashing."""
    rows, fallback = [], []
    for s in seeds if not isinstance(seeds, int) else range(seeds):
        env.reset(seed=s); rng = np.random.default_rng(s)
        for t in range(steps):
            _, r, term, _, info = env.step(policy(rng))
            row = [r] + [info[k] for k in KEYS]
            fallback.append(row)
            if term: break
            if t >= 50: rows.append(row)
    return np.mean(rows, axis=0) if rows else np.mean(fallback, axis=0)

print("2) calm stand vs thrashing (mean over steps 50-200, 10 starts)")
calm = run(lambda rng: np.zeros(12))
thrash = run(lambda rng: rng.uniform(-1, 1, 12))
print(f"   {'':16s}{'calm':>10s}{'thrashing':>12s}")
for name, a, b in zip(["reward"] + KEYS, calm, thrash):
    print(f"   {name:16s}{a:10.4f}{b:12.4f}")
print(f"\n   target: calm discount > 0.9 (got {calm[-1]:.3f}), thrashing discount < 0.5 (got {thrash[-1]:.3f})")
assert calm[-1] > 0.9

print("\nREWARD TERMS CHECKS PASSED")
