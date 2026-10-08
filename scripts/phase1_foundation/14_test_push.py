"""Test the push option: off by default, well-timed and reproducible when on, and the size range matches what was measured."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import numpy as np
import racevla.envs.go1_standing as mod
from racevla.envs.go1_standing import Go1StandingEnv, PUSH_GRACE_STEPS, PUSH_INTERVAL_STEPS, PUSH_KICK_RANGE
zero = np.zeros(12)

def push_log(env, seed, steps=1000):
    """Return [(step, kick)] for every push in one zero-action episode (runs to the end, ignores termination)."""
    env.reset(seed=seed); out = []
    for t in range(steps):
        _, _, term, trunc, info = env.step(zero)
        if info["push_kick"] > 0: out.append((t, info["push_kick"]))
        if term or trunc: break
    return out

print("1) push=False (default): never pushes, and info always has push_kick == 0.0")
env = Go1StandingEnv(); assert push_log(env, 0) == []; print("   no pushes")

print("2) push=True: timing and size (20 episodes to the end, or until the robot falls)")
env = Go1StandingEnv(push=True); firsts, gaps, kicks = [], [], []
for s in range(20):
    log = push_log(env, s)
    if not log: continue
    firsts.append(log[0][0]); kicks += [k for _, k in log]; gaps += list(np.diff([t for t, _ in log]))
print(f"   pushes seen: {len(kicks)}   first push at step {min(firsts)}..{max(firsts)} (grace {PUSH_GRACE_STEPS})")
print(f"   gap between pushes: {min(gaps)}..{max(gaps)} steps (allowed {PUSH_INTERVAL_STEPS[0]}..{PUSH_INTERVAL_STEPS[1] - 1})   kick size: {min(kicks):.2f}..{max(kicks):.2f} m/s (allowed {PUSH_KICK_RANGE})")
assert min(firsts) >= PUSH_GRACE_STEPS + PUSH_INTERVAL_STEPS[0] - 1 and min(gaps) >= PUSH_INTERVAL_STEPS[0] and max(gaps) < PUSH_INTERVAL_STEPS[1]
assert PUSH_KICK_RANGE[0] <= min(kicks) and max(kicks) <= PUSH_KICK_RANGE[1]

print("3) same seed -> exactly the same pushes")
a, b = push_log(Go1StandingEnv(push=True), 7), push_log(Go1StandingEnv(push=True), 7)
assert a == b and len(a) > 0; print(f"   {len(a)} pushes, identical")

print("4) push=False resets are not disturbed by the push option (same seed -> same first observation)")
o1, _ = Go1StandingEnv().reset(seed=11); o2, _ = Go1StandingEnv(push=True).reset(seed=11)
assert np.array_equal(o1, o2); print("   identical")

print("5) zero-action robot vs pushes of a fixed size (24 episodes of 300 steps): small ones survive, large ones topple")
def fall_rate(size):
    mod.PUSH_KICK_RANGE = (size, size)
    env = Go1StandingEnv(push=True); falls = 0
    for s in range(24):
        env.reset(seed=s)
        for _ in range(300):
            _, _, term, _, _ = env.step(zero)
            if term: falls += 1; break
    mod.PUSH_KICK_RANGE = PUSH_KICK_RANGE
    return falls
small, large = fall_rate(0.6), fall_rate(2.0)
print(f"   0.6 m/s pushes: {small}/24 fell     2.0 m/s pushes: {large}/24 fell")
assert small == 0 and large >= 20

print("\nPUSH CHECKS PASSED")
