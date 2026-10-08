"""Test termination rules: no false alarms while standing, each rule fires when it should."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import mujoco, numpy as np
from racevla.envs.go1_standing import Go1StandingEnv, MAX_TILT
env = Go1StandingEnv(); d = env.data
zero = np.zeros(12)

print("1) holding the home pose (zero action) from 100 different option-C starts: must NEVER terminate")
bad = 0
for i in range(100):
    env.reset(seed=i)
    for _ in range(300):
        _, _, term, trunc, info = env.step(zero)
        if term: bad += 1; print("   false alarm, seed", i, info["termination_reason"]); break
print(f"   false alarms: {bad}/100"); assert bad == 0

print("2) limp robot (zero torque) must terminate by body_contact")
env.reset(seed=3); n = 0
while True:
    d.ctrl[:] = 0
    for _ in range(10): mujoco.mj_step(env.model, d)
    n += 1
    if env._is_terminated() or n > 200: break
print(f"   reason={env.termination_reason} after {n} steps ({n*0.02:.2f} s)"); assert env.termination_reason == "body_contact"

def tilted_step(deg):
    env.reset(seed=1); a = np.deg2rad(deg) / 2
    d.qpos[3:7] = [np.cos(a), np.sin(a), 0, 0]; d.qpos[2] = 1.0; d.qvel[:] = 0    # roll, held high so nothing touches the floor
    mujoco.mj_forward(env.model, d)
    _, _, term, _, info = env.step(zero); return term, info.get("termination_reason")

print("3) tilt rule, robot held in the air (limit = 60 deg)")
for deg in (30, 55, 65, 100):
    print(f"   roll {deg:3d} deg -> terminated={tilted_step(deg)}")
assert tilted_step(55)[0] is False and tilted_step(65) == (True, "tilt")

print("4) bad numerical state must terminate")
env.reset(seed=1); d.qvel[0] = np.nan; env.step(zero); print(f"   NaN velocity -> terminated={env._is_terminated()} reason={env.termination_reason}"); assert env.termination_reason == "bad_state"

print("5) step() reports it and truncation is separate")
env.reset(seed=3)
for k in range(300):
    d.ctrl[:] = 0; _, r, term, trunc, info = env.step(zero)   # PD holds it, so it should NOT end here
print(f"   after 300 held steps: terminated={term} truncated={trunc}")
print("\nTERMINATION CHECKS PASSED")
