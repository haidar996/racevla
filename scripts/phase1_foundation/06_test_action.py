"""Check the action mapping, then check it physically drives the robot."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import mujoco, numpy as np
from racevla.robots.go1 import load_model, reset_home, ctrl_limits
from racevla.envs.action import action_to_target, ACTION_SCALE

model, data = load_model(); reset_home(model, data)
home = data.qpos[7:].copy(); lo, hi = ctrl_limits(model)
np.set_printoptions(precision=3, suppress=True)
T = lambda a: action_to_target(a, home, lo, hi)

print("1) zero action -> exactly the home pose")
assert np.allclose(T(np.zeros(12)), home); print("   ok")

print("2) +1 / -1 move each joint by exactly +/-0.25 rad (unless a limit stops it)")
up, dn = T(np.ones(12)), T(-np.ones(12))
print("   +1 offset:", up - home); print("   -1 offset:", dn - home)

print("3) out-of-range action (5.0) is treated like 1.0")
assert np.allclose(T(np.full(12, 5.0)), up); print("   ok")

print("4) targets never leave the joint limits, even with extreme actions")
rng = np.random.default_rng(0); worst = 0
for _ in range(10000):
    t = T(rng.normal(0, 10, 12)); assert np.all(t >= lo - 1e-9) and np.all(t <= hi + 1e-9)
print("   ok (10000 random actions)")

print("5) physical: apply +1 on all joints for 1 s, robot should crouch/shift but not explode")
for k in range(500):
    data.ctrl[:] = T(np.ones(12)); mujoco.mj_step(model, data)
print(f"   base z {data.qpos[2]:.3f} m | max |joint - target| {np.abs(data.qpos[7:] - data.ctrl).max():.3f} rad | finite: {np.isfinite(data.qpos).all()}")
print(f"\nACTION_SCALE = {ACTION_SCALE} rad")
