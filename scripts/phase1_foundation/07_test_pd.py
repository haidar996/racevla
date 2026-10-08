"""Validate our PD against MuJoCo's built-in position actuators, then look at the effect of kd."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import mujoco, numpy as np
from racevla.robots.go1 import load_model, reset_home
from racevla.controllers.pd import PDController
from racevla.envs.action import action_to_target
from racevla.robots.go1 import ctrl_limits
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(1); acts = rng.uniform(-1, 1, (20, 12))   # 20 random targets, 0.1 s each

def rollout(mode, kp=100.0, kd=0.0, steps_per=50):
    model, data = load_model(); home = reset_home(model, data)
    lo, hi = ctrl_limits(model); pd = PDController(model, kp, kd) if mode == "own" else None
    traj = []
    for a in acts:
        tgt = action_to_target(a, home, lo, hi)
        for _ in range(steps_per):
            if pd: pd.apply(data, tgt)
            else: data.ctrl[:] = tgt
            mujoco.mj_step(model, data); traj.append(data.qpos.copy())
    return np.array(traj)

print("1) our PD (kp=100, kd=0) must reproduce the built-in position actuators")
a, b = rollout("builtin"), rollout("own")
print(f"   max |qpos difference| over 2 s of random targets: {np.abs(a-b).max():.2e}  ->", "MATCH" if np.abs(a-b).max() < 1e-6 else "MISMATCH")

print("\n2) step response of one thigh joint (target +0.25 rad), by kd")
print(f"   {'kd':>5}{'overshoot rad':>15}{'settle time s':>15}{'final err rad':>15}{'peak |torque|':>15}")
for kd in (0.0, 0.5, 1.0, 2.0, 4.0):
    model, data = load_model(); home = reset_home(model, data); pd = PDController(model, 100.0, kd)
    tgt = home.copy(); tgt[1] += 0.25; qs, taus = [], []
    for _ in range(500):
        pd.apply(data, tgt); taus.append(abs(data.ctrl[1])); mujoco.mj_step(model, data); qs.append(data.qpos[8])
    qs = np.array(qs); final = qs[-1]; over = max(0, qs.max() - tgt[1])
    settled = np.where(np.abs(qs - final) > 0.02 * 0.25)[0]; ts = (settled[-1] + 1) * model.opt.timestep if len(settled) else 0
    print(f"   {kd:5.1f}{over:15.4f}{ts:15.3f}{tgt[1]-final:15.4f}{max(taus):15.1f}")
