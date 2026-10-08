"""Joint-position control test: (a) hold home pose, (b) sine sweep on each joint.
Headless by default; pass --view to watch. Position actuators already are the PD controller."""
import argparse, sys, time; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import mujoco, numpy as np
from racevla.robots.go1 import load_model, reset_home, ctrl_limits, JOINTS

ap = argparse.ArgumentParser(); ap.add_argument("--view", action="store_true"); args = ap.parse_args()
model, data = load_model()
home = reset_home(model, data)
lo, hi = ctrl_limits(model)

def run(ctrl_fn, seconds, viewer=None):
    n = int(seconds / model.opt.timestep); zs, errs = [], []
    for k in range(n):
        data.ctrl[:] = np.clip(ctrl_fn(k * model.opt.timestep), lo, hi)
        mujoco.mj_step(model, data)
        zs.append(data.qpos[2]); errs.append(np.abs(data.qpos[7:] - data.ctrl).max())
        if viewer: viewer.sync(); time.sleep(model.opt.timestep)
    return np.array(zs), np.array(errs)

def experiment(viewer=None):
    z, e = run(lambda t: home, 3.0, viewer)
    print(f"[hold home 3s]  base z: {z[0]:.3f} -> {z[-1]:.3f} m | max joint err (last 0.5s): {e[-250:].max():.3f} rad")
    up = data.xmat[1].reshape(3, 3)[2, 2]
    print(f"                upright (body z-axis . world z) = {up:.3f}  ({'STANDING' if z[-1] > 0.2 and up > 0.9 else 'FELL/COLLAPSED'})")
    reset_home(model, data)
    for j, name in enumerate(JOINTS):
        def f(t, j=j):
            c = home.copy(); c[j] += 0.3 * np.sin(2 * np.pi * 1.0 * t); return c
        z, e = run(f, 1.5, viewer)
        print(f"[sweep {name:8}] tracking err max {e.max():.3f} rad | base z min {z.min():.3f}")
        reset_home(model, data)

if args.view:
    import mujoco.viewer
    with mujoco.viewer.launch_passive(model, data) as v: experiment(v)
else:
    experiment()
