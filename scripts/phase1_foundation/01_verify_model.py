"""Load the Go1 and print what the RL env will need: joints, actuators, limits, sensors."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import mujoco, numpy as np
from racevla.robots.go1 import load_model, reset_home, JOINTS

model, data = load_model()
home = reset_home(model, data)
print(f"MuJoCo {mujoco.__version__} | bodies={model.nbody} geoms={model.ngeom} nq={model.nq} nv={model.nv} nu={model.nu}")
print(f"timestep={model.opt.timestep}s  mass={model.body_subtreemass[1]:.2f} kg  base z={data.qpos[2]:.3f} m\n")
lo, hi = model.actuator_ctrlrange.T
print(f"{'actuator':10}{'home':>8}{'lo':>8}{'hi':>8}{'kp':>6}{'max torque':>12}")
for i, name in enumerate(JOINTS):
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) == name
    print(f"{name:10}{home[i]:8.2f}{lo[i]:8.2f}{hi[i]:8.2f}{model.actuator_gainprm[i,0]:6.0f}{model.actuator_forcerange[i,1]:12.1f}")
print("\nsensors:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)] or "none")
print("cameras:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)] or "none")
