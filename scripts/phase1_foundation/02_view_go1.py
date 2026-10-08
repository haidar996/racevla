"""Open the interactive MuJoCo viewer with the Go1 holding its home pose (needs a display)."""
import sys, time; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import mujoco, mujoco.viewer
from racevla.robots.go1 import load_model, reset_home

model, data = load_model()
target = reset_home(model, data)
with mujoco.viewer.launch_passive(model, data) as v:
    while v.is_running():
        t0 = time.time()
        data.ctrl[:] = target
        mujoco.mj_step(model, data)
        v.sync()
        time.sleep(max(0, model.opt.timestep - (time.time() - t0)))
