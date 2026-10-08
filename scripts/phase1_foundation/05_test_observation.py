"""Check that each observation component is correct using cases where we know the answer."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
import mujoco, numpy as np
from racevla.robots.go1 import load_model, reset_home
from racevla.envs.observation import build_obs, OBS_DIM, OBS_SLICES as S

model, data = load_model(); reset_home(model, data)
home_joints = data.qpos[7:].copy(); zero_a = np.zeros(12)
np.set_printoptions(precision=3, suppress=True)

obs = build_obs(data, home_joints, zero_a)
print(f"shape={obs.shape} (expected {OBS_DIM}) dtype={obs.dtype}")
assert obs.shape == (OBS_DIM,)

print("\n1) At home, upright, at rest")
print("   joint_pos (want all 0):", obs[S['joint_pos']])
print("   gravity   (want 0 0 -1):", obs[S['gravity']])
assert np.allclose(obs[S['joint_pos']], 0) and np.allclose(obs[S['gravity']], [0, 0, -1], atol=1e-6)

print("\n2) Yaw the robot 90 deg (turn on the spot). Gravity must not change; world +x velocity must become body -y")
reset_home(model, data)
data.qpos[3:7] = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]   # 90 deg about world z
data.qvel[0:3] = [1.0, 0, 0]                                 # moving along world +x
mujoco.mj_forward(model, data)
o = build_obs(data, home_joints, zero_a)
print("   gravity (want 0 0 -1):", o[S['gravity']], "\n   lin_vel (want 0 -1 0):", o[S['lin_vel']])
assert np.allclose(o[S['gravity']], [0, 0, -1], atol=1e-6) and np.allclose(o[S['lin_vel']], [0, -1, 0], atol=1e-6)

print("\n3) Pitch the robot 30 deg nose-down. Gravity must gain an x component of about sin(30)=0.5")
reset_home(model, data)
a = np.deg2rad(30) / 2; data.qpos[3:7] = [np.cos(a), 0, np.sin(a), 0]  # rotation about world y
mujoco.mj_forward(model, data)
o = build_obs(data, home_joints, zero_a)
print("   gravity:", o[S['gravity']], "(|g| =", np.linalg.norm(o[S['gravity']]).round(3), ")")
assert abs(np.linalg.norm(o[S['gravity']]) - 1) < 1e-6 and abs(abs(o[S['gravity']][0]) - 0.5) < 1e-3

print("\n4) Joint offset and previous action land in the right slots")
reset_home(model, data); data.qpos[7] += 0.2; mujoco.mj_forward(model, data)
o = build_obs(data, home_joints, np.arange(12) * 0.1)
print("   joint_pos[0] (want 0.2):", o[0].round(3), "| prev_action[:3] (want 0 .1 .2):", o[S['prev_action']][:3])
assert abs(o[0] - 0.2) < 1e-6 and np.allclose(o[S['prev_action']], np.arange(12) * 0.1)
print("\nALL OBSERVATION CHECKS PASSED")
