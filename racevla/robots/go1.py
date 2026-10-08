"""Unitree Go1 constants and model loading (MuJoCo Menagerie MJCF)."""
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE_XML = ROOT / "assets" / "robots" / "unitree_go1" / "scene.xml"

LEGS = ("FR", "FL", "RR", "RL")
JOINTS = tuple(f"{leg}_{j}" for leg in LEGS for j in ("hip", "thigh", "calf"))  # actuator order
HOME_KEY = "home"


def load_model(xml_path=SCENE_XML):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    return model, mujoco.MjData(model)


def reset_home(model, data):
    """Reset to the 'home' keyframe (standing pose) and hold it with position targets."""
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, HOME_KEY)
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key] if model.nkey and np.any(model.key_ctrl[key]) else data.qpos[7:]
    mujoco.mj_forward(model, data)
    return data.ctrl.copy()


def ctrl_limits(model):
    """Joint-angle limits of the 12 actuated joints (valid in both position and torque actuator modes)."""
    rng = model.jnt_range[model.actuator_trnid[:, 0]]
    return rng[:, 0].copy(), rng[:, 1].copy()
