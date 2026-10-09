"""pytest setup: render off-screen with EGL (the depth camera tests); MuJoCo reads MUJOCO_GL when it is first imported, so this has to happen before any test module imports it."""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
