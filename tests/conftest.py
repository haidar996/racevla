"""pytest setup: render off-screen with EGL when the EGL library is installed (the depth camera test); MuJoCo reads MUJOCO_GL when it is first imported, so this has to happen before any test module imports it.
Where there is no EGL (for example the CI machine), MUJOCO_GL is left alone: importing mujoco must keep working, and the camera test skips itself."""
import ctypes.util
import os

if ctypes.util.find_library("EGL"): os.environ.setdefault("MUJOCO_GL", "egl")
