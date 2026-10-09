"""OnboardCamera (Phase 10, vision): the depth camera mounted at the front of the Go1 trunk (the camera 'onboard' in assets/robots/unitree_go1/go1.xml: 0.32 m ahead of the trunk centre, 0.04 m up, pitched 25 deg down,
58 deg vertical field of view). render(data) returns a (height, width) float32 array of DEPTH in metres: the distance along the camera's viewing axis to the first surface, clipped to max_depth (sky / very far = max_depth).
The camera is rigidly attached to the trunk, so the picture tilts and shakes with the robot's body, as a real camera would. Set MUJOCO_GL=egl BEFORE importing mujoco for off-screen rendering on a machine without a display."""
import mujoco
import numpy as np

CAMERA_NAME = "onboard"


def refresh_hfield(renderer, model):
    """MuJoCo's renderer copies a height field to the graphics card ONCE, when the renderer is created. racevla/envs/terrain.py rewrites the height data for every new terrain, which the physics sees at once but the renderer does not:
    without this call the camera keeps drawing the terrain that was loaded when it was created (a flat floor). Call it after the terrain changes (OnboardCamera.render does it by itself)."""
    for i in range(model.nhfield): mujoco.mjr_uploadHField(model, renderer._mjr_context, i)


class OnboardCamera:
    def __init__(self, model, width=64, height=48, max_depth=5.0):
        self.model, self.width, self.height, self.max_depth = model, width, height, float(max_depth); self._terrain_hash = None
        self.renderer = mujoco.Renderer(model, height, width); self.renderer.enable_depth_rendering()

    def render(self, data):
        h = hash(self.model.hfield_data.tobytes()) if self.model.nhfield else None
        if h != self._terrain_hash: refresh_hfield(self.renderer, self.model); self._terrain_hash = h          # the terrain changed since the last picture
        self.renderer.update_scene(data, camera=CAMERA_NAME)
        return np.minimum(self.renderer.render(), self.max_depth).astype(np.float32)

    def close(self): self.renderer.close()
