"""HeightMapper (Phase 10, V2): an elevation map built from the onboard depth camera, with NO access to the simulator's height field.
Every call to update(data): render the depth image, turn each pixel into a 3-D point with the camera geometry (pinhole model: focal length from the field of view; pixel (u, v) at depth d -> camera frame (d * x/f, d * y/f, -d), the camera looks
along its -z axis), move the points into the world with the camera pose, and write the highest point of every 4 cm map cell (same grid as the terrain) into the map; a cell keeps its most recent observation, so the nearest (most accurate) view wins.
query(x, y) = the estimated ground height (m, world frame) at world points (any array shape): the nearest observed cell within 2 cells, else the FALLBACK = the mean height of the two lowest feet (known from the joint angles and the body orientation),
i.e. 'the ground under the robot'. It has the same signature as TerrainMixin.ground_height_at, so a body can use it instead: body.ground_height_at = mapper.query (see attach()). The scan and the step-over foot-lift rule then use the camera's map.
What is still taken from the simulator (a real robot would estimate it): the camera pose in the world (position and orientation of the trunk: state estimator), the feet heights for the fallback (kinematics + the same pose), optional noise on the camera height (pose_noise)."""
import numpy as np
import mujoco

from racevla.envs.terrain import X0, X1, Y_HALF, NCOL, NROW
from racevla.vision.camera import OnboardCamera, CAMERA_NAME


class HeightMapper:
    def __init__(self, body, width=64, height=48, max_depth=5.0, pose_noise=0.0, seed=0):
        self.body, self.model = body, body.model; self.camera = OnboardCamera(body.model, width, height, max_depth); self.cam_id = mujoco.mj_name2id(body.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        f = (height / 2.0) / np.tan(np.radians(body.model.cam_fovy[self.cam_id]) / 2.0)
        u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5); self.rays = np.stack([(u - width / 2.0) / f, -(v - height / 2.0) / f, -np.ones_like(u)], axis=-1).reshape(-1, 3)      # camera frame, per unit of depth
        self.max_depth, self.pose_noise, self.rng = max_depth, float(pose_noise), np.random.default_rng(seed)
        self.ncol, self.x1 = getattr(body, 'ncol', NCOL), getattr(body, 'x1', X1)            # the course length (10 m, or the 46 m race course)
        self.H = np.full((NROW, self.ncol), np.nan, np.float32); self.ground_est = 0.0
        self.offsets = sorted([(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3)], key=lambda o: o[0] ** 2 + o[1] ** 2)           # nearest cell first

    def reset(self): self.H[:] = np.nan; self.pose_bias = float(self.rng.normal(0.0, self.pose_noise)) if self.pose_noise > 0 else 0.0

    def _index(self, x, y):
        return (np.clip(np.rint((np.asarray(y, float) + Y_HALF) / (2 * Y_HALF) * (NROW - 1)), 0, NROW - 1).astype(int), np.clip(np.rint((np.asarray(x, float) - X0) / (self.x1 - X0) * (self.ncol - 1)), 0, self.ncol - 1).astype(int))

    def update(self, data):
        img = self.camera.render(data); self.last_depth = img; d = img.reshape(-1); ok = (d > 0.05) & (d < self.max_depth - 1e-3)                    # no return (sky / beyond max depth) -> no point
        R = data.cam_xmat[self.cam_id].reshape(3, 3); pos = data.cam_xpos[self.cam_id]
        pts = pos + (d[ok, None] * self.rays[ok]) @ R.T; pts[:, 2] += getattr(self, "pose_bias", 0.0)             # world frame
        iy, ix = self._index(pts[:, 0], pts[:, 1]); tmp = np.full((NROW, self.ncol), -np.inf, np.float32); np.maximum.at(tmp, (iy, ix), pts[:, 2].astype(np.float32))
        seen = tmp > -np.inf; self.H[seen] = tmp[seen]
        foot = np.sort(data.geom_xpos[self.body.foot_ids, 2] - self.body.foot_radius); self.ground_est = float(foot[:2].mean())          # ground under the robot from the two lowest feet

    def query(self, x, y):
        x = np.asarray(x, float); shape = x.shape; iy, ix = self._index(x.ravel(), np.asarray(y, float).ravel()); out = np.full(ix.shape, np.nan)
        for dy, dx in self.offsets:
            v = self.H[np.clip(iy + dy, 0, NROW - 1), np.clip(ix + dx, 0, self.ncol - 1)]; m = np.isnan(out) & ~np.isnan(v); out[m] = v[m]
        out[np.isnan(out)] = self.ground_est; return out.reshape(shape)

    def attach(self):
        """Make the body read the camera's map instead of the simulator's height field."""
        self.body.ground_height_at = self.query; return self
