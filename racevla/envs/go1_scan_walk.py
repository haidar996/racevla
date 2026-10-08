"""Go1ScanWalkEnv: Phase 6, Skill B -- the terrain walking environment (Go1TerrainWalkEnv) with a HEIGHT SCAN in the observation and a higher reference foot lift (default 12 cm).
Observation = the 49 numbers of the terrain walking env (45 robot, 2 command, 2 clock) + SCAN_DIM = 18 scan numbers = 67. The scan is a small grid of terrain heights around the robot in a HEADING-ALIGNED frame:
SCAN_DX = 0.15 ... 0.90 m ahead of the base (6 rows) x SCAN_DY = -0.15, 0, +0.15 m to the sides (3 columns); each value = terrain height at that point minus the terrain height under the base (so a step ahead of the robot shows up as
+/- its height, a slope as a ramp, flat ground as zeros), multiplied by SCAN_SCALE = 4 (so a 15 cm step is 0.6) and with Gaussian noise (std scan_noise = 0.5 cm before scaling, training only) so that the policy does not rely on a perfect scan.
In the simulator the heights are read from the height field; later the vision phase supplies the same 18 numbers from a camera.
Terrain mix and ceilings (Skill B): flat 10 %, rough 15 %, slope up 15 %, slope down 10 %, stairs up 25 %, stairs down 25 %; ceilings rough 10 cm, slopes 15 deg, stairs up AND down 15 cm."""
import numpy as np

from racevla.envs.go1_terrain_walk import Go1TerrainWalkEnv
from gymnasium import spaces

SCAN_DX = np.array([0.15, 0.30, 0.45, 0.60, 0.75, 0.90]); SCAN_DY = np.array([-0.15, 0.0, 0.15])
SCAN_DIM = len(SCAN_DX) * len(SCAN_DY)          # 18
SCAN_SCALE = 4.0
SCAN_OBS_DIM = 49 + SCAN_DIM                    # 67
SCAN_TYPE_PROBS = {"flat": 0.10, "rough": 0.15, "slope_up": 0.15, "slope_down": 0.10, "stairs_up": 0.25, "stairs_down": 0.25}
SCAN_CAP = {"rough": 8, "slope_up": 4, "slope_down": 4, "stairs_up": 6, "stairs_down": 6}
DEFAULT_LIFT = 0.12


class Go1ScanWalkEnv(Go1TerrainWalkEnv):
    def __init__(self, lift=DEFAULT_LIFT, scan_noise=0.005, **kwargs):
        super().__init__(lift=lift, **kwargs)
        self.type_probs, self.cap, self.scan_noise = dict(SCAN_TYPE_PROBS), dict(SCAN_CAP), float(scan_noise)
        self.observation_space = spaces.Box(-np.inf, np.inf, (SCAN_OBS_DIM,), np.float32)
        gx, gy = np.meshgrid(SCAN_DX, SCAN_DY, indexing="ij"); self._offsets = np.stack([gx.ravel(), gy.ravel()], axis=1)       # (18, 2): (ahead, left) in the heading frame, row-major over (row, column)

    def scan(self, noise=True):
        """The 18 scan numbers (scaled terrain heights relative to the ground under the base, heading-aligned)."""
        R = self.data.xmat[1].reshape(3, 3); yaw = float(np.arctan2(R[1, 0], R[0, 0])); c, s = np.cos(yaw), np.sin(yaw); bx, by = float(self.data.qpos[0]), float(self.data.qpos[1])
        px = bx + c * self._offsets[:, 0] - s * self._offsets[:, 1]; py = by + s * self._offsets[:, 0] + c * self._offsets[:, 1]
        rel = self.ground_height_at(px, py) - float(self.ground_height_at(bx, by))
        if noise and self.scan_noise > 0: rel = rel + self.np_random.normal(0.0, self.scan_noise, rel.shape)
        return (SCAN_SCALE * rel).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options); return np.concatenate([obs, self.scan()]).astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action); return np.concatenate([obs, self.scan()]).astype(np.float32), reward, terminated, truncated, info
