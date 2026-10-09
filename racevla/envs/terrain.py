"""Terrain for the Go1 environments (Phase 6): a height field written into the scene at run time + a mixin that adds it to any environment class.
Course: x runs from -1 m to 9 m, y from -1.5 to 1.5 m (250 x 75 height-field points, 4 cm apart). The robot starts at x = 0 on a flat 2 m lead-in (x < 1), heading along +x, and the terrain section follows (x from 1 m).
Terrain kinds and difficulty levels (each level is one number):
  rough       random bumps over x from 1 to 7 m: heights uniform in [0, A] on a 0.1 m grid, smoothly interpolated;  A = 0, 1, 2, 3, 4, 5, 6, 8, 10 cm
  slope_up    ramp of angle a from x = 1 to 5 m, then a flat plateau;                                              a = 3, 6, 9, 12, 15 degrees
  slope_down  starts on a plateau (its height = the ramp's rise), ramp of angle a down from x = 1 to 5 m, flat below;  a = 3, 6, 9, 12, 15 degrees
  stairs_up   5 steps of height h and tread 0.4 m starting at x = 1, then a plateau;                              h = 2, 4, 6, 8, 10, 12, 15 cm
  stairs_down starts on a plateau, 5 steps down of height h and tread 0.4 m starting at x = 1;                    h = 2, 4, 6, 8, 10, 12, 15 cm
The geom is named 'floor' (like the flat plane), so every contact test that looks for the floor works unchanged.
Success of a crossing = the robot's x reaches X_GOAL; leaving the course sideways (|y| > Y_LIMIT) ends the episode ('left_course')."""
import pathlib
import mujoco
import numpy as np

from racevla.robots.go1 import ROOT

TERRAIN_XML = ROOT / "assets" / "robots" / "unitree_go1" / "scene_terrain.xml"
NROW, NCOL = 75, 250
X0, X1, Y_HALF = -1.0, 9.0, 1.5                 # course limits (m)
ELEV_Z = 1.5                                     # m, the height that data value 1.0 stands for (must match the XML)
X_TERRAIN_START, X_TERRAIN_END_ROUGH = 1.0, 7.0
RAMP_LENGTH, N_STEPS, TREAD = 4.0, 5, 0.4
X_GOAL, Y_LIMIT = 6.0, 1.4
LEVELS = {"rough": [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10], "slope_up": [3, 6, 9, 12, 15], "slope_down": [3, 6, 9, 12, 15],
          "stairs_up": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15], "stairs_down": [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]}
XS, YS = np.linspace(X0, X1, NCOL), np.linspace(-Y_HALF, Y_HALF, NROW)
X_HURDLE, HURDLE_THICK = 5.0, 0.08                           # hurdle kind (jumping skill): a bar across the WHOLE width of the course at x = 5 m, 8 cm thick (the grid is 4 cm), height = the level
HURDLE_LEVELS = [0.03, 0.05, 0.07, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]     # m (kept apart from LEVELS so that the walking / stairs code that loops over LEVELS is unchanged)


def _bumps(amplitude, rng):
    """Random heights in [0, amplitude] on a 0.1 m grid, bilinearly interpolated to the height-field grid, only over the rough section."""
    cx, cy = np.arange(X0, X1 + 0.1, 0.1), np.arange(-Y_HALF, Y_HALF + 0.1, 0.1)
    coarse = rng.uniform(0.0, amplitude, (len(cy), len(cx))) if amplitude > 0 else np.zeros((len(cy), len(cx)))
    along_x = np.array([np.interp(XS, cx, row) for row in coarse])                  # (len(cy), NCOL)
    h = np.array([np.interp(YS, cy, along_x[:, i]) for i in range(NCOL)]).T          # (NROW, NCOL)
    mask = (XS >= X_TERRAIN_START) & (XS <= X_TERRAIN_END_ROUGH)
    return h * mask[None, :]


def height_field(kind, level, seed=None):
    """Returns (H, z_start): heights in metres, shape (NROW, NCOL) (row = y, column = x), and the ground height at the start position."""
    rng = np.random.default_rng(seed); x = XS[None, :]
    if kind == "rough": H, z0 = _bumps(level, rng), 0.0
    elif kind in ("slope_up", "slope_down"):
        rise = np.tan(np.deg2rad(level)) * RAMP_LENGTH; ramp = np.clip(XS - X_TERRAIN_START, 0.0, RAMP_LENGTH) / RAMP_LENGTH * rise
        prof = ramp if kind == "slope_up" else rise - ramp; H, z0 = np.tile(prof, (NROW, 1)), float(prof[0])
    elif kind == "hurdle":
        prof = np.where((XS >= X_HURDLE) & (XS <= X_HURDLE + HURDLE_THICK), float(level), 0.0); H, z0 = np.tile(prof, (NROW, 1)), 0.0
    elif kind in ("stairs_up", "stairs_down"):
        k = np.clip(np.floor((XS - X_TERRAIN_START) / TREAD) + 1.0, 0.0, N_STEPS); prof = level * k if kind == "stairs_up" else level * (N_STEPS - k)
        H, z0 = np.tile(prof, (NROW, 1)), float(prof[0])
    else: raise ValueError(f"unknown terrain kind {kind}")
    assert H.max() < ELEV_Z and H.min() >= 0.0
    return H, z0


class TerrainMixin:
    """Put FIRST in the class list:  class Terrain...(TerrainMixin, Go1WalkingSineEnv): pass  (see make_terrain_env). reset(options={'terrain': (kind, level), 'terrain_seed': int, 'yaw': rad}).
    A subclass can use a longer course by overriding the three class attributes below (racevla/envs/race.py: 46 m long); the height field in the scene file must have ncol points from x = X0 to x = x1."""
    scene_path, ncol, x1 = TERRAIN_XML, NCOL, X1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, scene_xml=str(self.scene_path), **kwargs)
        m = self.model; self.hf_id = int(m.geom_dataid[self.floor_id]); assert m.hfield_nrow[self.hf_id] == NROW and m.hfield_ncol[self.hf_id] == self.ncol and abs(m.hfield_size[self.hf_id][2] - ELEV_Z) < 1e-9
        self.terrain = ("rough", 0.0)
        if self.ncol == NCOL: self.set_terrain("rough", 0.0, 0)
        else: self.set_height_field(np.zeros((NROW, self.ncol)), 0.0, ("rough", 0.0))                  # a longer course starts flat

    def set_terrain(self, kind, level, seed=None):
        H, z0 = height_field(kind, level, seed); self.set_height_field(H, z0, (kind, level))

    def set_height_field(self, H, z0, label):
        """Write a ready-made height array (NROW, ncol), in metres, into the scene; z0 = the ground height at the start; label = what self.terrain reports."""
        m = self.model; adr, n = int(m.hfield_adr[self.hf_id]), NROW * self.ncol; assert H.shape == (NROW, self.ncol) and H.max() < ELEV_Z and H.min() >= 0.0
        m.hfield_data[adr:adr + n] = (H / ELEV_Z).ravel().astype(np.float32); self.terrain, self.ground_height, self._H = label, z0, H

    def reset(self, seed=None, options=None):
        if options and options.get("height_field") is not None: self.set_height_field(*options["height_field"])          # (H, z0, label): a ready-made course
        elif options and options.get("terrain") is not None:
            self.set_terrain(*options["terrain"], seed=options.get("terrain_seed", seed))
        return super().reset(seed=seed, options=options)

    def ground_height_at(self, x, y):
        """Terrain height at world points (x, y) (arrays allowed) by bilinear interpolation of the stored height array (fast, no ray casting); points off the course are clamped to the border."""
        fx = np.clip((np.asarray(x, float) - X0) / (self.x1 - X0) * (self.ncol - 1), 0, self.ncol - 1.000001); fy = np.clip((np.asarray(y, float) + Y_HALF) / (2 * Y_HALF) * (NROW - 1), 0, NROW - 1.000001)
        ix, iy = fx.astype(int), fy.astype(int); ax, ay = fx - ix, fy - iy; H = self._H
        return (H[iy, ix] * (1 - ax) * (1 - ay) + H[iy, ix + 1] * ax * (1 - ay) + H[iy + 1, ix] * (1 - ax) * ay + H[iy + 1, ix + 1] * ax * ay)

    def ground_z(self, x, y=0.013):
        """Height of the terrain at (x, y), measured by a ray from above against the height field only (the robot is not hit). The default y avoids the exact grid line y = 0, where a ray can slip between two prisms."""
        z = mujoco.mj_rayHfield(self.model, self.data, self.floor_id, np.array([x, y, 3.0]), np.array([0.0, 0.0, -1.0]))
        return 3.0 - z if z >= 0 else float("nan")

    def _is_terminated(self) -> bool:
        if abs(self.data.qpos[1]) > Y_LIMIT: self.termination_reason = "left_course"; return True
        return super()._is_terminated()


def make_terrain_env(base_cls):
    """Terrain version of an environment class (Go1WalkingEnv, Go1WalkingSineEnv, Go1RunningSineEnv, ...)."""
    return type("Terrain" + base_cls.__name__, (TerrainMixin, base_cls), {})
