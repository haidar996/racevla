"""Race course (Phase 8): ONE long track made of several terrain sections in a row, drawn at random from a seed. Used by the rule supervisor + vision pipeline (scripts/phase8_race).
Layout (x in metres, the height field runs from -1 to 45 m, 3 m wide, 4 cm grid): a flat lead-in to x = START_OF_SECTIONS, then N sections, each followed by GAP m of flat ground, then the finish line.
Sections (same shapes as racevla/envs/terrain.py, but placed anywhere and starting at the CURRENT ground height, so the course stays continuous):
  rough       4 m of random bumps (0.1 m grid, heights 0..A), A from RACE_LEVELS
  slope_up    4 m ramp up, then a plateau;  slope_down: 4 m ramp down to a lower plateau
  stairs_up   5 steps of height h, tread 0.4 m (2 m long), then a plateau;  stairs_down likewise
  bar         a thin full-width bar (8 cm thick) lying on the ground, height h
Levels are chosen inside what the skills can do (RACE_LEVELS); a section that would climb above MAX_Z or dip below zero is replaced by one that fits."""
import pathlib
from dataclasses import dataclass, field

import numpy as np

from racevla.envs.terrain import X0, Y_HALF, NROW, HURDLE_THICK, RAMP_LENGTH, N_STEPS, TREAD, ELEV_Z
from racevla.robots.go1 import ROOT

RACE_XML = ROOT / "assets" / "robots" / "unitree_go1" / "scene_race.xml"
X1_RACE, NCOL_RACE = 45.0, 1150
XS_RACE, YS_RACE = np.linspace(X0, X1_RACE, NCOL_RACE), np.linspace(-Y_HALF, Y_HALF, NROW)
START_OF_SECTIONS, GAP, MAX_Z = 4.0, 3.0, 1.25
KINDS = ("rough", "slope_up", "slope_down", "stairs_up", "stairs_down", "bar")
RACE_LEVELS = {"rough": [0.04, 0.06], "slope_up": [9, 15], "slope_down": [9, 15], "stairs_up": [0.03, 0.04], "stairs_down": [0.04, 0.06], "bar": [0.03, 0.05]}
LENGTH = {"rough": 4.0, "slope_up": RAMP_LENGTH, "slope_down": RAMP_LENGTH, "stairs_up": N_STEPS * TREAD, "stairs_down": N_STEPS * TREAD, "bar": HURDLE_THICK}


@dataclass
class Section:
    kind: str; level: float; x_start: float; x_end: float; z_start: float; z_end: float


@dataclass
class Course:
    H: np.ndarray                  # (NROW, NCOL_RACE) heights in metres
    sections: list = field(default_factory=list)
    finish_x: float = 0.0


def _rise(kind, level):
    return {"slope_up": np.tan(np.deg2rad(level)) * RAMP_LENGTH, "slope_down": -np.tan(np.deg2rad(level)) * RAMP_LENGTH,
            "stairs_up": level * N_STEPS, "stairs_down": -level * N_STEPS}.get(kind, 0.0)


def build_course(seed, n_sections=4):
    """Random course for `seed`: n_sections sections (at most 5 fit), levels from RACE_LEVELS, never two bars in a row (the step-over skill needs room to run up)."""
    rng = np.random.default_rng(seed); base = np.zeros(NCOL_RACE); extra_1d = np.zeros(NCOL_RACE); bumps = np.zeros((NROW, NCOL_RACE)); sections = []; c, z = START_OF_SECTIONS, 0.0; prev = None
    for _ in range(n_sections):
        while True:
            kind = str(rng.choice(KINDS)); level = RACE_LEVELS[kind][int(rng.integers(len(RACE_LEVELS[kind])))]; dz = _rise(kind, level)
            if (kind == "bar" and prev == "bar") or z + dz > MAX_Z or z + dz < -1e-9: continue
            break
        x_end = c + LENGTH[kind]; inside = (XS_RACE >= c) & (XS_RACE <= x_end); u = XS_RACE - c
        if kind == "rough":
            cx, cy = np.arange(X0, X1_RACE + 0.1, 0.1), np.arange(-Y_HALF, Y_HALF + 0.1, 0.1); coarse = rng.uniform(0.0, level, (len(cy), len(cx)))
            along = np.array([np.interp(XS_RACE, cx, row) for row in coarse]); full = np.array([np.interp(YS_RACE, cy, along[:, i]) for i in range(NCOL_RACE)]).T; bumps += full * inside[None, :]
        elif kind in ("slope_up", "slope_down"): base[inside] = z + np.clip(u[inside] / RAMP_LENGTH, 0, 1) * dz
        elif kind in ("stairs_up", "stairs_down"): base[inside] = z + np.clip(np.floor(u[inside] / TREAD) + 1.0, 0, N_STEPS) * (level if dz > 0 else -level)
        elif kind == "bar": extra_1d[inside] = level
        z_new = z + dz; base[XS_RACE > x_end] = z_new          # (every later section overwrites this further on)
        sections.append(Section(kind, float(level), float(c), float(x_end), float(z), float(z_new)))
        prev, z, c = kind, z_new, x_end + GAP
    H = np.maximum(base[None, :] + extra_1d[None, :] + bumps, 0.0); assert H.max() < ELEV_Z
    return Course(H.astype(np.float64), sections, float(c))
