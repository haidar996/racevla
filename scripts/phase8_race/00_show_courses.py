"""Draw the side view (ground height along x, centre line) of random race courses, to check them by eye. Usage: python 00_show_courses.py [--seeds 4000 4001 ...]"""
import sys, pathlib, argparse; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from racevla.envs.race import build_course, XS_RACE, NROW
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[4000, 4001, 4002, 4003, 4004, 4005]); a = ap.parse_args()
fig, axes = plt.subplots(len(a.seeds), 1, figsize=(12, 1.6 * len(a.seeds)), sharex=True)
for ax, seed in zip(np.atleast_1d(axes), a.seeds):
    c = build_course(seed); ax.fill_between(XS_RACE, 0, c.H[NROW // 2], color="0.6"); ax.axvline(c.finish_x, color="g")
    for s in c.sections: ax.text((s.x_start + s.x_end) / 2, c.H[NROW // 2].max() + 0.05, f"{s.kind}\n{s.level:g}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.6); ax.set_ylabel(f"seed {seed}", fontsize=8)
plt.xlabel("x (m)"); plt.tight_layout(); out = ROOT / "outputs/analysis/race/courses.png"; plt.savefig(out, dpi=110); print(out)
for seed in a.seeds: print(seed, [(s.kind, s.level) for s in build_course(seed).sections], "finish", round(build_course(seed).finish_x, 1))
