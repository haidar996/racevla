"""Phase 8 diagnosis, part 2: drive the REAL supervisor (target 1.5 m/s: walk, then run) over a perfectly flat floor and log what the classifier says at every step, by the robot's x. Compares the 10 m training course (flat = ('rough', 0.0)), the 46 m race course at ground height 0 and the race course raised by 0.4 m.
If the classifier only works near the end of the 10 m field (where the field edge is visible), the 'says flat' share will jump there and be low everywhere on the race course. Part C (default) also logs the sideways position y: --part AB = the x comparison of the three courses, --part C = classifier answers by |y|. Usage: python 04_flat_drive_probe.py [--seeds 3] [--part AB|C]"""
import os, sys, json, argparse, pathlib; os.environ.setdefault("MUJOCO_GL", "egl"); ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
from collections import Counter
from racevla.skills.library import Supervisor
from racevla.skills.rules import RuleSupervisor
from racevla.vision.heightmap import HeightMapper
from racevla.vision.classifier import TerrainClassifier, CLASSES
from racevla.envs.terrain import NROW

ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--part", default="C", choices=["AB", "C"]); a = ap.parse_args()
clf = TerrainClassifier(ROOT / "models/terrain_classifier_cnn_round3.pt", use_gravity=True)


def drive(race, raised, x_end, seed):
    rs = RuleSupervisor(Supervisor(race=race, terrain=not race), hold_heading=True); sup, body = rs.sup, rs.sup.body; sup.vision = HeightMapper(body, 64, 48).attach(); log = []
    opts = {"height_field": (np.full((NROW, body.ncol), raised), raised, ("race", 0.0)), "yaw": 0.0} if race else {"terrain": ("rough", 0.0), "terrain_seed": seed, "yaw": 0.0}
    rs.reset(seed, options=opts, start_x=-0.5); rs.set_target(1.5, 0.0)
    for t in range(3000):
        x = float(body.data.qpos[0]); cls = clf.predict(sup.vision.last_depth, sup.base[24:27]); log.append((x, cls, sup.active.name, float(body.data.qpos[1]))); rs.set_hint(None); term, _ = rs.step()
        if term or x >= x_end: break
    return log


def report(title, logs, step):
    print(title); print("  x bin      steps  says flat   most common wrong answers    skills")
    for lo in np.arange(-1.0, max(l[0] for lg in logs for l in lg), step):
        sel = [l for lg in logs for l in lg if lo <= l[0] < lo + step]
        if len(sel) < 20: continue
        c = Counter(CLASSES[s[1]] for s in sel); sk = Counter(s[2] for s in sel); wrong = ", ".join(f"{k} {100 * v / len(sel):.0f}%" for k, v in c.most_common(4) if k != "flat")
        print(f"  {lo:5.1f}-{lo + step:4.1f} {len(sel):6d}   {100 * c['flat'] / len(sel):5.0f} %    {wrong:40s} {dict(sk)}")


if a.part == "AB": report("A: 10 m training course, flat, driving", [drive(False, 0.0, 8.8, s) for s in range(a.seeds)], 1.0)
if a.part == "AB": report("B1: 46 m race course, flat at height 0, driving", [drive(True, 0.0, 40.0, s) for s in range(a.seeds)], 4.0)
if a.part == "AB": report("B2: 46 m race course, flat raised 0.4 m, driving", [drive(True, 0.4, 40.0, s) for s in range(a.seeds)], 4.0)

if a.part == "C":
  print("C: does the sideways position explain it? B1 flat race course, all steps with the robot running (x > 1), by |y| (the field's edge is at |y| = 1.5):")
  logs = [drive(True, 0.0, 40.0, s) for s in range(a.seeds)]
  for lo, hi in ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.4)):
      sel = [l for lg in logs for l in lg if l[0] > 1.0 and lo <= abs(l[3]) < hi]
      if len(sel) >= 20: c = Counter(CLASSES[l[1]] for l in sel); print(f"  |y| {lo:.1f}-{hi:.1f}: {len(sel):5d} steps, says flat {100 * c['flat'] / len(sel):3.0f} %, " + ", ".join(f"{k} {100 * v / len(sel):.0f}%" for k, v in c.most_common(3)))
  print("  mean |y| by x:", " ".join(f"{lo}:{np.mean([abs(l[3]) for lg in logs for l in lg if lo <= l[0] < lo + 8]):.2f}" for lo in range(0, 40, 8)))
