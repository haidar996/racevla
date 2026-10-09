"""Record one race as a GIF for the README (headless, EGL): side-on tracking camera with a caption (skill, speed, classifier hint, next section). Same pipeline as 05_view_race.py (depth camera -> elevation map, terrain classifier -> hint). Usage: python 06_record_race.py [--seed 4015] [--hint vision|oracle] [--speed 1.5] [--every 7]  ->  docs/media/race.gif"""
import os, sys, pathlib, importlib; os.environ.setdefault("MUJOCO_GL", "egl"); ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts/phase8_race"))
import numpy as np, torch; torch.set_num_threads(2)
import make_demos as md
from racevla.envs.race import build_course
from racevla.skills.library import Supervisor
from racevla.skills.rules import RuleSupervisor
from racevla.vision.heightmap import HeightMapper
from racevla.vision.classifier import TerrainClassifier, to_hint
races = importlib.import_module("01_run_races")

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
EVERY = int(arg("--every", 7))
seed, hint_mode, speed = int(arg("--seed", 4015)), arg("--hint", "vision"), float(arg("--speed", 1.5))
course = build_course(seed); cmd = float(np.random.default_rng(seed).uniform(1.3, 1.7))
rs = RuleSupervisor(Supervisor(race=True), hold_heading=True); sup, body = rs.sup, rs.sup.body; sup.vision = HeightMapper(body, 64, 48).attach()
clf = TerrainClassifier(ROOT / "models/terrain_classifier_cnn_round3.pt", use_gravity=True) if hint_mode == "vision" else None
rec = md.Recorder(body, dist=3.2, elev=-14, azim=90)
rs.reset(seed, options={"height_field": (course.H, 0.0, ("race", 0.0)), "yaw": 0.0}, start_x=races.START_X); rs.set_target(cmd, 0.0); rec.terrain_changed()
layout = " > ".join(f"{s.kind.replace('_', ' ')} {s.level:g}" for s in course.sections); status = "timeout"
for t in range(races.MAX_STEPS):
    x = float(body.data.qpos[0]); o = races.oracle_hint(course, x)
    says = to_hint(clf.predict(sup.vision.last_depth, sup.base[24:27])) if clf else o; rs.set_hint(says)
    term, info = rs.step(); x = float(body.data.qpos[0]); sec = next((s for s in course.sections if x <= s.x_end + races.PASS_MARGIN), None)
    rec.grab(f"Race (seed {seed}, {hint_mode} hint): {layout}", f"x {x:4.1f}/{course.finish_x:.0f} m  {info['vx']:.1f} m/s  skill {sup.active.name}  hint {says or 'flat'}  next: {sec.kind.replace('_', ' ') + f' in {sec.x_start - x:+.1f} m' if sec else 'finish'}", every=EVERY)
    if term or x >= course.finish_x: status = "finished" if x >= course.finish_x and not term else "fell"; break
print(seed, status, f"{(t + 1) * 0.02:.1f} s", flush=True); assert status == "finished"
rec.save("race", speed=speed)
