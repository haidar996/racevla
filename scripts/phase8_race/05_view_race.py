"""Viewer for ONE race (Phase 8): the rule supervisor drives the Go1 over a random multi-section course with the camera pipeline (depth camera -> elevation map, classifier -> terrain hint). The window shows the course ahead, the active skill, the classifier's hint, the TRUE section under / ahead of the robot, speed and the outcome (FINISHED / FELL ...). After the end it waits 3 s and restarts the same race (loops until the window is closed).
Usage: python 05_view_race.py [--seed 4015] [--hint vision|oracle] [--slow 1] [--no-window]   (seed 4015 finished with the vision hint: slope up 9 deg, rough 6 cm, bar 5 cm, stairs down 6 cm). --no-window runs it without a window and prints the outcome. Needs a display for the window."""
import os, sys, time, pathlib; os.environ.setdefault("MUJOCO_GL", "egl"); ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))      # egl = the depth camera renders off-screen; the viewer window has its own OpenGL
import numpy as np
import torch; torch.set_num_threads(2)
import mujoco, mujoco.viewer
from racevla.envs.race import build_course
from racevla.skills.library import Supervisor
from racevla.skills.rules import RuleSupervisor
from racevla.vision.heightmap import HeightMapper
from racevla.vision.classifier import TerrainClassifier, to_hint, CLASSES

arg = lambda n, d: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
seed, hint_mode, slow = int(arg("--seed", 4015)), arg("--hint", "vision"), float(arg("--slow", 1.0))
sys.path.insert(0, str(ROOT / "scripts/phase8_race")); import importlib; races = importlib.import_module("01_run_races")
course = build_course(seed); cmd = float(np.random.default_rng(seed).uniform(1.3, 1.7))
rs = RuleSupervisor(Supervisor(race=True), hold_heading=True); sup, body = rs.sup, rs.sup.body; sup.vision = HeightMapper(body, 64, 48).attach()
clf = TerrainClassifier(ROOT / "models/terrain_classifier_cnn_round3.pt", use_gravity=True) if hint_mode == "vision" else None
layout = "  >  ".join(f"{s.kind} {s.level:g}" for s in course.sections)


def setup(viewer):
    def reset():
        rs.reset(seed, options={"height_field": (course.H, 0.0, ("race", 0.0)), "yaw": 0.0}, start_x=races.START_X); rs.set_target(cmd, 0.0)
    if viewer is None: reset()
    else:
        with viewer.lock(): reset()
        viewer.update_hfield(body.hf_id)                    # outside the lock (it takes the lock itself)


def run_race(viewer):
    setup(viewer); status = "running"; best_x, best_t = races.START_X, 0
    for t in range(races.MAX_STEPS):
        if viewer is not None and not viewer.is_running(): return None
        t0 = time.time(); x = float(body.data.qpos[0]); o = races.oracle_hint(course, x)
        says = None
        if clf: says = to_hint(clf.predict(sup.vision.last_depth, sup.base[24:27])); rs.set_hint(says)
        else: rs.set_hint(o)
        term, info = rs.step(); x = float(body.data.qpos[0])
        sec = next((s for s in course.sections if x <= s.x_end + races.PASS_MARGIN), None); under = f"{sec.kind} {sec.level:g} ({sec.x_start - x:+.1f} m)" if sec else "flat run-out"
        if x > best_x + races.STUCK_PROGRESS: best_x, best_t = x, t
        if term: status = "FELL" if info.get("termination_reason") != "left_course" else "LEFT THE COURSE"
        elif x >= course.finish_x: status = f"FINISHED in {(t + 1) * 0.02:.1f} s"
        elif t - best_t >= races.STUCK_STEPS: status = "STUCK (no progress for 10 s)"
        if viewer is not None:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING; viewer.cam.trackbodyid = 1; viewer.cam.distance = 4.0; viewer.cam.elevation = -12; viewer.cam.azimuth = 90
            hint_txt = f"classifier says: {says or 'flat'}" if clf else f"oracle hint: {o or 'none'}"
            viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, f"race seed {seed}: {layout}",
                              f"x {x:5.1f} / {course.finish_x:.1f} m   speed {info['vx']:.2f} m/s (target {cmd:.2f})   skill: {sup.active.name.upper()}   {hint_txt}\nnext / current section: {under}\n{status}"))
            viewer.sync(); time.sleep(max(0.0, 0.02 * slow - (time.time() - t0)))
        if status != "running":
            if viewer is not None: time.sleep(3.0)
            return status
    return "timeout"


if "--no-window" in sys.argv: print(f"seed {seed} ({layout}), hint {hint_mode}: {run_race(None)}", flush=True)
else:
    setup(None)
    with mujoco.viewer.launch_passive(body.model, body.data) as v:
        while v.is_running():
            res = run_race(v); print(f"seed {seed}: {res}", flush=True)
            if res is None: break
