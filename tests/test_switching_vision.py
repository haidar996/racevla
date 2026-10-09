"""Smoke tests for the skill switching, vision and race code (Phase 7 stages 1-4, Phase 8, Phase 10): the race course builder, the hint filter, the rule supervisor's decisions, the long race height field, the packaged terrain classifier and the camera elevation map.
The camera tests need off-screen rendering (EGL / OSMesa); they are skipped where it is not available. Run: pytest -q"""
import json, pathlib
from types import SimpleNamespace
import numpy as np
import pytest

from racevla.envs.race import build_course, XS_RACE, YS_RACE, NCOL_RACE, KINDS, START_OF_SECTIONS, GAP
from racevla.envs.terrain import NROW, ELEV_Z
from racevla.skills.hint_filter import HintFilter
from racevla.skills.rules import RuleSupervisor

ROOT = pathlib.Path(__file__).resolve().parents[1]
GAIT_TABLE = json.load(open(ROOT / "models/running_gait_table.json"))


def test_race_course_is_continuous_and_reproducible():
    for seed in (4000, 4001, 4015, 4022):
        c = build_course(seed); assert c.H.shape == (NROW, NCOL_RACE) and c.H.max() < ELEV_Z and c.H.min() >= 0.0 and len(c.sections) == 4
        assert np.array_equal(c.H, build_course(seed).H)                                                  # same seed, same course
        assert c.sections[0].x_start == START_OF_SECTIONS and c.finish_x > c.sections[-1].x_end
        mid = c.H.shape[0] // 2
        for a, b in zip(c.sections, c.sections[1:]):
            assert abs(b.x_start - a.x_end - GAP) < 1e-9 and abs(b.z_start - a.z_end) < 1e-9 and a.kind in KINDS            # a section starts at the height the previous one ended at, after a flat gap
            assert not (a.kind == "bar" and b.kind == "bar")
        assert all(0.0 <= s.z_start <= 1.25 + 1e-9 and 0.0 <= s.z_end <= 1.25 + 1e-9 for s in c.sections)
        gap_x = (XS_RACE > c.sections[0].x_end + 0.2) & (XS_RACE < c.sections[1].x_start - 0.2)
        assert np.ptp(c.H[mid, gap_x]) < 1e-9                                                              # the gap is flat


def test_hint_filter_ignores_flicker_but_passes_the_bar_at_once():
    f = HintFilter(hold=3); seq = [None, "rough", None, "rough", "rough", "rough", "slope_up", "bar", None, None, None, None]
    assert [f.update(h) for h in seq] == [None, None, None, None, None, "rough", "slope_up", "bar", "bar", "bar", None, None]
    assert [HintFilter(hold=1).update(h) for h in ("rough", None, "bar")] == ["rough", None, "bar"]       # hold = 1: the filter is off


def test_rule_supervisor_decisions():
    rs = RuleSupervisor(sup=SimpleNamespace())
    assert rs.want(0.0, 0.0, "walk") == "stand" and rs.want(0.3, 0.0, "stand") == "walk" and rs.want(1.5, 0.0, "walk") == "run"
    assert rs.want(0.12, 0.0, "stand") == "stand" and rs.want(0.12, 0.0, "walk") == "walk"                  # hysteresis around the stand threshold
    assert rs.want(0.6, 0.0, "run") == "run" and rs.want(0.6, 0.0, "walk") == "walk"                         # ... and around the run threshold
    rs.set_hint("bar"); assert rs.want(1.5, 0.0, "run") == "stepover"
    for hint in ("rough", "slope_up", "slope_down", "stairs_up", "stairs_down"): rs.set_hint(hint); assert rs.want(1.5, 0.0, "run") == "terrain"
    rs.set_hint(None); assert rs.want(1.5, 0.0, "run") == "run"
    assert list(rs.command_for("stand", 1.0, 0.2)) == [0.0, 0.0] and rs.command_for("walk", 2.0, 0.0)[0] == 0.6 and rs.command_for("stepover", 0.5, 0.0)[0] == 1.3


def test_race_body_long_field_and_supervisor_runs():
    from racevla.skills.library import Supervisor
    sup = Supervisor(race=True); body = sup.body; c = build_course(4015); assert body.ncol == NCOL_RACE and body.x1 == 45.0
    rs = RuleSupervisor(sup, hold_heading=True); rs.reset(4015, options={"height_field": (c.H, 0.0, ("race", 0.0)), "yaw": 0.0}, start_x=-0.5); rs.set_target(1.5, 0.0)
    s = c.sections[0]; assert abs(float(body.ground_height_at(s.x_start + 0.5 * (s.x_end - s.x_start), 0.0)) - float(np.interp(s.x_start + 0.5 * (s.x_end - s.x_start), XS_RACE, c.H[NROW // 2]))) < 0.02
    for t in range(150):
        term, info = rs.step(); assert not term
    assert body.data.qpos[0] > 0.5 and sup.active.name in ("walk", "run")                                    # stand -> walk -> run on the 46 m field


def test_terrain_classifier_packaged():
    from racevla.vision.classifier import TerrainClassifier, CLASSES, to_hint
    clf = TerrainClassifier(ROOT / "models/terrain_classifier_cnn_round3.pt", use_gravity=True)
    c = clf.predict(np.full((48, 64), 2.0, np.float32), np.array([0.0, 0.0, -1.0], np.float32)); assert 0 <= c < len(CLASSES)
    assert to_hint(0) is None and to_hint(CLASSES.index("bar")) == "bar"


def test_camera_elevation_map_sees_a_bar():
    from racevla.skills.body import RaceSwitchBody
    from racevla.vision.heightmap import HeightMapper
    body = RaceSwitchBody(GAIT_TABLE)
    try: mapper = HeightMapper(body, 64, 48).attach()
    except Exception as e: pytest.skip(f"no off-screen rendering here: {e}")
    H = np.zeros((NROW, NCOL_RACE)); H[:, (XS_RACE >= 2.0) & (XS_RACE <= 2.08)] = 0.07                       # a 7 cm bar across the field at x = 2 m
    body.reset(seed=0, options={"height_field": (H, 0.0, ("race", 0.0)), "yaw": 0.0}); mapper.reset(); mapper.update(body.data)
    assert mapper.last_depth.shape == (48, 64) and np.isfinite(mapper.last_depth).all()
    assert abs(float(mapper.query(1.0, 0.0))) < 0.02                                                       # flat floor in front of the bar
    assert 0.03 < float(mapper.query(2.02, 0.0)) < 0.11                                                     # the bar itself, from the depth picture only (the robot starts at x = 0, the camera range is 5 m)
