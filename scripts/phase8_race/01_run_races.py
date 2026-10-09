"""Phase 8: race the rule supervisor over random multi-section courses (racevla/envs/race.py). Usage: python 01_run_races.py [--races 40] [--sections 4] [--hint vision|oracle] [--model PATH] [--tag _vision]
Every race: seed 4000 + i draws the course (and the target speed, uniform 1.3-1.7 m/s: the step-over skill's range); the robot starts at x = -0.5, standing, heading +x; the supervisor gets the target speed and a terrain hint at every step:
  vision = the terrain CNN (racevla/vision/classifier.py, gravity input, single frame) looks at the newest depth image, and the elevation map comes from the camera; oracle = the hint is read from the course layout (terrain section: from 1.0 m before its start to 0.3 m after its end; bar: 2.0 m before to 1.2 m after), the elevation map comes from the camera too.
A race ENDS when: x >= the finish line (success), the robot falls or leaves the course, it makes less than 0.25 m of new progress in 500 steps (stuck), or MAX_STEPS (6000 = 120 s) pass.
The vision hint can be smoothed (--hold N, racevla/skills/hint_filter.py). Recorded: success, time to finish (s), sections passed (the robot's furthest x is beyond the end of the section + 0.3 m), where and in which skill it ended, steps per skill, switches, and the share of steps where the vision hint agrees with the oracle (flat / terrain / bar)."""
import sys, pathlib, argparse, csv, multiprocessing as mp; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np

SEED0, START_X, MAX_STEPS, STUCK_STEPS, STUCK_PROGRESS = 4000, -0.5, 6000, 500, 0.25
LOOK, AFTER, BAR_LOOK, BAR_AFTER, PASS_MARGIN = 1.0, 0.3, 2.0, 1.2, 0.3


def oracle_hint(course, x):
    for s in course.sections:
        lo, hi = (s.x_start - BAR_LOOK, s.x_end + BAR_AFTER) if s.kind == "bar" else (s.x_start - LOOK, s.x_end + AFTER)
        if lo <= x <= hi: return s.kind
    return None


def run(job):
    seeds, n_sections, hint_mode, model, hold = job
    import torch; torch.set_num_threads(1)
    from collections import deque, Counter
    from racevla.envs.race import build_course
    from racevla.skills.library import Supervisor
    from racevla.skills.rules import RuleSupervisor
    from racevla.vision.heightmap import HeightMapper
    from racevla.vision.classifier import TerrainClassifier, to_hint, CLASSES
    from collections import Counter as _C
    rs = RuleSupervisor(Supervisor(race=True), hold_heading=True); sup, body = rs.sup, rs.sup.body; sup.vision = HeightMapper(body, 64, 48).attach()
    clf = TerrainClassifier(ROOT / model, use_gravity=True) if hint_mode == "vision" else None
    from racevla.skills.hint_filter import HintFilter
    hf = HintFilter(hold)
    group = lambda h: None if h is None else ("bar" if h == "bar" else "terrain")
    rows = []
    for seed in seeds:
        course = build_course(seed, n_sections); cmd = float(np.random.default_rng(seed).uniform(1.3, 1.7))
        rs.reset(seed, options={"height_field": (course.H, 0.0, ("race", 0.0)), "yaw": 0.0}, start_x=START_X); rs.set_target(cmd, 0.0); hf.reset()
        out = dict(seed=seed, cmd=cmd, hint=hint_mode, layout=" ".join(f"{s.kind}:{s.level:g}" for s in course.sections), finish_x=course.finish_x, success=False, fell=False, left=False, stuck=False, timeout=False, reason="", end_skill="", end_x=0.0, time_s=float("nan"), passed=0, end_section="", hint_steps=0, hint_match=0)
        skill_steps = Counter(); best_x, best_t = START_X, 0; diag = _C()          # diag: (truth, raw classifier output, active skill, next section kind, 1 m bin of its start relative to the robot) -> steps
        for t in range(MAX_STEPS):
            x = float(body.data.qpos[0]); o = oracle_hint(course, x)
            if clf:
                cls = clf.predict(sup.vision.last_depth, sup.base[24:27]); nxt = next((q for q in course.sections if x <= q.x_end + AFTER), None); db = int(np.clip(np.floor(nxt.x_start - x), -5, 6)) if nxt else 99
                diag[f"{o or 'flat'}|{CLASSES[cls]}|{sup.active.name}|{nxt.kind if nxt else 'none'}|{db}"] += 1
                rs.set_hint(hf.update(to_hint(cls))); out["hint_steps"] += 1; out["hint_match"] += int(group(rs.hint) == group(o))
            else: rs.set_hint(o)
            skill_steps[sup.active.name] += 1; term, info = rs.step(); x = float(body.data.qpos[0])
            if x > best_x + STUCK_PROGRESS: best_x, best_t = x, t
            if term: out.update(fell=True, left=info["termination_reason"] == "left_course", reason=info["termination_reason"]); break
            if x >= course.finish_x: out.update(success=True, time_s=(t + 1) * 0.02); break
            if t - best_t >= STUCK_STEPS: out.update(stuck=True, reason="no_progress"); break
        else: out.update(timeout=True, reason="timeout")
        x = float(body.data.qpos[0]); out.update(end_x=x, end_skill=sup.active.name, passed=sum(x > s.x_end + PASS_MARGIN for s in course.sections), n_switches=len(rs.log), edges=" ".join(f"{a[:3]}>{b[:3]}@{s}" for s, a, b in rs.log))
        if not out["success"]:
            nxt = [s for s in course.sections if x <= s.x_end + PASS_MARGIN]; out["end_section"] = nxt[0].kind if nxt else "after_last"
        out["skill_steps"] = " ".join(f"{k}:{v}" for k, v in sorted(skill_steps.items())); out["diag"] = dict(diag); rows.append(out)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--races", type=int, default=40); ap.add_argument("--sections", type=int, default=4); ap.add_argument("--hint", default="vision", choices=["vision", "oracle"])
    ap.add_argument("--model", default="models/terrain_classifier_cnn_round3.pt"); ap.add_argument("--tag", default=""); ap.add_argument("--hold", type=int, default=1); ap.add_argument("--seed0", type=int, default=SEED0); ap.add_argument("--procs", type=int, default=4); a = ap.parse_args()
    seeds = [a.seed0 + i for i in range(a.races)]; jobs = [(seeds[i::a.procs], a.sections, a.hint, a.model, a.hold) for i in range(a.procs)]
    with mp.Pool(a.procs, maxtasksperchild=1) as pool: res = pool.map(run, jobs, chunksize=1)
    rows = sorted([r for rr in res for r in rr], key=lambda r: r["seed"]); out = ROOT / "outputs/analysis/race"; out.mkdir(parents=True, exist_ok=True)
    import json; json.dump({r["seed"]: dict(success=r["success"], counts=r["diag"]) for r in rows}, open(out / f"diag{a.tag}.json", "w")); [r.pop("diag") for r in rows]
    keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(open(out / f"races{a.tag}.csv", "w", newline=""), fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    n = len(rows); ok = [r for r in rows if r["success"]]; cause = lambda r: "finished" if r["success"] else "fell" if r["fell"] and not r["left"] else "left course" if r["left"] else "stuck" if r["stuck"] else "timeout"
    md = [f"Race results: {n} courses of {a.sections} sections, hint = {a.hint}" + (f" (smoothed: a new group must be asked for {a.hold} steps in a row; the bar enters at once)" if a.hold > 1 else "") + ".", "", f"- finished: **{100 * len(ok) / n:.0f} %** ({len(ok)} / {n})", f"- mean time of the finishers: {np.mean([r['time_s'] for r in ok]) if ok else float('nan'):.1f} s (course lengths {min(r['finish_x'] for r in rows):.0f}-{max(r['finish_x'] for r in rows):.0f} m)",
          f"- sections passed on average: {np.mean([r['passed'] for r in rows]):.2f} of {a.sections}", f"- endings: " + ", ".join(f"{c} {sum(cause(r) == c for r in rows)}" for c in ("finished", "fell", "left course", "stuck", "timeout"))]
    if a.hint == "vision": md.append(f"- hint agrees with the oracle (flat / terrain / bar) on {100 * sum(r['hint_match'] for r in rows) / max(1, sum(r['hint_steps'] for r in rows)):.0f} % of steps")
    md += ["", "Where the unfinished races ended (the section the robot was on or heading into):", "", "| section | races ended there |", "|---|---|"]
    for k in ("rough", "slope_up", "slope_down", "stairs_up", "stairs_down", "bar", "after_last"): md.append(f"| {k} | {sum(r['end_section'] == k for r in rows)} |")
    md += ["", "Sections passed (number of races): " + ", ".join(f"{k}: {sum(r['passed'] == k for r in rows)}" for k in range(a.sections + 1))]
    open(out / f"summary{a.tag}.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
    for r in rows:
        if not r["success"]: print(f"seed {r['seed']} {cause(r)} ({r['reason']}) skill {r['end_skill']} x {r['end_x']:.1f} / {r['finish_x']:.1f}  layout {r['layout']}  passed {r['passed']}  switches {r['edges'][-80:]}")
