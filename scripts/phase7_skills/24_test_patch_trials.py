"""Stage 4c: one-patch trials on the 10 m terrain course with the rule supervisor and an ORACLE terrain hint. Usage: python 24_test_patch_trials.py [--episodes 20] [--out outputs/analysis/switching_patches]
Every trial: the robot starts at x = -0.5 (heading +x, standing; the height field ends at x = -1), one patch of one terrain kind and level follows (rough x 1..7 m, slopes 1..5, stairs 1..3, a thin bar at x = 5.0), the course ends at x = 9.
Terrain patches (target 0.4-0.6 m/s): the oracle turns the hint on when the patch is within LOOK m ahead and off 0.3 m after its end; success = x >= 8.0 without falling / leaving the course (|y| > 1.4) within 1300 steps.
Bar (target 1.3-1.7 m/s, run approach): hint 'bar' from 2.0 m before the bar to 1.2 m after it; success = the rule of Go1HurdleEnv (2 s after x > 5.45 m upright at >= 50 % of the command, no trunk / hip / thigh touching the bar).
Policies compared in the SAME trials: 'supervisor' (rules + hint, switches skills), and fixed single skills: 'walk' / 'terrain' (terrain patches) or 'run' / 'stepover' (bar), all with the heading hold they were trained with.
Seeds: terrain seed = episode seed = 3000 + i. Per episode the supervisor's switches are recorded; a fall records the skill and the last switch before it."""
import sys, pathlib, argparse, csv, multiprocessing as mp; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np

TERRAIN_CELLS = [("rough", 0.06), ("rough", 0.10), ("slope_up", 9), ("slope_up", 15), ("slope_down", 15), ("stairs_up", 0.04), ("stairs_down", 0.06)]
BAR_CELLS = [("hurdle", 0.03), ("hurdle", 0.05), ("hurdle", 0.07), ("hurdle", 0.10)]
EXTENT = {"rough": (1.0, 7.0), "slope_up": (1.0, 5.0), "slope_down": (1.0, 5.0), "stairs_up": (1.0, 3.0), "stairs_down": (1.0, 3.0)}
SEED0 = 3000
LOOK, AFTER, START_X, GOAL_X, MAX_STEPS = 1.0, 0.3, -0.5, 8.0, 1300
BAR_X, BAR_LOOK, BAR_AFTER, CROSSED_X, X_EDGE, BAR_STEPS = 5.0, 2.0, 1.2, 5.45, 8.95, 700
POLICIES = {"terrain": ["supervisor", "walk", "terrain"], "bar": ["supervisor", "run", "stepover"]}


def run(job):
    policy, kind, level, n, *rest = job; vision = rest[0] if rest else None            # vision = None (the simulator's height map) or (width, height) of the depth camera whose elevation map replaces it
    clf_spec = rest[1] if len(rest) > 1 else None                                      # clf_spec = None (oracle hint) or (model path, vote[, use_gravity]): the hint comes from the terrain CNN looking at the latest depth image (majority of the last `vote` predictions)
    import torch; torch.set_num_threads(1)
    from racevla.skills.library import Supervisor
    from racevla.skills.rules import RuleSupervisor
    import mujoco
    rs = RuleSupervisor(Supervisor(terrain=True), hold_heading=True); sup, body = rs.sup, rs.sup.body
    if vision:
        from racevla.vision.heightmap import HeightMapper
        sup.vision = HeightMapper(body, *vision).attach()
    if clf_spec:
        from collections import deque, Counter
        from racevla.vision.classifier import TerrainClassifier, to_hint
        grav = len(clf_spec) > 2 and clf_spec[2]; clf = TerrainClassifier(ROOT / clf_spec[0], use_gravity=grav); votes = deque(maxlen=clf_spec[1])
    rows = []; bar = kind == "hurdle"
    for i in range(n):
        seed = SEED0 + i; rng = np.random.default_rng(seed); cmd = float(rng.uniform(1.3, 1.7) if bar else rng.uniform(0.4, 0.6)); opts = {"terrain": (kind, level), "terrain_seed": seed, "yaw": 0.0}
        if policy == "supervisor":
            rs.reset(seed, options=opts, start_x=START_X); rs.set_target(cmd, 0.0)
            if clf_spec: votes.clear(); votes.append(0)
        else:
            sup.pending = None; sup.steps = 0; sup.reset(seed, policy, [cmd, 0.0], warmup=0, options=opts); body.data.qpos[0] = START_X; mujoco.mj_forward(body.model, body.data)
        out = dict(policy=policy, kind=kind, level=level, seed=seed, cmd=cmd, success=False, fell=False, left=False, stuck=False, reason="", edges="", fall_skill="", fall_x=0.0, final_x=0.0, hint_steps=0, hint_match=0); crossed = None
        for t in range(BAR_STEPS if bar else MAX_STEPS):
            x = float(body.data.qpos[0])
            if policy == "supervisor":
                lo, hi = (BAR_X - BAR_LOOK, BAR_X + BAR_AFTER) if bar else (EXTENT[kind][0] - LOOK, EXTENT[kind][1] + AFTER)
                if clf_spec:
                    votes.append(clf.predict(sup.vision.last_depth, sup.base[24:27])); h = to_hint(Counter(votes).most_common(1)[0][0]); rs.set_hint(h)          # the hint is what the network sees in the newest picture
                    g = lambda z: None if z is None else ("bar" if z == "bar" else "terrain"); o = ("bar" if bar else kind) if lo <= x <= hi else None; out["hint_steps"] += 1; out["hint_match"] += int(g(h) == g(o))      # agreement with the oracle hint (flat / terrain patch / bar)
                else: rs.set_hint(("bar" if bar else kind) if lo <= x <= hi else None)
                term, info = rs.step()
            else:
                if policy in ("walk", "run"): body.command[1] = body.heading_hold()          # the fixed walk / run skills get the same heading hold (terrain and step-over do it themselves)
                term, info = sup.step()
            x = float(body.data.qpos[0])
            if term: out.update(fell=True, left=info["termination_reason"] == "left_course", reason=info["termination_reason"], fall_skill=sup.active.name, fall_x=x); break
            if bar:
                if body.hit_hurdle(): out.update(fell=True, reason="hit_hurdle", fall_skill=sup.active.name, fall_x=x); break
                if crossed is None and x > CROSSED_X: crossed = sup.steps
                if crossed is not None and (sup.steps - crossed >= 100 or x >= X_EDGE):
                    R = body.data.xmat[1].reshape(3, 3); v = float((R.T @ body.data.qvel[:3])[0]); out["success"] = body.tilt_deg() < 35.0 and v >= 0.5 * cmd; break
            elif x >= GOAL_X: out["success"] = True; break
        else: out["stuck"] = True
        out["final_x"] = float(body.data.qpos[0]); out["final_skill"] = sup.active.name
        if policy == "supervisor": out["edges"] = " ".join(f"{a[:3]}>{b[:3]}@{s}" for s, a, b in rs.log); out["n_switches"] = len(rs.log)
        rows.append(out)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=20); ap.add_argument("--out", default="outputs/analysis/switching_patches"); a = ap.parse_args(); n = a.episodes
    jobs = [(p, k, l, n) for cells, key in ((TERRAIN_CELLS, "terrain"), (BAR_CELLS, "bar")) for k, l in cells for p in POLICIES[key]]
    with mp.Pool(4, maxtasksperchild=1) as pool: res = pool.map(run, jobs, chunksize=1)
    rows = [r for rr in res for r in rr]; out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(open(out / "trials.csv", "w", newline=""), fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    md = ["| patch | supervisor | walk only | terrain only |", "|---|---|---|---|"]
    def cell(p, k, l): r = [x for x in rows if x["policy"] == p and x["kind"] == k and x["level"] == l]; s = sum(x["success"] for x in r); f = sum(x["fell"] for x in r); return f"{100 * s / len(r):.0f} % (falls {f})"
    for k, l in TERRAIN_CELLS: md.append(f"| {k} {l} | {cell('supervisor', k, l)} | {cell('walk', k, l)} | {cell('terrain', k, l)} |")
    md += ["", "| bar | supervisor | run only | step-over only |", "|---|---|---|---|"]
    for k, l in BAR_CELLS: md.append(f"| {l * 100:g} cm | {cell('supervisor', k, l)} | {cell('run', k, l)} | {cell('stepover', k, l)} |")
    open(out / "summary.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
    for r in rows:
        if r["policy"] == "supervisor" and not r["success"]: print(f"supervisor failed: {r['kind']} {r['level']} seed {r['seed']} ({'fell' if r['fell'] else 'stuck'} {r['reason']}) skill {r['final_skill']} x {r['final_x']:.2f} switches {r['edges']}")
