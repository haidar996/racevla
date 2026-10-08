"""Phase 4: comparison table of the three gait options (A clock, B sine reference, C reward-only), 3 seeds each.
Reads outputs/analysis/walk_eval/<run>.json (07_eval_walk_model.py) and outputs/<run>/eval_walk.csv. Writes outputs/analysis/walking_comparison.md."""
import sys, json, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]
import numpy as np
OPTIONS = {"A  clock + schedule reward": "walk3_clock_1M_seed{}", "B  sine reference + residual": "walk4_sine_1M_seed{}", "C  reward-only fixes": "walk5_gait_1M_seed{}"}
def first_good(csv):
    """first evaluation (steps) from which speed error <= 0.06, turn error <= 0.10 and falls <= 1 of 16 -- 'learned to walk well'."""
    rows = [r.split(",") for r in open(csv).read().strip().split("\n")[1:]]
    for r in rows:
        if float(r[4]) <= 0.06 and float(r[5]) <= 0.10 and int(r[3]) <= 1: return int(r[0])
    return None
lines = ["| Option | seeds done | falls /50 (per seed) | fwd error m/s | speed reached | turn error rad/s | turn achieved vs 0.40 | feet on floor (mean of 4) | foot slip | lift cm | steps to learn walking well |", "|---|---|---|---|---|---|---|---|---|---|---|"]
details = []
for name, pat in OPTIONS.items():
    R, G, done = [], [], 0
    for s in range(3):
        j = ROOT / "outputs" / "analysis" / "walk_eval" / (pat.format(s) + ".json"); c = ROOT / "outputs" / pat.format(s) / "eval_walk.csv"
        if j.exists(): R.append(json.load(open(j))["random"]); done += 1
        if c.exists(): G.append(first_good(c))
    if not R: lines.append(f"| {name} | 0 | (not run) | | | | | | | | |"); continue
    m = lambda k: np.mean([r[k] for r in R]); lift = [r["lift_cm_p95"] for r in R if r["lift_cm_p95"] is not None]
    gg = [g for g in G if g is not None]
    lines.append(f"| {name} | {done} | {', '.join(str(r['falls']) for r in R)} | {m('fwd_error'):.3f} | {100 * m('speed_ratio'):.0f}% | {m('turn_error'):.3f} | {m('turn_achieved'):.2f} | {np.mean([np.mean(r['on_floor']) for r in R]):.2f} | {m('slip'):.2f} | {np.mean(lift) if lift else float('nan'):.1f} | {', '.join(f'{g // 1000}k' if g else 'never' for g in G)} |")
    details.append((name, [r["on_floor"] for r in R]))
out = "\n".join(lines) + "\n\nTime each foot (FR FL RR RL) is on the floor, per seed (a trot is about 0.6 for all four):\n" + "\n".join(f"- {n}: " + " | ".join(" ".join(f"{x:.2f}" for x in f) for f in fl) for n, fl in details) + "\n"
(ROOT / "outputs" / "analysis" / "walking_comparison.md").write_text(out); print(out)
