"""Stage 3 test: the rule supervisor (racevla/skills/rules.py) follows random SCHEDULES of targets on flat ground. Usage: python 21_test_supervisor.py [--episodes 100] [--out outputs/analysis/switching_supervisor]
A schedule = a list of segments, each a target class (stand 0 | walk U(0.2,0.6) | run U(1.0,2.5) m/s) different from the previous one, held for SEG_STEPS. Three sets: slow (3 s per segment, 6 segments), fast (1.5 s, 10 segments: the target changes
before the robot has settled = stress test), turns (3 s, 6 segments, walk and run segments also get a turn rate in +-0.4 / +-0.3 rad/s half of the time). Seeds 8000+ (--seed-base). Metrics per episode: fall (and which skill / switch preceded it), number of switches,
and at the END of each segment whether the target was reached (walk/run: speed within 20 %, stand: |vx| < 0.1 m/s and tilt < 10 deg)."""
import sys, pathlib, argparse, csv, collections; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch; torch.set_num_threads(1)
from racevla.skills.rules import RuleSupervisor

SETS = {"slow": dict(seg_steps=150, n_seg=6, turns=False), "fast": dict(seg_steps=75, n_seg=10, turns=False), "turns": dict(seg_steps=150, n_seg=6, turns=True)}


def schedule(rng, n_seg, turns):
    segs, prev = [], None
    for _ in range(n_seg):
        cls = rng.choice([c for c in ("stand", "walk", "run") if c != prev]); prev = cls
        vx = {"stand": 0.0, "walk": rng.uniform(0.2, 0.6), "run": rng.uniform(1.0, 2.5)}[cls]
        wz = (rng.uniform(-0.4, 0.4) if cls == "walk" else rng.uniform(-0.3, 0.3)) if (turns and cls != "stand" and rng.random() < 0.5) else 0.0
        segs.append((cls, float(vx), float(wz)))
    return segs


def episode(rs, seed, cfg):
    rng = np.random.default_rng(seed); segs = schedule(rng, cfg["n_seg"], cfg["turns"]); rs.reset(seed); out = dict(seed=seed, fell=False, segments=len(segs), reached=0, switches=0, fall_segment=-1, fall_skill="", fall_edge="", fall_since_switch=-1)
    errs = []
    for k, (cls, vx, wz) in enumerate(segs):
        rs.set_target(vx, wz)
        for t in range(cfg["seg_steps"]):
            term, info = rs.step()
            if term:
                last = rs.log[-1] if rs.log else (0, "-", "-"); out.update(fell=True, fall_segment=k, fall_skill=rs.sup.active.name, fall_edge=f"{last[1]}->{last[2]}", fall_since_switch=rs.sup.steps - last[0], reason=info["termination_reason"], switches=len(rs.log))
                out["target_class_at_fall"] = cls; return out
            if t >= cfg["seg_steps"] - 25: errs.append(abs(info["vx"] - vx))
        tilt = rs.sup.body.tilt_deg(); last_vx = info["vx"]
        out["reached"] += int((abs(last_vx) < 0.1 and tilt < 10.0) if cls == "stand" else abs(last_vx - vx) <= 0.2 * vx)
    out.update(switches=len(rs.log), mean_err=float(np.mean(errs))); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=100); ap.add_argument("--out", default="outputs/analysis/switching_supervisor"); ap.add_argument("--seed-base", type=int, default=8000); a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); rs = RuleSupervisor(); rows = []; md = ["| set | episodes | falls | reached at segment end | switches per episode | mean speed error (m/s) |", "|---|---|---|---|---|---|"]
    for name, cfg in SETS.items():
        r = [dict(episode(rs, a.seed_base + i, cfg), set=name) for i in range(a.episodes)]; rows += r; falls = sum(x["fell"] for x in r); done = [x for x in r if not x["fell"]]
        seg_total = sum(x["segments"] for x in r); reached = sum(x["reached"] for x in r)          # a fallen episode counts its remaining segments as not reached
        line = f"| {name} | {len(r)} | {falls} | {reached}/{seg_total} ({100 * reached / seg_total:.0f} %) | {np.mean([x['switches'] for x in done]):.1f} | {np.mean([x['mean_err'] for x in done]):.3f} |"
        md.append(line); print(line, flush=True)
        for x in r:
            if x["fell"]: print(f"   fell: seed {x['seed']} segment {x['fall_segment']} (target {x['target_class_at_fall']}) skill {x['fall_skill']} last switch {x['fall_edge']} {x['fall_since_switch']} steps before ({x['reason']})", flush=True)
    keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(open(out / "episodes.csv", "w", newline=""), fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    open(out / "summary.md", "w").write("\n".join(md) + "\n"); print("saved", out)
