"""Stage 1 of skill switching: measure every hand-over on flat ground. Usage: python 20_test_transitions.py [--episodes 50] [--out outputs/analysis/switching] [--set direct|brake|terrain]
One episode = skill A runs until it is in steady state, then (after a random 0-19 step delay = a random point of the gait cycle) control goes to skill B, which runs for 6 s. Commands per skill: stand (0, 0), walk vx ~ U(0.3, 0.6),
run vx ~ U(1.5, 2.5), no turning. Controls (A = B, same command, nothing really changes) show the fall rate that exists WITHOUT a hand-over. Falls during A are counted separately and left out of the hand-over statistics.
Per episode: fell after the switch, step of the fall, settling time (first moment B's behaviour holds for 0.5 s: walk/run speed within 20 % of the command, stand |vx| < 0.1 m/s and tilt < 10 deg), peak tilt in the first second,
speed at the switch, gait phase at the switch. Seeds 7000+ (fresh, never used before)."""
import sys, pathlib, argparse, csv; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import torch; torch.set_num_threads(1)
from racevla.skills.library import Supervisor

STEADY_STEPS = {"stand": 150, "walk": 200, "run": 300, "terrain": 200, "stepover": 300}      # how long A runs before the switch (3 s / 4 s / 6 s: the run policy needs time to accelerate)
B_STEPS, HOLD = 300, 25                                      # B runs 6 s; "settled" = criterion true for 25 steps (0.5 s) in a row
CONDITIONS = [("stand", "walk", None), ("walk", "stand", 0), ("walk", "stand", 10), ("walk", "run", None), ("run", "walk", None), ("run", "stand", 0), ("run", "stand", 10),
              ("stand", "stand", 0), ("walk", "walk", None), ("run", "run", None)]       # (A, B, warm-up steps used when taking over as stand)


# --set brake: run -> stand through walk (Supervisor.go): (A, B, warm-up of the final stand, (brake command, brake speed)); the direct run -> stand rows are repeated for comparison
BRAKE_CONDITIONS = [("run", "stand", 0, None), ("run", "stand", 0, (0.0, 0.6)), ("run", "stand", 0, (0.4, 0.6)), ("run", "stand", 10, (0.0, 0.6)), ("run", "stand", 10, (0.4, 0.6)), ("run", "stand", 0, (0.0, 0.3))]


# --set terrain (stage 4b): the terrain and step-over skills on the infinite flat floor (Supervisor(flat_skills=True)); commands: terrain vx ~ U(0.4, 0.8) (its training range), step-over vx ~ U(1.3, 1.7); both skills hold the heading themselves
TERRAIN_CONDITIONS = [("walk", "terrain", 0, None), ("terrain", "walk", 0, None), ("stand", "terrain", 0, None), ("terrain", "stand", 0, None), ("run", "terrain", 0, None), ("terrain", "run", 0, None),
                      ("run", "stepover", 0, None), ("stepover", "run", 0, None), ("walk", "stepover", 0, None), ("stepover", "walk", 0, None), ("stepover", "stand", 0, None), ("terrain", "terrain", 0, None), ("stepover", "stepover", 0, None)]


def draw_command(rng, skill): return {"stand": (0.0, 0.0), "walk": (rng.uniform(0.3, 0.6), 0.0), "run": (rng.uniform(1.5, 2.5), 0.0), "terrain": (rng.uniform(0.4, 0.8), 0.0), "stepover": (rng.uniform(1.3, 1.7), 0.0)}[skill]


def settled(skill, cmd, info, tilt):
    if skill == "stand": return abs(info["vx"]) < 0.1 and tilt < 10.0
    return abs(info["vx"] - cmd[0]) <= 0.2 * cmd[0]


def episode(sup, A, B, warm, seed, via=None):
    rng = np.random.default_rng(seed); cA = draw_command(rng, A); cB = cA if A == B else draw_command(rng, B); delay = int(rng.integers(0, 20))
    sup.reset(seed, A, cA, warmup=10 if A == "stand" else None); out = dict(A=A, B=B, warmup=warm, via="" if via is None else f"walk cmd {via[0]} until <{via[1]} m/s", seed=seed, cmd_A=cA[0], cmd_B=cB[0], fell_in_A=False, fell=False)
    for t in range(STEADY_STEPS[A] + delay):
        term, info = sup.step()
        if term: out["fell_in_A"] = True; return out
    body = sup.body; out.update(speed_at_switch=info["vx"], phase=float(body.phase), tilt_at_switch=body.tilt_deg())
    if via is None: sup.switch(B, cB, warmup=warm if B == "stand" else None)
    else: sup.go(B, cB, brake_cmd=via[0], brake_speed=via[1], warmup=warm)
    s0 = sup.steps; run = 0; first = None; peak = 0.0; vx = []
    for t in range(B_STEPS):
        term, info = sup.step(); tilt = body.tilt_deg(); vx.append(info["vx"])
        if t < 50: peak = max(peak, tilt)
        run = run + 1 if settled(B, cB, info, tilt) else 0
        if run == HOLD and first is None: first = t - HOLD + 1
        if term: out.update(fell=True, fall_step=t, reason=info["termination_reason"]); break
    out.update(brake_s=None if (via is None or sup.handover_step is None) else (sup.handover_step - s0) * 0.02, peak_tilt_1s=peak, settle_s=None if first is None else first * 0.02, final_vx=float(np.mean(vx[-25:])))
    return out


def summarize(rows, A, B, warm, via=None):
    tag = "" if via is None else f"walk cmd {via[0]} until <{via[1]} m/s"
    r = [x for x in rows if x["A"] == A and x["B"] == B and x["warmup"] == warm and x["via"] == tag]; ok = [x for x in r if not x["fell_in_A"]]
    falls = sum(x["fell"] for x in ok); st = [x["settle_s"] for x in ok if not x["fell"] and x["settle_s"] is not None]; never = sum((not x["fell"]) and x["settle_s"] is None for x in ok)
    return dict(pair=f"{A} -> {B}" + (f" (warm-up {warm})" if B == "stand" else "") + (f" via {tag}" if tag else ""), n=len(ok), fell_in_A=len(r) - len(ok), falls=falls, never=never,
                med=np.median(st) if st else float("nan"), p90=np.percentile(st, 90) if st else float("nan"), tilt=np.mean([x["peak_tilt_1s"] for x in ok]) if ok else float("nan"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=50); ap.add_argument("--out", default=None); ap.add_argument("--set", default="direct", choices=["direct", "brake", "terrain"]); a = ap.parse_args()
    conds = {"direct": [(A, B, w, None) for A, B, w in CONDITIONS], "brake": BRAKE_CONDITIONS, "terrain": TERRAIN_CONDITIONS}[a.set]
    if a.set == "brake": B_STEPS = 400                       # the braking phase comes on top of the 6 s
    out = ROOT / (a.out or {"direct": "outputs/analysis/switching", "brake": "outputs/analysis/switching_brake", "terrain": "outputs/analysis/switching_terrain"}[a.set]); out.mkdir(parents=True, exist_ok=True); sup = Supervisor(flat_skills=(a.set == "terrain")); rows = []
    for A, B, warm, via in conds:
        for i in range(a.episodes): rows.append(episode(sup, A, B, warm, 7000 + i, via))
        s = summarize(rows, A, B, warm, via); print(f"{s['pair']:58s} n={s['n']:3d} falls={s['falls']:3d} never_settled={s['never']:3d} settle median {s['med']:.2f}s p90 {s['p90']:.2f}s peak tilt {s['tilt']:.1f} deg  (fell during A: {s['fell_in_A']})", flush=True)
    keys = sorted({k for r in rows for k in r}); w = csv.DictWriter(open(out / "transitions.csv", "w", newline=""), fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    with open(out / "transitions.md", "w") as f:
        f.write("| hand-over | episodes | falls after switch | never settled | settle median (s) | settle p90 (s) | peak tilt in 1 s (deg) | fell before switch |\n|---|---|---|---|---|---|---|---|\n")
        for A, B, warm, via in conds:
            s = summarize(rows, A, B, warm, via); f.write(f"| {s['pair']} | {s['n']} | {s['falls']} | {s['never']} | {s['med']:.2f} | {s['p90']:.2f} | {s['tilt']:.1f} | {s['fell_in_A']} |\n")
    print("saved", out)
