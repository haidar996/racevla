"""Phase 8: where does the classifier go wrong DURING a race? Reads outputs/analysis/race/diag<tag>.json (01_run_races.py --hint vision) and writes diagnose<tag>.md and a confusion figure.
Every logged step = (truth = the oracle hint's class, raw classifier output, active skill, kind of the next / current section, 1 m bin of that section's start relative to the robot: positive = ahead, negative = already past the start). Usage: python 02_diagnose_hints.py [--tag _diag]"""
import sys, pathlib, argparse, json; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from collections import Counter, defaultdict
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from racevla.vision.classifier import CLASSES

ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="_diag"); a = ap.parse_args()
D = json.load(open(ROOT / f"outputs/analysis/race/diag{a.tag}.json")); rows = []
for seed, d in D.items():
    for k, n in d["counts"].items(): t, p, sk, nk, db = k.split("|"); rows.append((int(seed), d["success"], t, p, sk, nk, int(db), n))
g = lambda c: "flat" if c == "flat" else "bar" if c == "bar" else "terrain"
total = sum(r[-1] for r in rows); md = [f"Classifier output during {len(D)} races ({total} steps), no smoothing.", ""]
# 1. confusion, 7 classes
M = np.zeros((7, 7)); [M.__setitem__((CLASSES.index(r[2]), CLASSES.index(r[3])), M[CLASSES.index(r[2]), CLASSES.index(r[3])] + r[-1]) for r in rows]
acc7 = np.trace(M) / M.sum(); acc3 = sum(r[-1] for r in rows if g(r[2]) == g(r[3])) / total
md += [f"Accuracy over all race steps: 7 classes {100 * acc7:.0f} %, 3 groups (flat / terrain / bar) {100 * acc3:.0f} %. (Single patches on a flat floor, held-out frames: 73 % / 81 %.)", "", "Truth (rows, with the share of all steps) -> what the classifier said (% of that row):", "",
       "| truth | steps % | " + " | ".join(CLASSES) + " |", "|---|---|" + "---|" * 7]
for i, c in enumerate(CLASSES): md.append(f"| {c} | {100 * M[i].sum() / total:.0f} | " + " | ".join(f"{100 * M[i, j] / max(M[i].sum(), 1):.0f}" for j in range(7)) + " |")
fig, ax = plt.subplots(figsize=(6.5, 5.5)); R = M / np.maximum(M.sum(1, keepdims=True), 1); im = ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
for i in range(7):
    for j in range(7): ax.text(j, i, f"{100 * R[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if R[i, j] > 0.5 else "black")
ax.set_xticks(range(7)); ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticks(range(7)); ax.set_yticklabels(CLASSES); ax.set_xlabel("classifier says"); ax.set_ylabel("truth (oracle)"); ax.set_title("Race steps: % of each truth row"); plt.tight_layout(); plt.savefig(ROOT / f"outputs/analysis/race/diagnose_confusion{a.tag}.png", dpi=120)
def table(title, keyf, keys, filt=lambda r: True, label="group accuracy"):
    acc = defaultdict(lambda: [0, 0]); [acc[keyf(r)].__setitem__(slice(None), [acc[keyf(r)][0] + r[-1] * (g(r[2]) == g(r[3])), acc[keyf(r)][1] + r[-1]]) for r in rows if filt(r)]
    return ["", title, "", "| | steps | " + label + " % |", "|---|---|---|"] + [f"| {k} | {acc[k][1]} | {100 * acc[k][0] / acc[k][1]:.0f} |" for k in keys if acc[k][1]]
# 2. by active skill
md += table("Group accuracy by the skill that was driving:", lambda r: r[4], ["stand", "walk", "run", "terrain", "stepover"])
# 3. false alarms on flat steps
fl = [r for r in rows if r[2] == "flat"]; n_fl = sum(r[-1] for r in fl); alarm = Counter(); [alarm.__setitem__(r[3], alarm[r[3]] + r[-1]) for r in fl]
md += ["", f"Truth = flat ({100 * n_fl / total:.0f} % of steps): what the classifier said:", "", "| says | % of the flat steps |", "|---|---|"] + [f"| {c} | {100 * alarm[c] / n_fl:.0f} |" for c in CLASSES]
md += table("False alarms (says patch or bar although truth = flat) by the distance to the next section's start (m; the oracle hint starts 1 m ahead of a patch, 2 m ahead of a bar):", lambda r: (r[6] if r[6] < 6 else 6), list(range(-5, 7)), lambda r: r[2] == "flat", "correct (says flat)")
# 4. by section kind when truth is a patch / the bar
for kind in ("rough", "slope_up", "slope_down", "stairs_up", "stairs_down", "bar"):
    sub = [r for r in rows if r[2] == kind]
    if not sub: continue
    n = sum(r[-1] for r in sub); say = Counter(); [say.__setitem__(r[3], say[r[3]] + r[-1]) for r in sub]
    md.append(f"\nTruth = {kind} ({n} steps): says " + ", ".join(f"{c} {100 * v / n:.0f} %" for c, v in say.most_common(4)))
# 5. successful vs failed races
for ok, name in ((True, "finished races"), (False, "unfinished races")):
    sub = [r for r in rows if r[1] == ok]
    if sub: md.append(f"\n{name}: group accuracy {100 * sum(r[-1] for r in sub if g(r[2]) == g(r[3])) / sum(r[-1] for r in sub):.0f} % over {sum(r[-1] for r in sub)} steps")
open(ROOT / f"outputs/analysis/race/diagnose{a.tag}.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
