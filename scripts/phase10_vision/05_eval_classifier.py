"""Phase 10, V3: evaluate the trained terrain CNN on the TEST episodes (never seen in training). Usage: python 05_eval_classifier.py [--model outputs/vision/terrain_cnn.pt]
Reports: accuracy and balanced accuracy over the 7 classes, over the 3 groups the supervisor acts on (flat / terrain patch / bar), the confusion matrix (figure + table), accuracy against the distance to the patch start, and the accuracy when the test depth gets extra noise.
Outputs outputs/analysis/vision/v3_classifier.md and v3_confusion.png."""
import sys, pathlib, argparse, importlib.util; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from racevla.vision.classifier import CLASSES, TerrainCNN

sp = importlib.util.spec_from_file_location("train", ROOT / "scripts/phase10_vision/04_train_classifier.py"); T = importlib.util.module_from_spec(sp); sp.loader.exec_module(T)
GROUP = np.array([0, 1, 1, 1, 1, 1, 2])           # class index -> 0 flat, 1 terrain patch, 2 bar


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="outputs/vision/terrain_cnn.pt"); ap.add_argument("--data", default="data/vision"); ap.add_argument("--gravity", action="store_true"); ap.add_argument("--tag", default=""); a = ap.parse_args()
    D, Y, M = T.load(ROOT / a.data); te = np.where(M[:, 0].astype(int) % 10 == 0)[0]; net = TerrainCNN(use_gravity=a.gravity); net.load_state_dict(torch.load(ROOT / a.model, map_location="cpu")); torch.set_num_threads(4)
    G = M[:, 7:10].astype(np.float32) if a.gravity else None; Gt = None if G is None else G[te]; p = T.predict(net, D[te], Gt); y = Y[te]; rng = np.random.default_rng(0); OUT = ROOT / "outputs" / "analysis" / "vision"; OUT.mkdir(parents=True, exist_ok=True)
    md = [f"Test set: {len(te)} frames from {len(set(M[te, 0].astype(int)))} episodes never used in training.", "", f"- 7-class accuracy {100 * (p == y).mean():.1f} %, balanced (mean per-class recall) {100 * T.balanced_accuracy(y, p):.1f} %",
          f"- 3 groups (flat / terrain patch / bar): accuracy {100 * (GROUP[p] == GROUP[y]).mean():.1f} %", "", "| class | frames | recall % | most common mistake |", "|---|---|---|---|"]
    cm = np.zeros((7, 7), int); np.add.at(cm, (y, p), 1)
    for c, name in enumerate(CLASSES):
        row = cm[c].copy(); n = row.sum(); row[c] = 0; md.append(f"| {name} | {n} | {100 * cm[c, c] / max(n, 1):.1f} | {CLASSES[int(row.argmax())] + f' ({100 * row.max() / max(n, 1):.1f} %)' if row.max() > 0 else '-'} |")
    kind, x = M[te, 5].astype(int), M[te, 2]; start = np.where(kind == 6, 5.0, 1.0); dist = start - x; md += ["", "Accuracy against the distance from the robot's base to the patch start (patch kinds only, flat episodes excluded; negative = already on the patch):", "", "| distance (m) | frames | 7-class acc % | 3-group acc % |", "|---|---|---|---|"]
    m0 = kind != 0
    for lo, hi in [(-9, -1), (-1, 0), (0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 9)]:
        s = m0 & (dist >= lo) & (dist < hi)
        if s.sum(): md.append(f"| {lo:g} to {hi:g} | {int(s.sum())} | {100 * (p[s] == y[s]).mean():.1f} | {100 * (GROUP[p[s]] == GROUP[y[s]]).mean():.1f} |")
    md += ["", "Robustness: extra depth noise on the test images (sigma = noise x depth, plus 1 % lost pixels):", "", "| extra noise | 7-class acc % | 3-group acc % |", "|---|---|---|"]
    for sig in (0.0, 0.01, 0.02, 0.05):
        d = T.augment(D[te].astype(np.float32), rng, sigma=sig, drop=0.01 if sig else 0.0) if sig else D[te]; q = T.predict(net, d, Gt); md.append(f"| {sig:g} | {100 * (q == y).mean():.1f} | {100 * (GROUP[q] == GROUP[y]).mean():.1f} |")
    open(OUT / f"v3_classifier{a.tag}.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1); ax.set_xticks(range(7)); ax.set_yticks(range(7)); ax.set_xticklabels(CLASSES, rotation=45, ha="right"); ax.set_yticklabels(CLASSES)
    for i in range(7):
        for j in range(7): ax.text(j, i, f"{100 * cm[i, j] / max(cm[i].sum(), 1):.0f}", ha="center", va="center", fontsize=8, color="white" if cm[i, j] / max(cm[i].sum(), 1) > 0.5 else "black")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("terrain CNN, test episodes (row % of each true class)"); fig.colorbar(im); fig.savefig(OUT / f"v3_confusion{a.tag}.png", dpi=110, bbox_inches="tight")
