"""Phase 10, V3: train the terrain CNN on the collected depth images. Usage: python 04_train_classifier.py [--epochs 12] [--data data/vision] [--out outputs/vision]
Split by EPISODE seed (seed % 10: 0 = test, 1 = validation, else training) so that no episode is in two sets. Class-balanced cross-entropy (the 'flat' class is by far the largest), Adam 1e-3 with cosine decay, batch 128. Augmentation (training only, new random draw every
batch): Gaussian depth noise (sigma 1 cm, scaled up with the distance: 1 cm * depth / 1 m) and 1 % of the pixels randomly lose their return (set to the maximum depth). The model with the best validation balanced accuracy is saved to <out>/terrain_cnn.pt."""
import sys, pathlib, argparse, time; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, torch, torch.nn as nn
from racevla.vision.classifier import CLASSES, TerrainCNN, MAX_DEPTH

KINDS = ["flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down", "hurdle"]


def load(data_dir):
    D, Y, M = [], [], []
    for k in KINDS:
        z = np.load(data_dir / f"{k}.npz"); D.append(z["depth"]); Y.append(z["label"]); m = z["meta"]; M.append(np.concatenate([m[:, :5], np.full((len(z["label"]), 1), KINDS.index(k), np.float32), m[:, 5:]], axis=1))
    return np.concatenate(D), np.concatenate(Y).astype(np.int64), np.concatenate(M)           # meta columns: seed, level, x, tilt, speed, kind index, gait mode (0 walk, 1 run; second dataset on), gravity vector x, y, z (third dataset on)


def augment(d, rng, sigma=0.01, drop=0.01):
    d = d + rng.normal(0.0, 1.0, d.shape).astype(np.float32) * sigma * np.maximum(d, 0.3); d[rng.random(d.shape) < drop] = MAX_DEPTH; return np.clip(d, 0.0, MAX_DEPTH)


def balanced_accuracy(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in range(len(CLASSES)) if (y == c).any()]))


@torch.no_grad()
def predict(net, D, G=None, batch=1024):
    net.eval(); out = []
    for i in range(0, len(D), batch): out.append(net(torch.from_numpy(D[i:i + batch].astype(np.float32) / MAX_DEPTH)[:, None], None if G is None else torch.from_numpy(G[i:i + batch].astype(np.float32))).argmax(1).numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--data", default="data/vision"); ap.add_argument("--out", default="outputs/vision"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--gravity", action="store_true", help="also feed the body gravity vector (meta columns 7-9) to the network"); a = ap.parse_args()
    torch.set_num_threads(4); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    D, Y, M = load(ROOT / a.data); G = M[:, 7:10].astype(np.float32) if a.gravity else None; s = M[:, 0].astype(int) % 10; tr, va, te = np.where(s >= 2)[0], np.where(s == 1)[0], np.where(s == 0)[0]
    print(f"frames {len(Y)}: train {len(tr)} validation {len(va)} test {len(te)}; class counts (train): " + ", ".join(f"{c} {int((Y[tr] == i).sum())}" for i, c in enumerate(CLASSES)), flush=True)
    w = 1.0 / np.bincount(Y[tr], minlength=len(CLASSES)); w = torch.tensor(w / w.mean(), dtype=torch.float32); net = TerrainCNN(use_gravity=a.gravity); opt = torch.optim.Adam(net.parameters(), lr=1e-3); lossf = nn.CrossEntropyLoss(weight=w)
    steps = a.epochs * (len(tr) // 128); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); best = -1.0; t0 = time.time()
    for ep in range(a.epochs):
        net.train(); perm = rng.permutation(tr); tot = 0.0
        for i in range(0, len(perm) - 127, 128):
            idx = perm[i:i + 128]; x = torch.from_numpy(augment(D[idx].astype(np.float32), rng) / MAX_DEPTH)[:, None]; y = torch.from_numpy(Y[idx])
            gx = torch.from_numpy(G[idx] + rng.normal(0.0, 0.02, G[idx].shape).astype(np.float32)) if a.gravity else None; loss = lossf(net(x, gx), y); opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += float(loss)
        pv = predict(net, D[va], None if G is None else G[va]); ba = balanced_accuracy(Y[va], pv); acc = float((pv == Y[va]).mean())
        if ba > best: best = ba; torch.save(net.state_dict(), out / "terrain_cnn.pt")
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / (len(perm) // 128):.3f} validation accuracy {acc:.3f} balanced {ba:.3f} {'(saved)' if ba == best else ''} [{(time.time() - t0) / 60:.1f} min]", flush=True)
    np.savez(out / "split.npz", train=tr, val=va, test=te)
