"""Live plot of a training run's evaluation curve, updated as new checkpoints land.
Reads outputs/<run>/evaluations.npz (written by SB3's EvalCallback) and redraws every few seconds.
Safe to run alongside training: it only reads that file, never writes to it.
Usage: python watch_plot.py <run_dir> [--every 5] [--baseline 60]"""
import sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib; matplotlib.use("TkAgg")   # a real window, not a saved file
import matplotlib.pyplot as plt
from racevla.envs.go1_standing import MAX_EPISODE_STEPS   # 1000: an episode this long means it never fell

args = [a for a in sys.argv[1:] if not a.startswith("--")]
run_dir = ROOT / args[0]
every = float(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 5.0
baseline_pct = float(sys.argv[sys.argv.index("--baseline") + 1]) if "--baseline" in sys.argv else 60.0
npz_path = run_dir / "evaluations.npz"

plt.ion()
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax2 = ax1.twinx()
fig.suptitle(f"training progress: {run_dir.name}  (live)")
ax1.set_xlabel("training steps"); ax1.set_ylabel("mean return"); ax2.set_ylabel("% episodes fallen")
ax2.set_ylim(0, 100)
ax2.axhline(baseline_pct, color="gray", linestyle="--", linewidth=1)

print(f"watching {npz_path} — close the plot window or Ctrl+C here to stop.")
last_n = -1
while plt.fignum_exists(fig.number):
    if npz_path.exists():
        try:
            ev = np.load(npz_path)
            steps, results, lengths = ev["timesteps"], ev["results"], ev["ep_lengths"]
        except Exception:
            time.sleep(every); continue          # file mid-write from the training process; try again next tick
        if len(steps) != last_n:
            last_n = len(steps)
            ret = results.mean(axis=1)
            n_eval = lengths.shape[1]
            fell_pct = 100 * (lengths < MAX_EPISODE_STEPS).sum(axis=1) / n_eval   # shorter than the cap => it fell
            ax1.clear(); ax2.clear()
            ax1.set_xlabel("training steps"); ax1.set_ylabel("mean return"); ax2.set_ylabel("% episodes fallen")
            ax2.set_ylim(0, 100); ax2.axhline(baseline_pct, color="gray", linestyle="--", linewidth=1, label=f"baseline (~{baseline_pct:.0f}%)")
            ax1.plot(steps, ret, "o-", color="#2b6cb0", label="mean return")
            ax2.plot(steps, fell_pct, "s--", color="#c53030", alpha=0.6, label="% fallen")
            ax1.set_title(f"{steps[-1]:,} steps so far  |  latest: return={ret[-1]:.0f}, fell={fell_pct[-1]:.0f}%")
            lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=8)
            fig.tight_layout()
    fig.canvas.draw_idle(); fig.canvas.flush_events()
    plt.pause(every)
print("plot window closed.")
