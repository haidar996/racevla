"""Phase 10, V3: the whole vision pipeline in the closed loop. Usage: python 06_eval_vision_pipeline.py [--trials 30] [--model outputs/vision/terrain_cnn.pt]
The patch trials of scripts/phase7_skills/24_test_patch_trials.py (one patch per trial, supervisor with all skills), three ways of getting the terrain information:
  oracle      : the oracle hint and the simulator's height field (the stage 4c setting)
  vision x1   : the hint from the terrain CNN on the newest depth image (one frame, no filtering) + the elevation map of the depth camera for the step-over skill
  vision x5   : the same with a majority vote over the last 5 predictions (0.1 s)
Outputs outputs/analysis/vision/v3_pipeline.md (success %, falls, share of steps where the hint agrees with the oracle)."""
import os, sys, pathlib, argparse, importlib.util, multiprocessing as mp; os.environ.setdefault("MUJOCO_GL", "egl"); ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np

OUT = ROOT / "outputs" / "analysis" / "vision"


def trial(job):
    sp = importlib.util.spec_from_file_location("trials", ROOT / "scripts/phase7_skills/24_test_patch_trials.py"); t = importlib.util.module_from_spec(sp); sp.loader.exec_module(t); return t.run(job)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=30); ap.add_argument("--model", default="outputs/vision/terrain_cnn.pt"); ap.add_argument("--tag", default=""); ap.add_argument("--gravity", action="store_true"); a = ap.parse_args(); n = a.trials
    sp = importlib.util.spec_from_file_location("trials", ROOT / "scripts/phase7_skills/24_test_patch_trials.py"); t = importlib.util.module_from_spec(sp); sp.loader.exec_module(t)
    cells = t.TERRAIN_CELLS + t.BAR_CELLS; V = [("oracle", None, None), ("vision x1", (64, 48), (a.model, 1, a.gravity)), ("vision x5", (64, 48), (a.model, 5, a.gravity))]
    jobs = [("supervisor", k, l, n, v, c) for k, l in cells for _, v, c in V]
    with mp.Pool(4, maxtasksperchild=1) as pool: res = pool.map(trial, jobs, chunksize=1)
    md = [f"Patch trials ({n} per cell), success % (falls / stuck), and agreement of the hint with the oracle hint (share of steps):", "", "| patch | " + " | ".join(v for v, _, _ in V) + " |", "|---|" + "---|" * len(V)]
    tot = {v: [0, 0] for v, _, _ in V}
    for k, l in cells:
        row = []
        for (name, _, c), j, r in zip(V * len(cells), jobs, res):
            pass
        for vi, (name, v, c) in enumerate(V):
            r = res[cells.index((k, l)) * len(V) + vi]; s = sum(x["success"] for x in r); f = sum(x["fell"] for x in r); st = sum(x["stuck"] for x in r); tot[name][0] += s; tot[name][1] += len(r)
            hm = f", hint agrees {100 * sum(x['hint_match'] for x in r) / max(sum(x['hint_steps'] for x in r), 1):.0f} %" if c else ""; row.append(f"{100 * s / len(r):.0f} % ({f} / {st}){hm}")
        md.append(f"| {k} {l} | " + " | ".join(row) + " |")
    md.append("| **all cells** | " + " | ".join(f"**{100 * tot[v][0] / tot[v][1]:.0f} %**" for v, _, _ in V) + " |")
    open(OUT / f"v3_pipeline{a.tag}.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
