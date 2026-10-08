"""Robust evaluation of step-over (hurdle) models: 20 episodes per bar height (empty course, 3, 5, 7, 10, 12.5, 15, 20 cm), command ~ U(1.3, 1.7) m/s, deterministic policy, seeds 2000..2019 (fresh).
Per cell: success (2 s after the bar's position upright at >= 50 % of the speed), cleared (crossed without trunk / hip / thigh touching the bar), hit, fell.
Usage: python 17_eval_stepover.py <label>=<model.zip> [<label>=<model.zip> ...]   -> outputs/analysis/jump/stepover_eval.csv and a printed table"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import multiprocessing as mp
import numpy as np

HEIGHTS = [("rough", 0.0), ("hurdle", 0.03), ("hurdle", 0.05), ("hurdle", 0.07), ("hurdle", 0.10), ("hurdle", 0.125), ("hurdle", 0.15), ("hurdle", 0.20)]
N_EP = 20


def run(job):
    label, path, terrain = job
    import torch; torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from racevla.envs.go1_stepover import Go1StepOverEnv
    from racevla.envs.wrappers import FixedObsNormalize
    model = PPO.load(ROOT / path, device="cpu"); env = FixedObsNormalize(Go1StepOverEnv()); out = {"success": 0, "cleared": 0, "hit": 0, "fell": 0}
    for s in range(N_EP):
        obs, _ = env.reset(seed=2000 + s, options={"terrain": terrain, "terrain_seed": 2000 + s})
        for t in range(450):
            obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            if term or trunc: break
        rs = info.get("termination_reason"); out["success"] += int(bool(info.get("success"))); out["hit"] += int(rs == "hit_hurdle"); out["fell"] += int(term and rs != "hit_hurdle"); out["cleared"] += int(info.get("crossed", 0) > 0 and rs != "hit_hurdle")
    return {"label": label, "terrain": terrain, **out}


if __name__ == "__main__":
    models = [a.split("=", 1) for a in sys.argv[1:]]; jobs = [(l, p, t) for l, p in models for t in HEIGHTS]
    with mp.Pool(4, maxtasksperchild=1) as pool: rows = pool.map(run, jobs, chunksize=1)
    import csv; od = ROOT / "outputs" / "analysis" / "jump"; od.mkdir(parents=True, exist_ok=True)
    with open(od / "stepover_eval.csv", "w", newline="") as f: w = csv.writer(f); w.writerow(["label", "kind", "height", "success", "cleared", "hit", "fell"]); [w.writerow([r["label"], r["terrain"][0], r["terrain"][1], r["success"], r["cleared"], r["hit"], r["fell"]]) for r in rows]
    print("| model | " + " | ".join("empty" if k == "rough" else f"{h * 100:g} cm" for k, h in HEIGHTS) + " |\n|---|" + "---|" * len(HEIGHTS))
    for l, _ in models: print(f"| {l} | " + " | ".join(f"{5 * next(r for r in rows if r['label'] == l and r['terrain'] == t)['success']}% (c{5 * next(r for r in rows if r['label'] == l and r['terrain'] == t)['cleared']})" for t in HEIGHTS) + " |")
    print("cells: success % of 20 episodes (c = cleared-the-bar %)")
