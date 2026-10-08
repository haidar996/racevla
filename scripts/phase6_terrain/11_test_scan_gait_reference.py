"""Phase 6, Skill B option A: the REFERENCE gait alone (policy output 0, no balance feedback, no learning) on the terrain, with and without the scan-aware terrain term (racevla/envs/go1_scan_gait.py).
Variants: base (linear IK, no terrain term = the old reference with a 12 cm lift), exactik (exact IK only), terrain (exact IK + terrain-following term). Forward command 0.4 / 0.6 m/s with heading hold, 10 episodes per cell.
Usage: python 11_test_scan_gait_reference.py   -> prints a table, writes outputs/analysis/terrain/scan_gait_reference.csv"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import multiprocessing as mp
import numpy as np

VARIANTS = {"base": dict(terrain_ref=False, exact_ik=False), "exactik": dict(terrain_ref=False, exact_ik=True), "terrain": dict(terrain_ref=True, exact_ik=True, ref_mode="additive"), "terrainmax": dict(terrain_ref=True, exact_ik=True, ref_mode="max")}
ONLY = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else list(VARIANTS)       # --only terrainmax: run just that variant (output scan_gait_reference_<names>.csv)
CONDITIONS = [("rough", 0.0), ("rough", 0.06), ("rough", 0.10), ("slope_up", 12), ("slope_up", 15), ("slope_down", 15), ("stairs_up", 0.04), ("stairs_up", 0.06), ("stairs_up", 0.08), ("stairs_up", 0.10), ("stairs_up", 0.12), ("stairs_up", 0.15),
              ("stairs_down", 0.06), ("stairs_down", 0.10), ("stairs_down", 0.12), ("stairs_down", 0.15)]
N_EP, MAX_STEPS = 10, 1000


def run(job):
    variant, speed, kind, level = job
    from racevla.envs.go1_scan_gait import Go1ScanGaitEnv
    from racevla.envs.terrain import X_GOAL
    env = Go1ScanGaitEnv(scan_noise=0.0, **VARIANTS[variant]); env._sample_command = lambda: None; out = {"success": 0, "fell": 0, "left_course": 0, "stuck": 0}; xs = []
    for s in range(N_EP):
        env.reset(seed=s, options={"terrain": (kind, level), "terrain_seed": s, "yaw": 0.0}); res, xm = "stuck", 0.0
        for t in range(MAX_STEPS):
            R = env.data.xmat[1].reshape(3, 3); yaw = float(np.arctan2(R[1, 0], R[0, 0])); env.command = np.array([speed, float(np.clip(-2.0 * yaw, -0.5, 0.5))])
            _, r, term, trunc, info = env.step(np.zeros(12, np.float32)); xm = max(xm, float(env.data.qpos[0]))
            if xm >= X_GOAL: res = "success"; break
            if term: res = "left_course" if env.termination_reason == "left_course" else "fell"; break
        out[res] += 1; xs.append(xm)
    return {"variant": variant, "speed": speed, "kind": kind, "level": level, **out, "mean_x": float(np.mean(xs))}


if __name__ == "__main__":
    jobs = [(v, sp, k, lv) for v in ONLY for sp in (0.4, 0.6) for k, lv in CONDITIONS]; print(f"{len(jobs)} conditions x {N_EP} episodes", flush=True); rows = []
    with mp.Pool(4, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run, jobs, chunksize=1): rows.append(r)
    import csv; out = ROOT / "outputs" / "analysis" / "terrain"; out.mkdir(parents=True, exist_ok=True)
    with open(out / ("scan_gait_reference.csv" if ONLY == list(VARIANTS) else "scan_gait_reference_" + "_".join(ONLY) + ".csv"), "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    d = {(r["variant"], r["speed"], r["kind"], r["level"]): r for r in rows}
    print("| terrain | level | " + " | ".join(f"{v} {sp}" for sp in (0.4, 0.6) for v in ONLY) + " |\n|---|---|" + "---|" * (2 * len(ONLY)))
    for k, lv in CONDITIONS:
        sh = f"{lv:g} deg" if k.startswith("slope") else f"{lv * 100:g} cm"
        print(f"| {k} | {sh} | " + " | ".join(f"{d[(v, sp, k, lv)]['success'] * 10}% (x {d[(v, sp, k, lv)]['mean_x']:.1f})" for sp in (0.4, 0.6) for v in ONLY) + " |")
