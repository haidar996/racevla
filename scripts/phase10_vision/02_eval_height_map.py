"""Phase 10, V2: how good is the elevation map built from the depth camera (racevla/vision/heightmap.py), and does the step-over skill still work with it?
Usage: python 02_eval_height_map.py [--episodes 5] [--trials 40]
Part A (accuracy): the rule supervisor drives over bars / stairs / rough ground / a slope (simulator heights as before, the map is only compared); at every step the map's 24-number scan and its heights near the robot (0-0.8 m ahead of the base, the area the
 foot-lift rule reads, mostly in the camera's blind zone) are compared with the true ones. Error = estimate minus truth in cm.
Part B (closed loop): bar trials of scripts/phase7_skills/24_test_patch_trials.py (run approach, oracle 'bar' hint, step-over skill) with the scan AND the foot-lift rule reading (1) the simulator's height field, (2) the camera map at 64 x 48, (3) the camera map at 96 x 72.
Outputs: outputs/analysis/vision/v2_map_accuracy.md, v2_bar_trials.md."""
import os, sys, pathlib, importlib.util, multiprocessing as mp; os.environ.setdefault("MUJOCO_GL", "egl"); ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np

OUT = ROOT / "outputs" / "analysis" / "vision"; OUT.mkdir(parents=True, exist_ok=True)
ACC_CELLS = [("hurdle", 0.05), ("hurdle", 0.10), ("stairs_up", 0.04), ("stairs_down", 0.06), ("rough", 0.06), ("slope_up", 9)]
BAR_CELLS = [0.03, 0.05, 0.07, 0.10]; VARIANTS = [("simulator", None), ("camera 64x48", (64, 48)), ("camera 96x72", (96, 72))]


def load_trials():
    sp = importlib.util.spec_from_file_location("trials", ROOT / "scripts/phase7_skills/24_test_patch_trials.py"); t = importlib.util.module_from_spec(sp); sp.loader.exec_module(t); return t


def trial(job): return load_trials().run(job)               # (a top-level function, so that the process pool can pickle it)


def accuracy(job):
    kind, level, res, n = job
    import torch; torch.set_num_threads(1)
    from racevla.skills.library import Supervisor
    from racevla.skills.rules import RuleSupervisor
    from racevla.vision.heightmap import HeightMapper
    from racevla.envs.go1_hurdle import Go1HurdleEnv
    from racevla.envs.terrain import TerrainMixin
    import mujoco
    t = load_trials(); rs = RuleSupervisor(Supervisor(terrain=True), hold_heading=True); sup, body = rs.sup, rs.sup.body; mapper = HeightMapper(body, *res); sup.vision = mapper
    truth_fn = lambda x, y: TerrainMixin.ground_height_at(body, x, y)                          # the simulator's height field (class method, not the attribute that attach() sets)
    scan_err, near_err = [], []; bar = kind == "hurdle"
    for i in range(n):
        seed = 3000 + i; cmd = 1.5 if bar else 0.5
        rs.reset(seed, options={"terrain": (kind, level), "terrain_seed": seed, "yaw": 0.0}, start_x=-0.5); rs.set_target(cmd, 0.0)
        for step in range(900 if not bar else 450):
            x = float(body.data.qpos[0])
            lo, hi = (t.BAR_X - t.BAR_LOOK, t.BAR_X + t.BAR_AFTER) if bar else (t.EXTENT.get(kind, (1.0, 7.0))[0] - t.LOOK, t.EXTENT.get(kind, (1.0, 7.0))[1] + t.AFTER)
            rs.set_hint(("bar" if bar else kind) if lo <= x <= hi else None); term, info = rs.step()
            if term: break
            if lo - 1.0 <= x <= hi and step % 2 == 0:
                body.scan_noise = 0.0; est = Go1HurdleEnv.scan(body); body.ground_height_at = truth_fn; true = Go1HurdleEnv.scan(body); body.ground_height_at = mapper.query      # (the true scan uses the simulator's heights)
                scan_err.append((est - true).reshape(8, 3).mean(axis=1) / 4.0 * 100)                                    # cm, per scan row (mean over the 3 columns)
                R = body.data.xmat[1].reshape(3, 3); yaw = float(np.arctan2(R[1, 0], R[0, 0])); bx, by = body.data.qpos[:2]
                ahead = np.arange(0.0, 0.81, 0.1); px = bx + np.cos(yaw) * ahead[:, None] - np.sin(yaw) * np.array([-0.127, 0.0, 0.127])[None]; py = by + np.sin(yaw) * ahead[:, None] + np.cos(yaw) * np.array([-0.127, 0.0, 0.127])[None]
                near_err.append((mapper.query(px, py) - truth_fn(px, py)).mean(axis=1) * 100)                         # cm, per distance ahead of the base
    return dict(kind=kind, level=level, res=res, scan=np.array(scan_err), near=np.array(near_err))


if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=5); ap.add_argument("--trials", type=int, default=40); a = ap.parse_args()
    jobs = [(k, l, r, a.episodes) for k, l in ACC_CELLS for r in ((64, 48), (96, 72))]
    with mp.Pool(4, maxtasksperchild=1) as pool: res = pool.map(accuracy, jobs, chunksize=1)
    md = ["Map error in cm (estimate minus truth) while the robot drives over the patch. Scan rows = 0.2 ... 1.6 m ahead of the base (mean over the 3 columns); near = 0 ... 0.8 m ahead of the base.", "",
          "| patch | camera | scan RMS (all rows) | scan |error| 95th pct | scan RMS row 0.2 m | row 0.8 m | row 1.6 m | near-field RMS 0-0.8 m | near-field bias |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        s, nr = r["scan"], r["near"]; md.append(f"| {r['kind']} {r['level']} | {r['res'][0]}x{r['res'][1]} | {np.sqrt((s ** 2).mean()):.1f} | {np.percentile(np.abs(s), 95):.1f} | {np.sqrt((s[:, 0] ** 2).mean()):.1f} | {np.sqrt((s[:, 3] ** 2).mean()):.1f} | {np.sqrt((s[:, 7] ** 2).mean()):.1f} | {np.sqrt((nr ** 2).mean()):.1f} | {nr.mean():+.1f} |")
    open(OUT / "v2_map_accuracy.md", "w").write("\n".join(md) + "\n"); print("\n".join(md), flush=True)
    t = load_trials(); jobs = [("supervisor", "hurdle", h, a.trials, v) for h in BAR_CELLS for _, v in VARIANTS]
    with mp.Pool(4, maxtasksperchild=1) as pool: out = pool.map(trial, jobs, chunksize=1)
    md = ["Bar trials (success %, falls), run approach, oracle 'bar' hint, step-over skill reading heights from:", "", "| bar | " + " | ".join(n for n, _ in VARIANTS) + " |", "|---|" + "---|" * len(VARIANTS)]
    for h in BAR_CELLS:
        row = [out[i] for i, j in enumerate(jobs) if j[2] == h]; md.append(f"| {h * 100:g} cm | " + " | ".join(f"{100 * sum(x['success'] for x in r) / len(r):.0f} % (falls {sum(x['fell'] for x in r)}, stuck {sum(x['stuck'] for x in r)})" for r in row) + " |")
    open(OUT / "v2_bar_trials.md", "w").write("\n".join(md) + "\n"); print("\n".join(md))
