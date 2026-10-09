"""Phase 10, V3: collect depth images with terrain labels. Usage: python 03_collect_depth_data.py [--per-kind 160] [--out data/vision]
The rule supervisor (oracle hint, simulator heights: this is only the DRIVER) crosses one patch per episode on the terrain course exactly like scripts/phase7_skills/24_test_patch_trials.py; every 10th step (5 Hz) the onboard depth image (64 x 48) is saved with its label
(racevla/vision/classifier.py: hint_label), the patch kind and level, the robot x, tilt, speed and the episode's gait mode. GAIT MODES (added after the first classifier learned 'running -> bar' because bar episodes were always run and everything else walked):
every kind is driven half of the time WALKING (target 0.3-0.7 m/s, oracle hint as before) and half of the time RUNNING (target 1.0-2.0 m/s; no hint on terrain patches, so the run skill keeps going and may fall: the frames until then are valid; the bar keeps its hint with
target 1.3-1.7); the bar in walking mode: target 0.3-0.7 with the 'bar' hint, so the supervisor accelerates to the run first. Kinds: flat (rough level 0), rough 2-10 cm, slopes 3-15 deg, stairs 2-10 cm, bar 3-10 cm; levels uniform. Episode seeds 10000 + i; the split is by episode seed
(seed % 10: 0 = test, 1 = validation, else training), never by frame. One .npz shard per kind in the output folder."""
import os, sys, pathlib, argparse, multiprocessing as mp; os.environ.setdefault("MUJOCO_GL", "egl"); ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np

LEVELS = {"flat": [0.0], "rough": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10], "slope_up": [3, 6, 9, 12, 15], "slope_down": [3, 6, 9, 12, 15], "stairs_up": [0.02, 0.04, 0.06, 0.08, 0.10], "stairs_down": [0.02, 0.04, 0.06, 0.08, 0.10], "hurdle": [0.03, 0.05, 0.07, 0.10]}
EVERY, GOAL_X, MAX_STEPS, BAR_STEPS = 10, 8.0, 1300, 700


def collect(job):
    kind, n, first = job
    import torch; torch.set_num_threads(1)
    from racevla.skills.library import Supervisor
    from racevla.skills.rules import RuleSupervisor
    from racevla.vision.camera import OnboardCamera
    from racevla.vision.classifier import hint_label, to_hint, CLASSES
    rs = RuleSupervisor(Supervisor(terrain=True), hold_heading=True); sup, body = rs.sup, rs.sup.body; cam = OnboardCamera(body.model); rng0 = np.random.default_rng(first)
    D, Y, META = [], [], []; bar = kind == "hurdle"
    for i in range(n):
        seed = first + i; rng = np.random.default_rng(seed); level = float(rng.choice(LEVELS[kind])); run_mode = bool(rng.random() < 0.5); cmd = float((rng.uniform(1.3, 1.7) if bar else rng.uniform(1.0, 2.0)) if run_mode else rng.uniform(0.3, 0.7))
        terrain = ("rough", 0.0) if kind == "flat" else (kind, level)
        rs.reset(seed, options={"terrain": terrain, "terrain_seed": seed, "yaw": 0.0}, start_x=-0.5); rs.set_target(cmd, 0.0)
        for t in range(BAR_STEPS if bar else MAX_STEPS):
            x = float(body.data.qpos[0]); lab = hint_label(kind, x); rs.set_hint(to_hint(lab) if (bar or not run_mode) else None); term, info = rs.step()
            if term: break
            if t % EVERY == 0:
                D.append(cam.render(body.data).astype(np.float16)); Y.append(lab); META.append((seed, level, x, body.tilt_deg(), info["vx"], float(run_mode), *sup.base[24:27]))      # last three = the gravity vector in the body frame (IMU)
            if (not bar) and x >= GOAL_X or bar and x >= 8.9: break
    return kind, np.array(D), np.array(Y, np.int8), np.array(META, np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--per-kind", type=int, default=160); ap.add_argument("--out", default="data/vision"); a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True); kinds = list(LEVELS); chunk = 20
    jobs = [(k, chunk, 10000 + 100000 * ki + c * chunk) for ki, k in enumerate(kinds) for c in range(a.per_kind // chunk)]
    shards = {k: [] for k in kinds}
    with mp.Pool(4, maxtasksperchild=1) as pool:
        for j, (k, D, Y, M) in enumerate(pool.imap_unordered(collect, jobs, chunksize=1)):
            shards[k].append((D, Y, M)); print(f"[{j + 1}/{len(jobs)}] {k}: {len(Y)} frames", flush=True)
    for k, parts in shards.items():
        np.savez_compressed(out / f"{k}.npz", depth=np.concatenate([p[0] for p in parts]), label=np.concatenate([p[1] for p in parts]), meta=np.concatenate([p[2] for p in parts])); print("saved", k, sum(len(p[1]) for p in parts), "frames")
