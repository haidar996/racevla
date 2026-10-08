"""Phase 6, lift test: how does the peak foot lift of the reference gait (default 8 cm) change the crossing of tall bumps and steps? Evaluates ONE model at one or more lift heights (the lift is set on the environment, the policy is not changed)
on a fixed set of terrains at a forward command of 0.6 m/s with heading hold, 20 episodes per cell, same harness as 02_baseline_policies.py. Output: outputs/analysis/terrain/lift_<label>.csv.
Usage: python 07_lift_eval.py --model path.zip --lifts 0.08 0.10 0.12 --label name"""
import sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import multiprocessing as mp
import numpy as np

CONDITIONS = [("rough", 0.0), ("rough", 0.08), ("rough", 0.10), ("slope_up", 15), ("stairs_up", 0.04), ("stairs_up", 0.06), ("stairs_up", 0.08), ("stairs_up", 0.10), ("stairs_up", 0.12), ("stairs_down", 0.06), ("stairs_down", 0.10)]
SPEED, N_EP, MAX_STEPS = 0.6, 20, 1000


def run(job):
    model_path, lift, kind, level = job
    import torch; torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from racevla.envs.terrain import make_terrain_env, X_GOAL
    from racevla.envs.go1_walking_sine import Go1WalkingSineEnv
    from racevla.envs.wrappers import FixedObsNormalize
    model = PPO.load(ROOT / model_path, device="cpu"); env = FixedObsNormalize(make_terrain_env(Go1WalkingSineEnv)(push=False)); raw = env.unwrapped; raw._sample_command = lambda: None; raw.lift = lift
    out = {"success": 0, "fell": 0, "left_course": 0, "stuck": 0}
    for s in range(N_EP):
        obs, _ = env.reset(seed=s, options={"terrain": (kind, level), "terrain_seed": s, "yaw": 0.0}); res = "stuck"
        for t in range(MAX_STEPS):
            R = raw.data.xmat[1].reshape(3, 3); yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            raw.command = np.array([SPEED, float(np.clip(-2.0 * yaw, -0.5, 0.5))]); obs[-4:-2] = raw.command; obs[-2:] = raw._clock()
            obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            if raw.data.qpos[0] >= X_GOAL: res = "success"; break
            if term: res = "left_course" if raw.termination_reason == "left_course" else "fell"; break
        out[res] += 1
    return {"lift": lift, "kind": kind, "level": level, **out}


if __name__ == "__main__":
    arg = lambda n, d=None: sys.argv[sys.argv.index(n) + 1] if n in sys.argv else d
    model_path, label = arg("--model"), arg("--label"); i = sys.argv.index("--lifts") + 1; lifts = []
    while i < len(sys.argv) and not sys.argv[i].startswith("--"): lifts.append(float(sys.argv[i])); i += 1
    jobs = [(model_path, lf, k, lv) for lf in lifts for k, lv in CONDITIONS]; print(f"{len(jobs)} conditions x {N_EP} episodes", flush=True)
    rows = []
    with mp.Pool(4, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run, jobs, chunksize=1): rows.append(r); print(f"lift {r['lift']} {r['kind']} {r['level']}: success {r['success']}/{N_EP}, fell {r['fell']}, left {r['left_course']}, stuck {r['stuck']}", flush=True)
    import csv; out = ROOT / "outputs" / "analysis" / "terrain"; out.mkdir(parents=True, exist_ok=True)
    with open(out / f"lift_{label}.csv", "w", newline="") as f: w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
