"""Phase 6, terrain baseline: how much does terrain break the EXISTING policies (no retraining)? The packaged walking policy (models/walking_policy_sine_seed1.zip) at 0.4 and 0.6 m/s and the packaged running policy
(models/running_policy_2p5_seed0.zip) at 1.0 and 1.5 m/s cross every terrain kind and level (racevla/envs/terrain.py) with a fixed forward command plus a simple HEADING HOLD (turn command = -2 x heading error, limited to +-0.5 rad/s; without it the policies drift sideways off the 3 m wide course even on easy terrain), 20 episodes per condition (terrain seed = episode seed),
starting at x = 0 heading along +x, at most 20 s. Outcome per episode: success (x reaches 6 m), fell (tilt / trunk contact), left_course (|y| > 1.4 m), stuck (still on the course after 20 s without reaching 6 m).
Usage: python 02_baseline_policies.py [--episodes 20] [--set terrain|terrain2m] [--model path.zip --label name [--lift 0.12]]   -> outputs/analysis/terrain/baseline.csv, baseline.md, baseline.png  (--set terrain2m: the 2M-step seed 0 terrain policy -> baseline_terrain2m.* with the old walking and the pilot columns; --set terrain: the terrain-trained pilot policy at 0.4 / 0.6 m/s -> baseline_terrain.* with the old walking columns for comparison)"""
import sys, os, json, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import multiprocessing as mp
import numpy as np

POLICIES = {"walking 0.4": ("models/walking_policy_sine_seed1.zip", "walk", 0.4), "walking 0.6": ("models/walking_policy_sine_seed1.zip", "walk", 0.6),
            "running 1.0": ("models/running_policy_2p5_seed0.zip", "run", 1.0), "running 1.5": ("models/running_policy_2p5_seed0.zip", "run", 1.5)}
NEW = {"terrain-trained 0.4": ("outputs/terr_pilot_seed0/latest_model.zip", "walk", 0.4), "terrain-trained 0.6": ("outputs/terr_pilot_seed0/latest_model.zip", "walk", 0.6)}
NEW2 = {"terrain-2M 0.4": ("outputs/terr_2M_seed0/latest_model.zip", "walk", 0.4), "terrain-2M 0.6": ("outputs/terr_2M_seed0/latest_model.zip", "walk", 0.6)}
ALL = {**POLICIES, **NEW, **NEW2}          # --set terrain: only the NEW policies are run, and the table / figure also show the old walking columns (read from baseline.csv)
MAX_STEPS = 1000


def run_condition(job):
    name, kind, level, n_ep, spec = job
    import torch; torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from racevla.envs.terrain import make_terrain_env, X_GOAL
    from racevla.envs.go1_walking_sine import Go1WalkingSineEnv
    from racevla.envs.go1_running_sine import Go1RunningSineEnv
    from racevla.envs.wrappers import FixedObsNormalize
    path, family, speed, *rest = spec; lift = rest[0] if rest else None; model = PPO.load(ROOT / path, device="cpu")
    if family == "walk": env = FixedObsNormalize(make_terrain_env(Go1WalkingSineEnv)(push=False))
    else:
        table = json.load(open(ROOT / "models/running_gait_table.json")); env = FixedObsNormalize(make_terrain_env(Go1RunningSineEnv)(push=False, gait={"table": table}))
    raw = env.unwrapped; raw._sample_command = lambda: None
    if lift is not None: raw.lift = lift                                    # peak foot lift of the reference gait (models trained with --lift)
    out = {"success": 0, "fell": 0, "left_course": 0, "stuck": 0}; xmax, tgoal = [], []
    for s in range(n_ep):
        obs, _ = env.reset(seed=s, options={"terrain": (kind, level), "terrain_seed": s, "yaw": 0.0}); res, xm = "stuck", 0.0
        for t in range(MAX_STEPS):
            R = raw.data.xmat[1].reshape(3, 3); yaw = float(np.arctan2(R[1, 0], R[0, 0]))
            raw.command = np.array([speed, float(np.clip(-2.0 * yaw, -0.5, 0.5))]); obs[-4:-2] = raw.command; obs[-2:] = raw._clock()      # forward speed + heading hold
            obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); xm = max(xm, float(raw.data.qpos[0]))
            if xm >= X_GOAL: res = "success"; tgoal.append((t + 1) * 0.02); break
            if term: res = "left_course" if raw.termination_reason == "left_course" else "fell"; break
        out[res] += 1; xmax.append(xm)
    return {"policy": name, "kind": kind, "level": level, "episodes": n_ep, **out, "mean_x": float(np.mean(xmax)), "mean_time_to_goal": float(np.mean(tgoal)) if tgoal else float("nan")}


if __name__ == "__main__":
    n_ep = int(sys.argv[sys.argv.index("--episodes") + 1]) if "--episodes" in sys.argv else 20
    from racevla.envs.terrain import LEVELS
    which = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else ""
    if "--model" in sys.argv:                                              # any model: --model path.zip --label name -> policies "name 0.4" and "name 0.6", outputs baseline_<name>.*
        model_path, label = sys.argv[sys.argv.index("--model") + 1], sys.argv[sys.argv.index("--label") + 1]
        lift = float(sys.argv[sys.argv.index("--lift") + 1]) if "--lift" in sys.argv else None; NEW_CUSTOM = {f"{label} 0.4": (model_path, "walk", 0.4, lift), f"{label} 0.6": (model_path, "walk", 0.6, lift)}; ALL.update(NEW_CUSTOM); which = "custom"
    terrain_set = which in ("terrain", "terrain2m", "custom")
    run_policies = {"terrain": NEW, "terrain2m": NEW2}.get(which, NEW_CUSTOM if which == "custom" else POLICIES); suffix = {"terrain": "_terrain", "terrain2m": "_terrain2m"}.get(which, f"_{label}" if which == "custom" else "")
    jobs = [(p, k, lv, n_ep, ALL[p]) for p in run_policies for k, lvls in LEVELS.items() for lv in lvls]; print(f"{len(jobs)} conditions x {n_ep} episodes", flush=True)
    t0 = time.time(); rows = []
    with mp.Pool(4, maxtasksperchild=1) as pool:
        for i, r in enumerate(pool.imap_unordered(run_condition, jobs, chunksize=1)):
            rows.append(r); print(f"[{i + 1}/{len(jobs)} {(time.time() - t0) / 60:.1f} min] {r['policy']} {r['kind']} {r['level']}: success {r['success']}/{n_ep}, fell {r['fell']}, left {r['left_course']}, stuck {r['stuck']}", flush=True)
    out = ROOT / "outputs" / "analysis" / "terrain"; out.mkdir(parents=True, exist_ok=True)
    import csv
    if terrain_set:                                                      # add the old walking columns for comparison
        old = list(csv.DictReader(open(out / "baseline.csv")))
        for r in old:
            if r["policy"] in ("walking 0.4", "walking 0.6"): rows.append({**r, "level": float(r["level"]), **{k: int(r[k]) for k in ("episodes", "success", "fell", "left_course", "stuck")}})
        if which == "terrain2m":                                          # also add the pilot columns
            for r in csv.DictReader(open(out / "baseline_terrain.csv")):
                if r["policy"].startswith("terrain-trained"): rows.append({**r, "level": float(r["level"]), **{k: int(r[k]) for k in ("episodes", "success", "fell", "left_course", "stuck")}})
            shown_policies = {"walking 0.4": 0, "walking 0.6": 0, "terrain-trained 0.4": 0, "terrain-trained 0.6": 0, **run_policies}
        elif which == "custom":                                           # old walking + the 2M seed-0 model for comparison
            for r in csv.DictReader(open(out / "baseline_terrain2m.csv")):
                if r["policy"].startswith("terrain-2M"): rows.append({**r, "level": float(r["level"]), **{k: int(r[k]) for k in ("episodes", "success", "fell", "left_course", "stuck")}})
            shown_policies = {"walking 0.4": 0, "walking 0.6": 0, "terrain-2M 0.4": 0, "terrain-2M 0.6": 0, **run_policies}
        else: shown_policies = {"walking 0.4": 0, "walking 0.6": 0, **run_policies}
    else: shown_policies = POLICIES
    order = {k: i for i, k in enumerate(LEVELS)}; rows.sort(key=lambda r: (list(ALL).index(r["policy"]), order[r["kind"]], r["level"]))
    with open(out / f"baseline{suffix}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    md = ["| terrain | level | " + " | ".join(shown_policies) + " |", "|---|---|" + "---|" * len(shown_policies)]
    unit = {"rough": "cm", "slope_up": "deg", "slope_down": "deg", "stairs_up": "cm", "stairs_down": "cm"}
    for k, lvls in LEVELS.items():
        for lv in lvls:
            cells = []
            for p in shown_policies:
                r = next(x for x in rows if x["policy"] == p and x["kind"] == k and abs(float(x["level"]) - lv) < 1e-9); cells.append(f"{100 * r['success'] / n_ep:.0f}% ({r['fell']}f {r['stuck']}s {r['left_course']}l)")
            shown = lv * 100 if unit[k] == "cm" else lv; md.append(f"| {k} | {shown:g} {unit[k]} | " + " | ".join(cells) + " |")
    md.append(f"\nCells: success rate over {n_ep} episodes (f = fell, s = stuck, l = left the course). Success = x reaches 6 m within 20 s.")
    (out / f"baseline{suffix}.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 5, figsize=(20, 3.6))
    for ax, (k, lvls) in zip(axs, LEVELS.items()):
        for p in shown_policies:
            ax.plot([lv * 100 if unit[k] == "cm" else lv for lv in lvls], [100 * next(x for x in rows if x["policy"] == p and x["kind"] == k and abs(float(x["level"]) - lv) < 1e-9)["success"] / n_ep for lv in lvls], marker="o", label=p, linestyle="--" if p.startswith("walking") else "-")
        ax.set_title(k); ax.set_xlabel(f"level ({unit[k]})"); ax.set_ylim(-3, 103); ax.grid(alpha=0.3)
    axs[0].set_ylabel("success rate (%)"); axs[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(out / f"baseline{suffix}.png", dpi=130)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
