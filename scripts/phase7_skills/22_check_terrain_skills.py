"""Stage 4a: do the terrain and step-over skills, driven through the Supervisor on the shared TerrainSwitchBody, reproduce their model cards? Usage: python 22_check_terrain_skills.py [--episodes 20]
Terrain skill (models/terrain_policy_seed2.zip): fixed forward command 0.4 / 0.6 m/s + heading hold, terrain seed = episode seed 0..N-1, success = x reaches 6 m within 20 s (the harness of scripts/phase6_terrain/02_baseline_policies.py).
Step-over skill (models/hurdle_stepover_seed0.zip): command ~ U(1.3, 1.7) m/s from rest, seeds 2000.., success = 2 s after passing the bar (x > 5.45 m) upright (tilt < 35 deg) at >= 50 % of the command, no hit, within 450 steps (the rule of Go1HurdleEnv)."""
import sys, pathlib, argparse, multiprocessing as mp; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np

TERRAIN_CELLS = [("rough", 0.06), ("rough", 0.08), ("rough", 0.10), ("slope_up", 15), ("slope_down", 15), ("stairs_up", 0.04), ("stairs_down", 0.06)]
STEP_CELLS = [("rough", 0.0), ("hurdle", 0.03), ("hurdle", 0.05), ("hurdle", 0.07), ("hurdle", 0.10)]
CARD_TERRAIN = {("rough", 0.06): "100/100", ("rough", 0.08): "95/100", ("rough", 0.10): "90/80", ("slope_up", 15): "100/100", ("slope_down", 15): "100/100", ("stairs_up", 0.04): "100/100", ("stairs_down", 0.06): "100/100"}
CARD_STEP = {("rough", 0.0): "100", ("hurdle", 0.03): "100", ("hurdle", 0.05): "90", ("hurdle", 0.07): "80", ("hurdle", 0.10): "35"}
X_GOAL, CROSSED_X, X_EDGE = 6.0, 5.45, 8.95


def run(job):
    skill, kind, level, speed, n = job
    import torch; torch.set_num_threads(1)
    from racevla.skills.library import Supervisor
    sup = Supervisor(terrain=True); body = sup.body; ok = fell = 0
    for i in range(n):
        if skill == "terrain":
            sup.reset(i, "terrain", [speed, 0.0], warmup=0, options={"terrain": (kind, level), "terrain_seed": i, "yaw": 0.0})
            for t in range(1000):
                term, info = sup.step()
                if term or body.data.qpos[0] >= X_GOAL: break
            ok += int((not term) and body.data.qpos[0] >= X_GOAL); fell += int(term)
        else:
            cmd = float(np.random.default_rng(2000 + i).uniform(1.3, 1.7)); sup.reset(2000 + i, "stepover", [cmd, 0.0], warmup=0, options={"terrain": (kind, level), "terrain_seed": 2000 + i, "yaw": 0.0})
            crossed = None; res = False; dead = False
            for t in range(450):
                term, info = sup.step(); x = float(body.data.qpos[0])
                if (not term) and body.level_h > 0 and body.hit_hurdle(): dead = True; break
                if term: dead = True; break
                if crossed is None and x > CROSSED_X: crossed = sup.steps
                if crossed is not None and (sup.steps - crossed >= 100 or x >= X_EDGE):
                    R = body.data.xmat[1].reshape(3, 3); v = float((R.T @ body.data.qvel[:3])[0]); res = body.tilt_deg() < 35.0 and v >= 0.5 * cmd; break
            ok += int(res and not dead); fell += int(dead)
    return dict(skill=skill, kind=kind, level=level, speed=speed, ok=ok, fell=fell, n=n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=20); a = ap.parse_args(); n = a.episodes
    jobs = [("terrain", k, l, v, n) for k, l in TERRAIN_CELLS for v in (0.4, 0.6)] + [("stepover", k, l, 0.0, n) for k, l in STEP_CELLS]
    with mp.Pool(4, maxtasksperchild=1) as pool: rows = pool.map(run, jobs, chunksize=1)
    print(f"terrain skill, success % of {n} (0.4 / 0.6 m/s)   [model card in brackets]")
    for k, l in TERRAIN_CELLS:
        r = {x["speed"]: x for x in rows if x["skill"] == "terrain" and x["kind"] == k and x["level"] == l}; print(f"  {k:12s} {l:5}: {100 * r[0.4]['ok'] / n:.0f} / {100 * r[0.6]['ok'] / n:.0f}   [{CARD_TERRAIN[(k, l)]}]   falls {r[0.4]['fell']} / {r[0.6]['fell']}")
    print(f"step-over skill, success % of {n}   [model card in brackets]")
    for k, l in STEP_CELLS:
        r = next(x for x in rows if x["skill"] == "stepover" and x["kind"] == k and x["level"] == l); print(f"  {'empty' if k == 'rough' else f'{l * 100:g} cm':6s}: {100 * r['ok'] / n:.0f}   [{CARD_STEP[(k, l)]}]   hit/fell {r['fell']}")
