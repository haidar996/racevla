"""Pick the best checkpoint of a run on a LARGE held-out seed set instead of trusting SB3's 30-episode 'best_model.zip'.
1) evaluates every <run>/checkpoints/model_<steps>.zip (saved by 04_train_sac.py at each evaluation) on the SELECTION seeds,
2) picks the one with the fewest falls (ties: higher return), 3) re-tests that winner on separate CONFIRMATION seeds, so the number you report is
not inflated by having been the best of ~20 candidates. Writes <run>/checkpoint_selection.csv and copies the winner to <run>/selected_model.zip.
Assumes policies trained with FixedObsNormalize (all runs since the seed sweep). Failures before step 100 are 'start-up', later ones 'push'.
Usage: python 06_select_checkpoint.py outputs/<run> [--algo sac|ppo|td3] [--select-n 100] [--confirm-n 200] [--min-step 200000]"""
import argparse, re, shutil, sys, pathlib, multiprocessing as mp; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np

STARTUP_STEPS = 100    # first push can happen at step 100 at the earliest (50 grace + 50 minimum interval)


def evaluate(job):
    """One checkpoint on a list of seeds -> (path, per-episode lengths, returns). Runs in its own process."""
    algo, path, seeds = job
    import torch; torch.set_num_threads(1)
    from stable_baselines3 import PPO, SAC, TD3
    from racevla.envs.go1_standing import Go1StandingEnv, MAX_EPISODE_STEPS
    from racevla.envs.wrappers import FixedObsNormalize
    model = {"ppo": PPO, "sac": SAC, "td3": TD3}[algo].load(path, device="cpu")
    env = FixedObsNormalize(Go1StandingEnv(push=True)); lengths, returns = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed)); total = 0.0
        for t in range(1, MAX_EPISODE_STEPS + 1):
            obs, r, term, trunc, _ = env.step(model.predict(obs, deterministic=True)[0]); total += r
            if term or trunc: break
        lengths.append(t); returns.append(total)
    return path, np.array(lengths), np.array(returns)


def stats(lengths, returns):
    fell = lengths < 1000
    return {"falls": int(fell.sum()), "startup": int((fell & (lengths < STARTUP_STEPS)).sum()), "push": int((fell & (lengths >= STARTUP_STEPS)).sum()), "ret": float(returns.mean())}


def run_jobs(pool, algo, paths, seeds):
    out = {}
    for path, L, R in pool.imap_unordered(evaluate, [(algo, str(p), seeds) for p in paths]): out[pathlib.Path(path)] = stats(L, R)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); ap.add_argument("--algo", default="sac", choices=["sac", "ppo", "td3"])
    ap.add_argument("--select-n", type=int, default=100); ap.add_argument("--select-first-seed", type=int, default=2000)
    ap.add_argument("--confirm-n", type=int, default=200); ap.add_argument("--confirm-first-seed", type=int, default=4000)
    ap.add_argument("--min-step", type=int, default=200_000, help="skip earlier checkpoints (they are rarely the best and cost time)")
    ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
    run = pathlib.Path(a.run_dir); run = run if run.is_absolute() else ROOT / run
    sel_seeds = np.arange(a.select_first_seed, a.select_first_seed + a.select_n); con_seeds = np.arange(a.confirm_first_seed, a.confirm_first_seed + a.confirm_n)
    if set(sel_seeds) & set(con_seeds): sys.exit("selection and confirmation seeds overlap -- pick different --confirm-first-seed")
    step_of = lambda p: int(re.search(r"model_(\d+)\.zip", p.name).group(1))
    paths = sorted((p for p in (run / "checkpoints").glob("model_*.zip") if step_of(p) >= a.min_step), key=step_of)
    if not paths: sys.exit(f"no checkpoints >= step {a.min_step} in {run / 'checkpoints'} (runs before the checkpoint-saving change only have best/latest)")

    with mp.get_context("spawn").Pool(min(a.workers, len(paths))) as pool:
        print(f"Evaluating {len(paths)} checkpoints on selection seeds {sel_seeds[0]}..{sel_seeds[-1]} ...", flush=True)
        table = run_jobs(pool, a.algo, paths, sel_seeds)
        winner = min(table, key=lambda p: (table[p]["falls"], -table[p]["ret"]))
        print(f"Re-testing winner {winner.name} on fresh confirmation seeds {con_seeds[0]}..{con_seeds[-1]} ...", flush=True)
        confirm = run_jobs(pool, a.algo, [winner], con_seeds)[winner]

    ev = run / "evaluations.npz"; sb3_step = int(np.load(ev)["timesteps"][int(np.load(ev)["results"].mean(1).argmax())]) if ev.exists() else None
    print(f"\nSelection seeds, n={a.select_n}:  (SB3's own 30-episode pick = step {sb3_step})")
    with open(run / "checkpoint_selection.csv", "w") as f:
        f.write("steps,falls,startup_falls,push_falls,mean_return,n_episodes\n")
        for p in paths:
            s = table[p]; f.write(f"{step_of(p)},{s['falls']},{s['startup']},{s['push']},{s['ret']:.1f},{a.select_n}\n")
            print(f"  {step_of(p):>8d}  falls {s['falls']:3d}/{a.select_n} ({100 * s['falls'] / a.select_n:4.1f}%)  start-up {s['startup']:3d}  push {s['push']:3d}  return {s['ret']:6.1f}"
                  + ("   <-- selected" if p == winner else "") + ("   <-- SB3 best_model" if step_of(p) == sb3_step else ""))
    shutil.copy(winner, run / "selected_model.zip")
    c = confirm; print(f"\nSELECTED: step {step_of(winner)} -> copied to {run / 'selected_model.zip'}")
    print(f"Honest estimate on fresh seeds {con_seeds[0]}..{con_seeds[-1]}: falls {c['falls']}/{a.confirm_n} ({100 * c['falls'] / a.confirm_n:.1f}%) | start-up {c['startup']} | push {c['push']} | return {c['ret']:.1f}")
    print(f"(on the selection seeds it scored {100 * table[winner]['falls'] / a.select_n:.1f}% -- expect the honest number to be a bit worse, that gap is the selection bias)")
