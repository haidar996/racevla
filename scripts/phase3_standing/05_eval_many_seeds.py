"""Held-out evaluation: every policy on the same N fresh seeds (default 2000..2199, never used for checkpoint selection, which used 1000-1029).
Splits falls into START-UP (before the first possible push, step < 100) and PUSH (after) so the two failure causes can be told apart.
Usage: python 05_eval_many_seeds.py [--n 200] [--first-seed 2000]      Writes outputs/eval_heldout/<name>.npz and prints a table."""
import argparse, sys, pathlib, multiprocessing as mp; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np

# name -> (algo, path); algo "zero" = hold the home pose. All learned models here were trained with FixedObsNormalize.
POLICIES = {
    "zero_action":  ("zero", None),
    "ppo_best":     ("ppo", "outputs/ppo_1M_final_seed0/best_model.zip"),
    "sac_old_best": ("sac", "outputs/sac_push_1M_seed0/best_model.zip"),
    "sac_old_last": ("sac", "outputs/sac_push_1M_seed0/latest_model.zip"),
    "sac_v2_best":  ("sac", "outputs/sac_push_1M_v2_seed0/best_model.zip"),
    "sac_v2_last":  ("sac", "outputs/sac_push_1M_v2_seed0/latest_model.zip"),
    "sac_v3_best":  ("sac", "outputs/sac_push_1M_v3_ep250_seed0/best_model.zip"),
    "sac_v3_last":  ("sac", "outputs/sac_push_1M_v3_ep250_seed0/latest_model.zip"),
}
STARTUP_STEPS = 100    # first push can happen at step 100 at the earliest (50 grace + 50 minimum interval)


def evaluate(job):
    name, seeds = job
    import torch; torch.set_num_threads(1)
    from stable_baselines3 import PPO, SAC
    from racevla.envs.go1_standing import Go1StandingEnv, MAX_EPISODE_STEPS
    from racevla.envs.wrappers import FixedObsNormalize
    algo, path = POLICIES[name]
    model = None if algo == "zero" else (PPO if algo == "ppo" else SAC).load(ROOT / path, device="cpu")
    env = FixedObsNormalize(Go1StandingEnv(push=True))
    length, ret, reason, last_kick = [], [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed)); total, kick, term_reason = 0.0, 0.0, ""
        for t in range(1, MAX_EPISODE_STEPS + 1):
            a = np.zeros(12, np.float32) if model is None else model.predict(obs, deterministic=True)[0]
            obs, r, term, trunc, info = env.step(a); total += r
            if info["push_kick"] > 0: kick = info["push_kick"]
            if term: term_reason = info["termination_reason"]; break
            if trunc: break
        length.append(t); ret.append(total); reason.append(term_reason); last_kick.append(kick)
    out = ROOT / "outputs" / "eval_heldout"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}.npz", seeds=seeds, length=length, ret=ret, reason=reason, last_kick=last_kick)
    return name


def summarize(name, n):
    d = np.load(ROOT / "outputs" / "eval_heldout" / f"{name}.npz"); L = d["length"]; fell = L < 1000
    start = fell & (L < STARTUP_STEPS); push = fell & (L >= STARTUP_STEPS)
    kick = d["last_kick"][push]
    return (f"{name:13s} falls {fell.sum():3d}/{n} ({100 * fell.mean():4.1f}%) | start-up {start.sum():3d} | push {push.sum():3d} | "
            f"return {d['ret'].mean():6.1f} | mean push size at push-falls {kick.mean() if len(kick) else float('nan'):.2f} m/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200); ap.add_argument("--first-seed", type=int, default=2000)
    ap.add_argument("--only", nargs="*", help="subset of policy names"); a = ap.parse_args()
    seeds = np.arange(a.first_seed, a.first_seed + a.n); names = a.only or list(POLICIES)
    with mp.get_context("spawn").Pool(min(4, len(names))) as pool:
        for done in pool.imap_unordered(evaluate, [(nm, seeds) for nm in names]): print("finished", done, flush=True)
    print(f"\nHeld-out seeds {seeds[0]}..{seeds[-1]} (start-up = fell before step {STARTUP_STEPS})")
    for nm in names: print(summarize(nm, a.n))
