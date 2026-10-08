"""Step 15: watch a calm episode (zero action) and a thrashing one (random actions) in the MuJoCo viewer,
then save a plot of the reward and its terms. Needs a display; use --no-viewer to only make the plot."""
import sys, time, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import matplotlib; matplotlib.use("Agg")                        # draw to a file, no plot window
import matplotlib.pyplot as plt
import numpy as np
from racevla.envs.go1_standing import Go1StandingEnv

STEPS = 300                                                     # policy steps per episode = 6 s
KEYS = ["reward", "r_upright", "r_height", "r_pose", "r_discount"]
env = Go1StandingEnv()
viewer = None
if "--no-viewer" not in sys.argv:
    import mujoco.viewer
    viewer = mujoco.viewer.launch_passive(env.model, env.data)  # window shares env.data, so it shows every step


def run_episode(name, policy, seed):
    """Run one episode, return {key: array over steps} and the termination reason (None if it survived)."""
    rng = np.random.default_rng(seed)
    env.reset(seed=seed); log = {k: [] for k in KEYS}; reason = None
    print(f"running: {name}")
    for _ in range(STEPS):
        if viewer is not None and not viewer.is_running(): break
        t0 = time.time()
        _, r, term, trunc, info = env.step(policy(rng))
        for k in KEYS: log[k].append(r if k == "reward" else info[k])
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))     # 0.02 s = one policy step, so it plays in real time
        if term or trunc:
            reason = info.get("termination_reason"); break
    return {k: np.array(v) for k, v in log.items()}, reason


episodes = {"calm (zero action)": run_episode("calm (zero action)", lambda rng: np.zeros(12), seed=0),
            "thrashing (random actions)": run_episode("thrashing (random actions)", lambda rng: rng.uniform(-1, 1, 12), seed=0)}
if viewer is not None: viewer.close()

fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey="row")
for col, (name, (log, reason)) in enumerate(episodes.items()):
    t = np.arange(len(log["reward"])) * 0.02
    axes[0, col].plot(t, log["reward"], "k", lw=2, label="reward"); axes[0, col].plot(t, log["r_discount"], "r--", label="discount")
    axes[0, col].set_title(f"{name}" + (f"  (ended: {reason})" if reason else "")); axes[0, col].set_ylim(0, 1.05)
    for k in ("r_upright", "r_height", "r_pose"): axes[1, col].plot(t, log[k], label=k[2:])
    axes[1, col].set_xlabel("time (s)"); axes[1, col].set_ylim(0, 1.05)
axes[0, 0].legend(); axes[1, 0].legend(); axes[0, 0].set_ylabel("reward"); axes[1, 0].set_ylabel("kernel value (1 = perfect)")
fig.tight_layout()
out = ROOT / "outputs" / "reward_terms.png"; out.parent.mkdir(exist_ok=True); fig.savefig(out, dpi=120)
for name, (log, reason) in episodes.items():
    print(f"{name:28s} steps={len(log['reward']):3d}  mean reward={log['reward'].mean():.3f}  mean discount={log['r_discount'].mean():.3f}  ended={reason}")
print("saved", out)
