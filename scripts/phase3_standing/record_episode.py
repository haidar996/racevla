"""Record a push episode with a trained policy to a GIF and a few key PNG stills (just before / after each push).
Usage: python record_episode.py <run_dir> [checkpoint file] [--seed N] [--raw] [--algo ppo|sac|td3]"""
import os, sys, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, numpy as np, imageio.v2 as imageio
from stable_baselines3 import PPO, SAC, TD3
from racevla.envs.go1_standing import Go1StandingEnv, MAX_EPISODE_STEPS
from racevla.envs.wrappers import FixedObsNormalize

ALGO_CLASSES = {"ppo": PPO, "sac": SAC, "td3": TD3}
args = [a for a in sys.argv[1:] if not a.startswith("--")]
run_dir = ROOT / args[0]; ckpt = args[1] if len(args) > 1 else "best_model.zip"
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1000
raw = "--raw" in sys.argv
algo_name = sys.argv[sys.argv.index("--algo") + 1] if "--algo" in sys.argv else "ppo"
out_dir = ROOT / "outputs" / "report"; out_dir.mkdir(parents=True, exist_ok=True)

model = ALGO_CLASSES[algo_name].load(run_dir / ckpt, device="cpu")
raw_env = Go1StandingEnv(push=True)                       # kept unwrapped: the renderer needs its .model/.data directly
env = raw_env if raw else FixedObsNormalize(raw_env)       # what the policy actually sees
print(f"observation mode: {'RAW' if raw else 'normalized'}")
renderer = mujoco.Renderer(raw_env.model, 480, 640)
cam = mujoco.MjvCamera(); cam.fixedcamid = mujoco.mj_name2id(raw_env.model, mujoco.mjtObj.mjOBJ_CAMERA, "tracking")
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

frames, push_frames = [], []
obs, _ = env.reset(seed=seed)
for t in range(MAX_EPISODE_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, r, term, trunc, info = env.step(action)
    if t % 2 == 0 or info["push_kick"] > 0:               # ~25 fps GIF (every other 50 Hz step), plus every push frame
        renderer.update_scene(raw_env.data, camera=cam)
        frames.append(renderer.render().copy())
    if info["push_kick"] > 0:
        push_frames.append((t, info["push_kick"], len(frames) - 1))
    if term or trunc:
        reason = info.get("termination_reason", "survived (time limit)")
        break
else:
    reason = "survived (time limit)"

gif_path = out_dir / f"episode_seed{seed}.gif"
imageio.mimsave(gif_path, frames, fps=25, loop=0)
print(f"episode: seed={seed} steps={t+1} return≈ ended={reason}")
print(f"saved GIF: {gif_path}  ({len(frames)} frames)")

# a handful of PNG stills: just before / just after each push, plus the very end
saved = []
for step, kick, fi in push_frames[:6]:
    before = out_dir / f"seed{seed}_step{step}_before.png"; imageio.imwrite(before, frames[max(fi - 3, 0)])
    after  = out_dir / f"seed{seed}_step{step}_after.png";  imageio.imwrite(after,  frames[min(fi + 5, len(frames)-1)])
    saved += [str(before), str(after)]
end_png = out_dir / f"seed{seed}_end.png"; imageio.imwrite(end_png, frames[-1]); saved.append(str(end_png))
print("stills saved:", *saved, sep="\n  ")
