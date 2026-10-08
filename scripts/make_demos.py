"""Render the demo GIFs shown in the README (headless, EGL). Usage: python scripts/make_demos.py [standing walking running terrain stepover]  (default: all)
Each GIF is a side-on tracking camera with a caption (skill, command, speed) drawn on top. Output: docs/media/<skill>.gif"""
import os, sys, json, pathlib; ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco, imageio.v2 as imageio, torch; torch.set_num_threads(2)
from PIL import Image, ImageDraw
from stable_baselines3 import PPO, TD3
from racevla.envs.wrappers import FixedObsNormalize

W, H, OUT = 560, 350, ROOT / "docs" / "media"; OUT.mkdir(parents=True, exist_ok=True)


class Recorder:
    def __init__(self, raw, dist=2.4, elev=-10, azim=90):
        self.raw = raw; self.r = mujoco.Renderer(raw.model, H, W); self.cam = mujoco.MjvCamera(); self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = 1; self.cam.distance, self.cam.elevation, self.cam.azimuth = dist, elev, azim; self.frames = []; self.n = 0
    def terrain_changed(self):
        mujoco.mjr_uploadHField(self.raw.model, self.r._mjr_context, self.raw.hf_id)
    def grab(self, title, line2="", every=2):
        self.n += 1
        if self.n % every: return
        self.r.update_scene(self.raw.data, camera=self.cam); img = Image.fromarray(self.r.render()); d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 46], fill=(20, 20, 28)); d.text((10, 6), title, fill=(255, 255, 255)); d.text((10, 26), line2, fill=(160, 220, 255)); self.frames.append(np.asarray(img))
    def save(self, name, fps=25):
        p = OUT / f"{name}.gif"; ims = [Image.fromarray(f).quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in self.frames]
        ims[0].save(p, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0, optimize=True); print(f"{p}  {len(self.frames)} frames  {p.stat().st_size / 1e6:.1f} MB", flush=True)


def standing():
    from racevla.envs.go1_standing import Go1StandingEnv, MAX_EPISODE_STEPS
    model = TD3.load(ROOT / "models/standing_policy_td3_seed0.zip", device="cpu"); raw = Go1StandingEnv(push=True); env = FixedObsNormalize(raw); rec = Recorder(raw, 1.7, -8)
    obs, _ = env.reset(seed=1000); kicks = 0
    for t in range(450):
        act = np.zeros(12, np.float32) if t < 10 else model.predict(obs, deterministic=True)[0]          # TD3 model: 10 passive warm-up steps (models/standing_policy_final.md)
        obs, r, term, trunc, info = env.step(act); kicks += int(info["push_kick"] > 0)
        rec.grab("Skill 1: standing balance + push recovery (TD3)", f"random pushes up to 1.2 m/s   pushes so far: {kicks}   t = {t * 0.02:.1f} s", every=3)
        if term or trunc: break
    rec.save("standing")


def walking():
    from racevla.envs.go1_walking_sine import Go1WalkingSineEnv
    model = PPO.load(ROOT / "models/walking_policy_sine_seed1.zip", device="cpu"); raw = Go1WalkingSineEnv(push=False); env = FixedObsNormalize(raw); raw._sample_command = lambda: None; rec = Recorder(raw, 1.8)
    script = [(0.4, 0.0, "walk forward 0.4 m/s"), (0.6, 0.0, "walk forward 0.6 m/s"), (0.4, 0.4, "forward 0.4 + turn left 0.4 rad/s"), (0.0, 0.5, "turn in place 0.5 rad/s")]
    obs, _ = env.reset(seed=3000, options={"yaw": 0.0})
    def setc(o, c): raw.command = np.array(c, float); o[-4:-2] = raw.command; o[-2:] = raw._clock()
    for vx, wz, label in script:
        for t in range(120):
            setc(obs, (vx, wz)); obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); setc(obs, (vx, wz))
            rec.grab("Skill 2: velocity-command walking (sine gait reference + PPO residual)", f"command: {label}   actual vx {info['vx']:.2f} m/s", every=3)
    rec.save("walking")


def running():
    from racevla.envs.go1_running_sine import Go1RunningSineEnv
    table = json.load(open(ROOT / "models/running_gait_table.json")); model = PPO.load(ROOT / "models/running_policy_2p5_seed0.zip", device="cpu")
    raw = Go1RunningSineEnv(push=False, gait={"table": table}); env = FixedObsNormalize(raw); raw._sample_command = lambda: None; rec = Recorder(raw, 2.3, -8)
    script = [(1.0, "jog 1.0 m/s"), (1.5, "run 1.5 m/s"), (2.0, "run 2.0 m/s"), (2.5, "run 2.5 m/s")]; obs, _ = env.reset(seed=3000, options={"yaw": 0.0})
    def setc(o, c): raw.command = np.array(c, float); o[-4:-2] = raw.command; o[-2:] = raw._clock()
    for vx, label in script:
        for t in range(120):
            setc(obs, (vx, 0.0)); obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); setc(obs, (vx, 0.0)); air = raw._foot_contacts().sum() == 0
            rec.grab("Skill 3: running up to 2.5 m/s with flight phases", f"command: {label}   actual {info['vx']:.2f} m/s" + ("   FLIGHT (all feet off the floor)" if air else ""), every=3)
            if term: break
    rec.save("running")


def terrain():
    from racevla.envs.go1_terrain_walk import Go1TerrainWalkEnv
    model = PPO.load(ROOT / "models/terrain_policy_seed2.zip", device="cpu"); raw = Go1TerrainWalkEnv(); env = FixedObsNormalize(raw); raw._sample_command = lambda: None; rec = Recorder(raw, 1.8, -12)
    cases = [("stairs_up", 0.04, "stairs UP, 4 cm steps"), ("stairs_down", 0.06, "stairs DOWN, 6 cm steps"), ("slope_up", 15, "slope UP, 15 degrees"), ("rough", 0.10, "rough ground, bumps up to 10 cm")]
    for i, (kind, h, label) in enumerate(cases):
        raw.set_terrain(kind, h, i); rec.terrain_changed(); obs, _ = env.reset(seed=i, options={"terrain": (kind, h), "terrain_seed": i, "yaw": 0.0})
        for t in range(700):
            raw.command = np.array([0.6, raw.command[1]]); obs[-4:-2] = raw.command; obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0]); raw.command[0] = 0.6; obs[-4:-2] = raw.command
            rec.grab("Skill 4: blind terrain walking (no camera, proprioception only)", f"{label}   x = {raw.data.qpos[0]:.1f} m (goal 6 m)", every=7)
            if term or raw.data.qpos[0] >= 6.0: break
        print(f"  {label}: ended x = {raw.data.qpos[0]:.2f} m, fell = {bool(term)}", flush=True)
    rec.save("terrain")


def stepover():
    from racevla.envs.go1_stepover import Go1StepOverEnv
    model = PPO.load(ROOT / "models/hurdle_stepover_seed0.zip", device="cpu"); raw = Go1StepOverEnv(); env = FixedObsNormalize(raw); rec = Recorder(raw, 2.2, -8)
    for i, h in enumerate((0.05, 0.07)):
        raw.set_terrain("hurdle", h, 2000 + i); rec.terrain_changed(); obs, _ = env.reset(seed=2000 + i, options={"terrain": ("hurdle", h), "terrain_seed": 2000 + i, "yaw": 0.0})
        for t in range(300):
            obs, r, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
            rec.grab("Skill 5: step over a bar while running (terrain scan + PPO), slow motion x2", f"bar height {h * 100:g} cm at x = 5 m   x = {raw.data.qpos[0]:.1f} m   speed {info['vx']:.2f} m/s", every=3)
            if term or trunc or raw.data.qpos[0] > 6.6: break
        print(f"  bar {h*100:g} cm: ended x = {raw.data.qpos[0]:.2f} m, hit/fell = {bool(term)}", flush=True)
    rec.save("stepover")


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["standing", "walking", "running", "terrain", "stepover"]): print("==", name, flush=True); globals()[name]()
