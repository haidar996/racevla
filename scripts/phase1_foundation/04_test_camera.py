"""Check headless RGB rendering works (needed for the Phase 10 vision pipeline)."""
import os, sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco, numpy as np, imageio.v2 as imageio
from racevla.robots.go1 import ROOT, load_model, reset_home

model, data = load_model(); reset_home(model, data)
for _ in range(500): mujoco.mj_step(model, data)
try:
    r = mujoco.Renderer(model, 240, 320); r.update_scene(data); img = r.render()
except Exception as e:
    print(f"MUJOCO_GL={os.environ['MUJOCO_GL']} failed: {e}\nTry: MUJOCO_GL=osmesa (needs libosmesa6) or run with a display."); sys.exit(1)
out = ROOT / "results" / "plots" / "phase1_render_test.png"; imageio.imwrite(out, img)
print("render ok", img.shape, "mean pixel", img.mean().round(1), "->", out)
