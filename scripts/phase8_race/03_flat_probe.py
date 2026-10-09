"""Phase 8 diagnosis: what does the terrain classifier say about a perfectly FLAT floor, standing upright, depending on where the robot is? Hypothesis: on the 10 m training course the camera often sees the END of the height field (a drop to nothing); on the 46 m race course it never does, so the network may have learned 'edge in view = flat'.
Part A: the 10 m training course, flat floor (flat = ('rough', 0.0)), x = -0.5 ... 8.5. Part B: the 46 m race course, flat floor at ground height 0, then a flat floor RAISED by 0.4 m, x = 0 ... 40. Gravity input = (0, 0, -1) (upright). Usage: python 03_flat_probe.py
RESULT: NOT INFORMATIVE. A perfectly level standing pose is not what the classifier was trained on (it was trained on a moving robot), so it says "stairs down" almost everywhere on flat floor. The informative probe is 04_flat_drive_probe.py (the real supervisor driving). Kept as a record."""
import os, sys, json, pathlib; os.environ.setdefault("MUJOCO_GL", "egl"); ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, mujoco
from racevla.skills.body import TerrainSwitchBody, RaceSwitchBody
from racevla.vision.camera import OnboardCamera
from racevla.vision.classifier import TerrainClassifier, CLASSES
from racevla.envs.terrain import NROW

table = json.load(open(ROOT / "models/running_gait_table.json")); clf = TerrainClassifier(ROOT / "models/terrain_classifier_cnn_round3.pt", use_gravity=True)


def probe(body, options, xs):
    cam = OnboardCamera(body.model); out = []
    for x in xs:
        body.reset(seed=0, options=options); d = body.data; d.qpos[0], d.qpos[1], d.qpos[3:7], d.qpos[7:], d.qvel[:] = x, 0.0, [1, 0, 0, 0], body.home_joints, 0.0; d.qpos[2] = body.home_qpos[2]; mujoco.mj_forward(body.model, d)
        d.qpos[2] += float(body.ground_height_at(x, 0.0)) + 0.01 - body._lowest_foot_bottom(); mujoco.mj_forward(body.model, d)
        depth = cam.render(d); out.append((x, CLASSES[clf.predict(depth, np.array([0.0, 0.0, -1.0], np.float32))], float(depth.min()), float((depth >= 4.99).mean())))
    cam.close(); return out


def show(title, res):
    print(title); print("  x     says         nearest-depth  share of pixels with no return (sky / void)")
    for x, c, dmin, void in res: print(f"  {x:5.1f}  {c:11s}  {dmin:5.2f}          {100 * void:3.0f} %")


show("A: 10 m training course, flat floor", probe(TerrainSwitchBody(table), {"terrain": ("rough", 0.0), "terrain_seed": 0, "yaw": 0.0}, np.arange(-0.5, 8.6, 1.0)))
b = RaceSwitchBody(table); xs = np.arange(0.0, 40.1, 4.0)
show("B1: 46 m race course, flat floor at height 0", probe(b, {"height_field": (np.zeros((NROW, b.ncol)), 0.0, ("race", 0.0)), "yaw": 0.0}, xs))
show("B2: 46 m race course, flat floor raised by 0.4 m", probe(b, {"height_field": (np.full((NROW, b.ncol), 0.4), 0.4, ("race", 0.0)), "yaw": 0.0}, xs))
