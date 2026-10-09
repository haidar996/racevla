"""Phase 10, V1: what does the onboard depth camera see? Usage: python 01_show_camera.py [--gif]
Contact sheet outputs/analysis/vision/camera_views.png: one row per terrain (flat, rough 8 cm, slope up 12 deg, slope down 12 deg, stairs up 8 cm, stairs down 8 cm, thin bar 10 cm); columns = a third-person view of the robot standing
in front of the patch, and what the depth camera sees from a far and a near distance (colour = distance in metres, 64 x 48 pixels, shown enlarged). With --gif: outputs/analysis/vision/camera_bar_run.gif, a run to a 7 cm bar and the step-over,
third-person view next to the depth image."""
import os, sys, json, pathlib; os.environ.setdefault("MUJOCO_GL", "egl"); ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from racevla.skills.body import TerrainSwitchBody
from racevla.vision.camera import OnboardCamera, refresh_hfield

OUT = ROOT / "outputs" / "analysis" / "vision"; OUT.mkdir(parents=True, exist_ok=True)
VIEWS = [("rough 8 cm", ("rough", 0.08), (-0.2, 0.5)), ("slope up 12 deg", ("slope_up", 12), (-0.2, 0.5)), ("slope down 12 deg", ("slope_down", 12), (-0.2, 0.5)),
         ("stairs up 8 cm", ("stairs_up", 0.08), (-0.2, 0.5)), ("stairs down 8 cm", ("stairs_down", 0.08), (-0.2, 0.5)), ("bar 10 cm", ("hurdle", 0.10), (3.6, 4.4))]


def third_person(renderer, body, dist=3.4, azimuth=60):
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE; x, y, z = body.data.qpos[:3]
    cam.lookat[:] = [x + 0.9, y, z]; cam.distance, cam.azimuth, cam.elevation = dist, azimuth, -30
    refresh_hfield(renderer, body.model); renderer.update_scene(body.data, cam); return renderer.render().copy()           # (re-upload: the terrain may have changed)


def place(body, terrain, x):
    """The robot standing upright in the home pose (no reset noise: a slanted horizon would hide the terrain), heading +x, feet on the ground at x."""
    body.reset(seed=0, options={"terrain": terrain, "terrain_seed": 0, "yaw": 0.0}); d = body.data
    d.qpos[0], d.qpos[1], d.qpos[3:7], d.qpos[7:], d.qvel[:] = x, 0.0, [1, 0, 0, 0], body.home_joints, 0.0; d.qpos[2] = body.home_qpos[2]; mujoco.mj_forward(body.model, d)
    d.qpos[2] += float(body.ground_height_at(x, 0.0)) + 0.01 - body._lowest_foot_bottom(); mujoco.mj_forward(body.model, d)


if __name__ == "__main__":
    body = TerrainSwitchBody(json.load(open(ROOT / "models/running_gait_table.json"))); cam = OnboardCamera(body.model); rgb = mujoco.Renderer(body.model, 360, 480)
    fig, ax = plt.subplots(len(VIEWS), 5, figsize=(18, 3.0 * len(VIEWS)))
    for r, (name, terrain, (xf, xn)) in enumerate(VIEWS):
        place(body, terrain, xf); ax[r, 0].imshow(third_person(rgb, body)); ax[r, 0].set_ylabel(name, fontsize=12, rotation=0, ha="right", va="center")
        for c, x in ((1, xf), (2, xn)):
            place(body, terrain, x); d = cam.render(body.data); im = ax[r, c].imshow(d, cmap="turbo_r", vmin=0.2, vmax=3.0, interpolation="nearest")
            ax[r, c].set_title(f"depth, robot at x = {x:+.1f} m" if r == 0 else "", fontsize=10)
        for c, x in ((3, xf), (4, xn)):                                                                  # same pose on a flat floor: where does the picture differ?
            place(body, ("rough", 0.0), x); flat = cam.render(body.data); place(body, terrain, x); diff = cam.render(body.data) - flat
            imd = ax[r, c].imshow(diff, cmap="seismic", vmin=-0.5, vmax=0.5, interpolation="nearest"); ax[r, c].set_title(("far view" if c == 3 else "near view") + " minus flat floor" if r == 0 else "", fontsize=10)
        ax[r, 0].set_title("third-person view" if r == 0 else "", fontsize=10)
        for a in ax[r]: a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=ax[:, 1:3], shrink=0.4, label="distance (m)", pad=0.02); fig.colorbar(imd, ax=ax[:, 3:], shrink=0.4, label="difference (m): blue = closer than a flat floor (ground rises), red = farther (ground drops)", pad=0.02); fig.savefig(OUT / "camera_views.png", dpi=110, bbox_inches="tight"); print("saved", OUT / "camera_views.png")
    if "--gif" in sys.argv:
        import imageio
        from racevla.skills.library import Supervisor
        from racevla.skills.rules import RuleSupervisor
        sup = Supervisor(terrain=True); rs = RuleSupervisor(sup, hold_heading=True); b = sup.body; cam2 = OnboardCamera(b.model); rgb2 = mujoco.Renderer(b.model, 360, 480)
        rs.reset(3002, options={"terrain": ("hurdle", 0.07), "terrain_seed": 3002, "yaw": 0.0}, start_x=-0.5); rs.set_target(1.5, 0.0); frames = []
        for t in range(420):
            x = float(b.data.qpos[0]); rs.set_hint("bar" if 3.0 <= x <= 6.2 else None); term, info = rs.step()
            if t % 3 == 0:
                d = cam2.render(b.data); col = (plt.get_cmap("turbo_r")(np.clip((d - 0.2) / 2.8, 0, 1))[:, :, :3] * 255).astype(np.uint8)
                col = np.kron(col, np.ones((7, 7, 1), np.uint8)); left = third_person(rgb2, b); h = min(left.shape[0], col.shape[0]); frames.append(np.concatenate([left[:h], col[:h]], axis=1))
            if term or x > 7.0: break
        imageio.mimsave(OUT / "camera_bar_run.gif", frames, duration=0.06, loop=0); print("saved gif", len(frames), "frames; skill at the end:", sup.active.name)
