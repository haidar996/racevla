# Final terrain policy (Phase 6, Skill A: blind terrain walking) -- written 2026-10-07

**Use `terrain_policy_seed2.zip`** (PPO, 2M steps, the option-B walking policy retrained on mixed terrain). Backup: `terrain_policy_seed0_backup.zip` (same recipe, other seed; 100 % on stairs up 6 cm, but weaker on rough 10 cm at 0.6 m/s).
Seed 1 (`outputs/terr_2M_seed1`) was equally good in scope but is not packaged. The comparison of baseline and all three seeds is in `terrain_policy_comparison.md`.
Scope: blind (no height scan, no camera): rough ground up to 10 cm, slopes up and down to 15 deg, stairs up to 4 cm and stairs down to 6 cm (that is where the curriculum stops). NO pushes.

## How to run it
- Environment: `Go1TerrainWalkEnv(...)` from `racevla/envs/go1_terrain_walk.py` (terrain mixin + Go1WalkingSineEnv: same 49-D observation as the walking policy: 45 robot + 2 command + 2 clock), wrapped in `FixedObsNormalize`. Load with `PPO.load`, `predict(obs, deterministic=True)`.
- The terrain course (racevla/envs/terrain.py): `env.reset(seed=s, options={"terrain": (kind, level), "terrain_seed": s, "yaw": 0.0})`, kinds `rough` (m), `slope_up`, `slope_down` (deg), `stairs_up`, `stairs_down` (m). The course is 10 m x 3 m, flat lead-in for x < 1 m, success = x reaches 6 m (Go1TerrainWalkEnv ends the episode with a truncation then).
- Command: forward speed 0.4-0.8 m/s plus a HEADING HOLD: turn command = clip(-2 x yaw error, -0.5, 0.5) rad/s, refreshed every step and written into the observation (Go1TerrainWalkEnv does this itself; the evaluation harness scripts/phase6_terrain/02_baseline_policies.py does it in the script). Without a heading hold the robot drifts sideways off the 3 m wide course on any terrain.
- Viewer: `python scripts/phase6_terrain/03_view_terrain.py walking slope_up 15 0.6 --hold --model models/terrain_policy_seed2.zip` (kind, level, speed, --hold).

## Results (20 fresh terrain seeds per cell, forward command 0.4 / 0.6 m/s, 1000 steps = 20 s, success = x reaches 6 m)
Seed 2 (primary), success % at 0.4 / 0.6 m/s (the old walking policy in brackets):
| Terrain | Seed 2 | Old walking policy |
|---|---|---|
| rough 0-6 cm | 100 / 100 | 100 / 100 |
| rough 8 cm | 95 / 100 | 85 / 100 |
| rough 10 cm | 90 / 80 | 10 / 40 |
| slope up 3 deg | 100 / 100 | 100 / 100 |
| slope up 6, 9, 12 deg | 100 / 100 | 20 / 100, 0 / 100, 0 / 100 |
| slope up 15 deg | 100 / 100 | 0 / 15 |
| slope down 3-15 deg | 100 / 100 | 100 / 100 |
| stairs up 2, 4 cm | 100 / 100 | 100 / 100 |
| stairs down 2, 4, 6 cm | 100 / 100 | 100 / 100 |
Out of scope (not trained for; do NOT rely on): stairs up 6 cm 100 / 100, 8 cm 0 / 35, >= 10 cm 0 (blocked or falls); stairs down 8, 10 cm 100 / 100, 12 cm 95 / 60, 15 cm 45 / 15.
Average over the in-scope levels: seed 2 99 / 99 %, seed 0 100 / 99, seed 1 98 / 98, old policy 80 / 94. In-training check on 9 fixed terrains (5 episodes each): mean 1.00 at 2M steps.

## Training config
`scripts/phase6_terrain/05_train_terrain_ppo.py --run-name terr_2M_seed2 --seed 2 --steps 2000000 --eval-every 200000 --init outputs/walk4_sine_1M_seed2/checkpoints/model_1000000.zip`: PPO, 4 envs, n_steps 512, 10 epochs, clip 0.2, gae 0.95, lr 3e-4 constant, action noise reset to log_std -0.8, ~315 steps/s (91 min).
Terrain mix per episode: flat 20 %, rough 20 %, slope up 20 %, slope down 10 %, stairs up 15 %, stairs down 15 %; per-type curriculum (a level unlocks at >= 80 % success over >= 15 episodes at the top level); caps rough 10 cm, slopes 15 deg, stairs up 4 cm, stairs down 6 cm. All ladders were at their caps by about 650k steps. Reward = the walking reward (no terrain-specific terms).

## Known limits
- Only 0.4 and 0.6 m/s were evaluated (training commands 0.4-0.8 m/s); no running on terrain; no pushes; no backward walking; turning only through the heading hold.
- Rough 10 cm is the one in-scope cell that is not reliably 100 % (seeds: 100 / 60 / 90 % at 0.4 m/s, 65 / 65 / 80 % at 0.6 m/s).
- Stairs up of 8 cm and more are not solved (a blind policy cannot see the step); this needs a height scan (Skill B). Out-of-scope ability differs between seeds.
- Tested only on the generator in racevla/envs/terrain.py (same simulator, no friction / mass / motor randomization), simulation only.
