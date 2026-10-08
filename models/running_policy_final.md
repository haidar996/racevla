# Final running policy (Phase 5, speed-dependent option-B reference with push-off) -- written 2026-10-03

**Use `running_policy_2p5_seed0.zip`** (commands up to 2.5 m/s) together with `running_gait_table.json` (the gait table is part of the policy: the policy was trained with it and must be run with it).
Backup: `running_policy_2p5_seed1_backup.zip` (more flight, smaller speed error, but 3/50 falls). For the 0-2.0 m/s range only: `running_policy_2p0_seed1.zip` (stage 2, 0/50 falls).
Training: stage 1 (commands 0-1.5 m/s, 1M steps) from the walking policy of the same seed (walk4_sine_1M_seed<s>), stage 2 (0-2.0, 600k), stage 3 (0-2.5, 600k), PPO as in walking, 3 seeds, no pushes. Scripts: scripts/phase5_running/.

## How to run it
- Environment: `Go1RunningSineEnv(push=False, vx_range=(0, 2.5), wz_range=(-0.5, 0.5), gait={"table": json.load(open("running_gait_table.json"))})` (racevla/envs/go1_running_sine.py) wrapped in `FixedObsNormalize`.
- Observation 49 = 45 robot + 2 command (vx, wz) + 2 clock (sin, cos; 0 when told to stand). Action 12 in [-1,1], residual scale 0.25 rad on top of the reference leg motion; PD KP 100 / KD 0.5; 50 Hz.
- Gait table (blended linearly between speeds): speed 0 / 0.6 / 1.0 / 1.5 / 2.0 / 2.5 m/s -> step rate 2.5 / 2.5 / 3.0 / 3.0 / 3.5 / 4.0 Hz, swing share 0.4 / 0.4 / 0.5 / 0.5 / 0.5 / 0.55, lift 8 cm, stance push-off 0 / 0 / 2 / 4 / 4 / 4 cm. The 2.5 row is extrapolated (the reference alone could not track 2.5).
- Load with `PPO.load`, `predict(obs, deterministic=True)`. Viewer: `python scripts/phase5_running/06_view_running.py models/running_policy_2p5_seed0.zip` (it reads the gait table from outputs/analysis/running/gait_table.json).

## Results (50 fresh random-command episodes, 1000 steps, seeds 5000-5049, commands 0-2.5 m/s, no pushes)
| Model | Falls | Fwd error (all) | Speed reached at cmd>=1 | Flight share at cmd>=1 | Slip |
|---|---|---|---|---|---|
| 2p5 seed 0 (primary) | 1/50 | 0.122 | 94 % | 28 % | 0.27 |
| 2p5 seed 1 (backup) | 3/50 | 0.107 | 96 % | 35 % | 0.23 |
| 2p5 seed 2 | 4/50 | 0.120 | 95 % | 34 % | 0.25 |
Steady commands (4 episodes each, speed reached / flight share): seed 0: 1.5 -> 1.47 / 34 %, 2.0 -> 1.99 / 31 %, 2.5 -> 2.48 / 34 %, 0 falls; seed 1: 1.50 / 36 %, 2.04 / 42 %, 2.46 / 42 %, 0 falls.
Full table of all 9 runs: outputs/analysis/running_comparison.md. "Flight" = all four feet off the floor in the simulator's contact test.

## Known limits
- Falls in 50 episodes are 1, 3 and 4 for the three 2.5 seeds: with n = 50 these are not clearly different from each other.
- Not tested with pushes, uneven ground, speeds above 2.5 m/s, backward running or turn rates above 0.5 rad/s. The steady-speed tests are only 4 episodes each.
- The 2.5 m/s row of the gait table is extrapolated; the learned residual does the rest.
- Stage-2 seed 2 fell in 2 of 4 steady 2.0 m/s episodes (1/50 in the random check).
- Flight is a by-product of running, not yet a trained jump: take-off from standing, obstacles and landing are not covered.
