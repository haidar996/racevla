# Terrain policy, variant with a 12 cm reference foot lift (Phase 6, Skill A12) -- written 2026-10-08

**`terrain_policy_lift12_seed2.zip`** = the same blind terrain walking skill as `terrain_policy_seed2.zip` (see terrain_policy_final.md) but trained with the reference gait's peak foot lift raised from 8 cm to **12 cm**, for only 600k steps.
Use it when taller steps / bumps matter (stairs up 6-8 cm, rough 10 cm); the main package (`terrain_policy_seed2.zip`, 8 cm lift, 2M steps) is the default. Same scope: rough ground to 10 cm, slopes to 15 deg, stairs up to 4 cm and stairs down to 6 cm in training; NO pushes; blind (no scan).

## How to run it (differences to the main package)
- The environment must use the 12 cm lift: `Go1TerrainWalkEnv(lift=0.12)` (racevla/envs/go1_terrain_walk.py), or any terrain-wrapped Go1WalkingSineEnv with `env.lift = 0.12` set. With the default 8 cm lift the policy is out of its training conditions (the lift test showed that a lift the policy was not trained for hurts).
- Everything else as in terrain_policy_final.md: 49-D observation, FixedObsNormalize, forward command 0.4-0.8 m/s plus the heading hold (turn = clip(-2 x yaw error, +-0.5)), `PPO.load`, `predict(obs, deterministic=True)`.

## Results (20 fresh terrain seeds per cell, forward command 0.4 / 0.6 m/s, success = x reaches 6 m within 20 s; `terrain_policy_lift12_comparison.md` has every cell)
| | old walking policy | seed 2 (8 cm lift, 2M steps) | this model (12 cm lift, 600k steps) |
|---|---|---|---|
| average over the in-scope cells | 80 / 94 % | 99 / 99 % | 100 / 100 % |
| average over the out-of-scope cells | 14 / 14 % | 49 / 46 % | 46 / 59 % |
| rough 10 cm | 10 / 40 | 90 / 80 | 95 / 90 |
| stairs up 6 cm | 0 / 0 | 100 / 100 | 100 / 100 |
| stairs up 8 cm | 0 / 0 | 0 / 35 | 0 / 70 |
| stairs up 10-15 cm | 0 / 0 | 0 / 0 | 0 / 0 |
| stairs down 10 cm | 30 / 55 | 100 / 100 | 100 / 95 |
| stairs down 12 cm | 10 / 45 | 95 / 60 | 90 / 100 |
| stairs down 15 cm | 0 / 0 | 45 / 15 | 20 / 65 |
All other in-scope cells (rough to 8 cm, all slopes up and down to 15 deg, stairs up 2 and 4 cm, stairs down 2-6 cm) are 100 / 100 for both models.

## Training config
`scripts/phase6_terrain/05_train_terrain_ppo.py --run-name terr_lift12_seed2 --seed 2 --steps 600000 --eval-every 200000 --lift 0.12 --init outputs/walk4_sine_1M_seed2/checkpoints/model_1000000.zip` (same recipe and curriculum caps as the main package; only the lift and the length differ; 600k steps took about 40 min).
It reached a perfect in-training check (9 fixed terrains, 5 episodes each) at 400k and 600k steps although its curriculum was slower (rough ground only at 6 cm at the end).

## Known limits
- One seed, 600k steps; the main package has three seeds at 2M steps. The comparison above is a single run each.
- Stairs up of 8 cm only work at 0.6 m/s (70 %), 0 % at 0.4 m/s; 10 cm and above 0 %. A height scan (Skill B) did not fix this cleanly (racevla/envs/go1_scan_walk.py, go1_scan_gait.py; see the memory notes): the scan-aware gait gained a little on stairs up but lost on slopes and stairs down.
- Only 0.4 and 0.6 m/s evaluated, no pushes, no running on terrain, simulation only, one terrain generator, no friction / mass randomization.
