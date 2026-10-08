# Hurdle crossing while running by STEPPING OVER (Phase 7, option 1) -- written 2026-10-08

**`hurdle_stepover_seed0.zip`** = checkpoint at 750k steps of the run `stepover_seed0` (PPO, 1M steps planned; the 750k checkpoint was the best of five checkpoints on a 20-episode evaluation, see `hurdle_stepover_eval.csv`; the 1M model and a 2M-step continuation were NOT better).
It runs at 1.3-1.7 m/s and crosses a thin full-width bar (8 cm thick, at x = 5 m) by lifting the feet over it (high-stepping), without a jump, and keeps running. NOT a jump: no flight phase is used for the crossing.

## How to run it
- Environment: `Go1StepOverEnv()` (racevla/envs/go1_stepover.py; built on racevla/envs/go1_hurdle.py and the terrain code) wrapped in `FixedObsNormalize`. Observation 75 = 49 (45 robot, 2 command, 2 clock) + 24 far height scan (8 rows 0.2-1.6 m ahead, each the HIGHEST ground in a 0.2 m cell, 3 columns, x4, noise 0.5 cm in training) + 2 jump numbers (always 0 here). Action 12 = joint corrections (scale 0.25 rad) on top of the high-stepping reference gait.
- Reference gait: the speed-dependent running gait (models/running_gait_table.json) + a term that lifts each swing foot to (highest ground within 0.32 m ahead + 6 cm) with an early plateau, cap 17.5 cm above the home foot height.
- Reset: `env.reset(seed, options={"terrain": ("hurdle", height_in_m), "terrain_seed": seed})`; the command is drawn from U(1.3, 1.7) m/s with a heading hold. Load with `PPO.load`, `predict(obs, deterministic=True)`.
- Success (environment rule): 2 s after passing x = 5.45 m (or at x = 8.95 m) the robot is upright (tilt < 35 deg) and runs at >= 50 % of the commanded speed. A "hit" = trunk, hip or thigh touches the bar (calf / foot may brush it).

## Results (20 fresh episodes per cell, seeds 2000-2019; success % (cleared-the-bar %))
| model | empty | 3 cm | 5 cm | 7 cm | 10 cm | 12.5 cm | 15 cm | 20 cm |
|---|---|---|---|---|---|---|---|---|
| **this model (750k)** | 100 (0) | 100 (100) | 90 (90) | **80 (80)** | 35 (50) | 10 (10) | 0 (5) | 0 (0) |
| same run, 500k | 100 | 95 (100) | 80 (90) | 45 (50) | 30 (40) | 20 (30) | 0 | 0 |
| same run, 1M | 95 | 90 (90) | 90 (90) | 60 (70) | 20 (25) | 10 (25) | 0 (5) | 0 |
| continuation (2M plan, stopped), 500k | 95 | 95 | 75 | 45 | 30 | 15 | 0 | 0 |
| continuation, 1M | 95 | 90 | 70 | 45 | 15 | 0 | 0 | 0 |
No hurdle: the running policy it started from; bars up to 5 cm already worked at about 40-50 % without any training, 7 cm at 3/12 (plain reference) and 5/12 (step-over reference).

## Known limits (honest)
- Reliable to about 7 cm (80 %); 10 cm only about a third of the episodes; 12.5 cm and above essentially not crossed (the best cell 20 %).
- The 20 cm target of 10-25 cm hurdles is NOT reached. A scripted JUMP (racevla/envs/jump_primitive.py, models/jump_primitive_hurdle.json) was tried first: it clears the height (24-43 cm in the height-only test) but the flight is too short (0.05-0.5 m) for a 0.65 m long robot to pass over a bar; the best searched jump cleared 10 cm in 6/20 and 15 cm in 2/20 starts at 2.0 m/s; learning the trigger (3 pilots) did not get past 3 cm bars.
- Speeds 1.3-1.7 m/s only; one seed; the scan comes from the simulator height map (a camera would have to supply it); no pushes; thin bar only (8 cm thick, 4 cm grid); evaluations of single 5-episode checks swing by +-20 points, so use the 20-episode table.
