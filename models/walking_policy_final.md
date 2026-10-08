# Final walking policy (Phase 4, option B) -- written 2026-10-03

**Use `walking_policy_sine_seed1.zip`** (PPO, option B: sine reference leg motion from a gait clock + small learned residual; 1M steps, last checkpoint).
Backup: `walking_policy_sine_seed2_backup.zip` (same recipe, other seed). Seed 0 was equally good (0/50 falls, fwd err 0.033) but is not packaged.
Options A (clock + schedule reward) and C (reward-only) are NOT packaged: A matches B but needs twice the steps; C never learned a trot (see outputs/analysis/walking_comparison.md).

## How to run it
- Environment: `Go1WalkingSineEnv(push=False, vx_range=(0, 0.6), wz_range=(-0.5, 0.5))` from `racevla/envs/go1_walking_sine.py`, wrapped in `FixedObsNormalize` (racevla/envs/wrappers.py). Raw observations make the policy meaningless.
- Observation: 49 numbers = 45 robot + 2 command (vx, wz) + 2 clock (sin, cos; both 0 when told to stand). The clock runs at 2.5 Hz only while |command| > 0.1.
- Action: 12 joint offsets in [-1, 1], residual scale 0.25 rad, added to home pose + the reference leg motion; PD controller KP=100, KD=0.5, 50 Hz policy.
- Load with SB3: `PPO.load(path)`, always `predict(obs, deterministic=True)`. No warm-up needed.
- Command is part of the observation: set `env.command = [vx, wz]` and refresh obs[-4:-2] (command) and obs[-2:] (`env._clock()`), as `scripts/phase4_walking/03_view_walk.py --sine` does.
- Viewer: `python scripts/phase4_walking/03_view_walk.py models/walking_policy_sine_seed1.zip --sine`

## Results (50 fresh episodes, 1000 steps, random commands, no pushes)
| Model | Falls | Fwd error m/s | Speed reached | Turn error rad/s | Turn achieved (0.40 asked) | Feet on floor FR/FL/RR/RL | Slip | Lift (p95) |
|---|---|---|---|---|---|---|---|---|
| seed 1 (primary) | 0/50 | 0.030 | 96% | 0.054 | 0.36 | 0.67/0.67/0.67/0.70 | 0.11 | 7.4 cm |
| seed 2 (backup) | 0/50 | 0.037 | 96% | 0.052 | 0.38 | 0.68/0.66/0.69/0.69 | 0.11 | 6.8 cm |
Steady commands (seed 1, 4 episodes each, 0 falls): vx 0.2 -> 0.198, 0.4 -> 0.390, 0.6 -> 0.596 m/s; turn in place 0.5 -> 0.38 / -0.33 rad/s; standing: all four feet down, speed ~0.

## Training config
`scripts/phase4_walking/02_train_walk_ppo.py --env sine --run-name walk4_sine_1M_seed1 --seed 1`: PPO, 4 envs, n_steps 512, 10 epochs, clip 0.2, GAE 0.95, lr 3e-4 constant, log_std_init -0.8, 1M steps, no pushes.
Commands: vx 0-0.6 m/s, wz +-0.5 rad/s, resampled every 200 steps, 15 % stand. Learned a good gait after ~200-250k steps.

## Known limits
- NEVER tested with pushes, backwards walking, speeds above 0.6 m/s or turn rates above 0.5 rad/s, or on uneven ground.
- Speed reached is 96 % of the command (slightly under); turn achieved is 90-95 % of the command.
- Episodes end on trunk contact or tilt > 60 deg; recovery after a fall is not trained.
