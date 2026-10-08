# Standing/recovery PPO model — Phase 3 final (updated 2026-09-28)

Source: `outputs/ppo_1M_final_seed0/best_model.zip`, checkpoint at **500,000** of 1,000,000 training
steps (seed 0 of the "original" 3-seed sweep). This is the best single result across **every** PPO
experiment run in this project (9 runs total, see `docs/ppo_experiments_log.md` for the full table).

**IMPORTANT — observation normalization:** this model was trained WITH `FixedObsNormalize`
(`racevla/envs/wrappers.py`). Any script that loads it must wrap the environment with that same
wrapper before calling `model.predict()`, e.g. `watch_latest.py` / `record_episode.py` WITHOUT the
`--raw` flag. Feeding it raw observations makes its behavior meaningless (this bug happened once
already during this project — see the memory log).

## Training config
- PPO, Stable-Baselines3, gamma=0.99, gae_lambda=0.95 (default), clip_range=0.2 (default),
  n_epochs=10 (default), batch_size=64, lr=3e-4 constant (no decay), ent_coef=0.0, vf_coef=0.5.
- `log_std_init = -0.8` (exploration std ≈ 0.45 at start).
- `n_steps = 512` (2048 samples/update), 4 parallel envs.
- `ACTION_SCALE = 0.5` rad (racevla/envs/action.py).
- Push disturbance on (`push=True`): 0.4–1.2 m/s horizontal kick, every 1–3 s, after a 1 s grace period.
- Seed 0.

## Evaluation (30 fixed episodes, seeds 1000-1029, deterministic)
- **0/30 falls (0%)** at this checkpoint (500k) — the single best result of the whole project.
- Same seed's run finished at 1,000,000 steps with 2/30 (7%) — so the *final* checkpoint of this
  training run is also excellent, not just this one lucky evaluation point.

## Known limits (unchanged from before)
- Only trained/tested against `Go1StandingEnv`'s reset and push distribution (small pose noise,
  horizontal velocity kicks up to 1.2 m/s). NOT tested for recovery from a fall, a jump landing, or
  a running stop — those states are never visited during training (termination ends the episode at
  60° tilt), so behavior there is undefined.
- PPO training on this task showed real seed-to-seed variance (best fall rate ranged 0-27% across
  9 separate training runs); this specific checkpoint is the best case, not the typical case.
