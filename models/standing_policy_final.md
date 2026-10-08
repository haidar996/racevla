# Final standing / push-recovery policy (Phase 3) -- written 2026-10-02

**Use `standing_policy_td3_seed0.zip` with a short passive warm-up.** Backups: `standing_policy_td3_seed1_backup.zip` (TD3, same recipe, other seed) and
`standing_policy_ppo_wide_seed2_nowarmup.zip` (PPO trained against pushes up to 1.6 m/s; needs no warm-up). This file supersedes the "0/30 falls" claim in
`standing_recovery_ppo_final.md` (that model's true held-out fall rate is about 9 %).

## How to run it (all three models)
- Environment: `Go1StandingEnv(push=True)` wrapped in `FixedObsNormalize` (racevla/envs/wrappers.py). Feeding raw observations makes the policy meaningless.
- Action: 12 joint offsets in [-1, 1], scaled by 0.5 rad (ACTION_SCALE), PD controller KP=100, KD=0.5, 50 Hz policy.
- Load with SB3: `TD3.load(path)` for the TD3 models, `PPO.load(path)` for the PPO model; always call `predict(obs, deterministic=True)`.
- **Passive warm-up (TD3 models only):** for the first 10 to 25 policy steps (0.2 to 0.5 s) command ZERO action (hold the home pose), then switch the policy on.
  Without it the TD3 models fall at the very start of ~2-4.5 % of episodes (shallow lower-leg floor contacts); with it 0 of 200.

## Results on 200 fresh seeds (4000-4199, never used for selection), 20 s episodes, pushes 0.4-1.2 m/s every 1-3 s
| Model | Checkpoint | No warm-up | With warm-up | Mean return |
|---|---|---|---|---|
| TD3 seed 0 (primary) | 750k of 1M | 9/200 (4.5 %) | **0/200** (k=10 and k=25) | 906 |
| TD3 seed 1 (backup) | 700k of 1M | 4/200 (2.0 %) | **0/200** (k=10 and k=25) | 915 |
| PPO wide-push seed 2 | 750k of 1M | 1/200 (0.5 %) | not needed | 968 |
(For comparison: doing nothing falls 61 %; original PPO best models 9-35 %.)

Push robustness, one controlled push at step 100 (falls out of ~30 episodes): TD3 seed 0: 0/29 up to 1.4 m/s, 3/29 at 1.6; TD3 seed 1: 0/30, 1/30, 3/30, 5/30 at 1.0/1.2/1.4/1.6 m/s.

## Training config (TD3)
`scripts/phase3_standing/07_train_td3.py`: 1M steps, 4 envs, buffer 1M, lr 3e-4 decayed linearly to 0, 5-step returns, 4 gradient steps per env step, batch 256,
tau 0.005, policy delay 2, target noise 0.2/clip 0.5, exploration noise sigma 0.1, nets 256x256, ACTION_SCALE 0.5, standing push task. Checkpoint chosen by
`scripts/phase3_standing/06_select_checkpoint.py` (best of 17 on 100 seeds, then re-tested on 200 fresh seeds).

## Known limits
- Never trained or tested for recovery after a fall (episodes end at 60 deg tilt or a >3 mm non-foot floor contact), jump landings or running stops.
- Results are for a flat floor and the fixed push distribution above; two training seeds per TD3 setting, one for the PPO backup.
- Final (1M-step) checkpoints of PPO runs are much worse than the selected ones; always use the selected files here, not `latest_model.zip` from the run folders.
