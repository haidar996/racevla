# Released policies

Stable-Baselines3 `.zip` files. Always wrap the environment in `FixedObsNormalize` (`racevla/envs/wrappers.py`) and call `predict(obs, deterministic=True)`. Each skill has a model card with the exact environment, training command, results and known limits. The cards mention run folders such as `outputs/walk4_sine_1M_seed1`: those are not stored in the repository, the learning curves are in [`../experiments/runs`](../experiments/runs).

| Skill | Use this | Card | Algorithm | Backups / variants |
|---|---|---|---|---|
| Stand + push recovery | `standing_policy_td3_seed0.zip` (TD3.load, 0.2 s zero-action warm-up) | [standing_policy_final.md](standing_policy_final.md) | TD3 | `standing_policy_ppo_wide_seed2_nowarmup.zip` (PPO, no warm-up needed); `standing_recovery_ppo_final.zip` is the earlier PPO model, see [standing_recovery_ppo_final.md](standing_recovery_ppo_final.md) (its "0/30" claim is superseded by the first card) |
| Walking | `walking_policy_sine_seed1.zip` | [walking_policy_final.md](walking_policy_final.md) | PPO | `walking_policy_sine_seed2_backup.zip` |
| Running | `running_policy_2p5_seed0.zip` + `running_gait_table.json` | [running_policy_final.md](running_policy_final.md) | PPO | `running_policy_2p5_seed1_backup.zip`, `running_policy_2p0_seed1.zip` (0-2.0 m/s only) |
| Blind terrain | `terrain_policy_seed2.zip` | [terrain_policy_final.md](terrain_policy_final.md), [comparison](terrain_policy_comparison.md) | PPO | `terrain_policy_seed0_backup.zip`; `terrain_policy_lift12_seed2.zip` ([card](terrain_policy_lift12_final.md)) needs `Go1TerrainWalkEnv(lift=0.12)` |
| Step over a bar | `hurdle_stepover_seed0.zip` | [hurdle_stepover_final.md](hurdle_stepover_final.md) | PPO | evaluation of all checkpoints: `hurdle_stepover_eval.csv` |
| Terrain classifier (vision) | `terrain_classifier_cnn_round3.pt` (PyTorch, `TerrainClassifier(path, use_gravity=True)`) | [terrain_classifier_final.md](terrain_classifier_final.md) | small CNN, depth image + gravity vector | `terrain_classifier_cnn_round1.pt` (shortcut-biased, 83 % was inflated) and `_round2.pt` (no gravity input, 69 %): load with `use_gravity=False`; kept for the record. Not a policy: the classifier replaces the simulator's terrain hint |

`jump_primitive_hurdle.json` holds the parameters of a scripted jump experiment that is not part of any released skill (see [`../experiments`](../experiments#dropped-experiments-negative-results)).
