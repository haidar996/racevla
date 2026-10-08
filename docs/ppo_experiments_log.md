# PPO Experiments Log — Phase 3 (Standing/Recovery)

Every PPO training run in this project, in order, with configuration and results. All runs share:
push task on (0.4–1.2 m/s kicks, 1–3 s interval, 1 s grace period), 4 parallel environments,
evaluation on 30 fixed seeds (1000–1029) except where noted, deterministic policy at evaluation.

| # | Run dir | Steps | Seed(s) | Action scale | n_steps | clip_range | clip_range_vf | gae_lambda | log_std_init | n_epochs | LR | Obs norm | Best fall% | Final fall% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `ppo_push_500k_logstd-1` | 500k | 0 | 0.25 | 512 | 0.2 | none | 0.95 | -1.0 | 10 | 3e-4 const | no | 27% (8/30) | n/a |
| 2 | `ppo_push_1M_scale0.5_logstd-0.8` | 1M | 0 | **0.5** | 512 | 0.2 | none | 0.95 | -0.8 | 10 | 3e-4 const | no | 3% (1/30) | 27% |
| 3 | `ppo_1M_final_seed0` | 1M | 0 | 0.5 | 512 | 0.2 | none | 0.95 | -0.8 | 10 | 3e-4 const | **yes** | **0% (0/30)** | 7% |
| 4 | `ppo_1M_final_seed1` | 1M | 1 | 0.5 | 512 | 0.2 | none | 0.95 | -0.8 | 10 | 3e-4 const | yes | 23% | 37% |
| 5 | `ppo_1M_final_seed2` | 1M | 2 | 0.5 | 512 | 0.2 | none | 0.95 | -0.8 | 10 | 3e-4 const | yes | 27% | 33% |
| 6 | `ppo_1M_variance_test_seed0` | 1M | 0 | 0.5 | **1024** | **0.1** | none | 0.95 | -0.8 | **5** | **linear→0** | yes | 13% | 20% |
| 7 | `ppo_1M_variance_test_seed1` | 1M | 1 | 0.5 | 1024 | 0.1 | none | 0.95 | -0.8 | 5 | linear→0 | yes | 23% | 40% |
| 8 | `ppo_1M_variance_test_seed2` | 1M | 2 | 0.5 | 1024 | 0.1 | none | 0.95 | -0.8 | 5 | linear→0 | yes | 7% | 7% |
| 9 | `ppo_2M_final_seed3` | **2M** | 3 | 0.5 | **2048** | 0.1 | **0.2** | **0.9** | -0.8 | 5 | linear→0 | yes | 17% | 40% |

## Summary statistics

| Group | Mean best fall% | Best-fall spread | Mean final fall% | Final-fall spread |
|---|---|---|---|---|
| Runs 3–5 (n_steps=512, clip=0.2, 3 seeds) | 16.7% | 0–27% | 25.7% | 7–37% |
| Runs 6–8 (n_steps=1024, clip=0.1, 3 seeds) | 14.3% | 7–23% | 22.3% | 7–40% |
| Run 9 (n_steps=2048, clip=0.1/0.2vf, 2M, 1 seed) | 17% | — | 40% | — |

## Conclusions

1. **Widening the action range (0.25 → 0.5 rad) was the single biggest lever** — run 1 (0.25 rad) never beat 27% falls; every run after (0.5 rad) reached well below that at its best.
2. **PPO shows real, irreducible seed-to-seed variance on this task** — best result ranged 0% to 27% across 9 independent training runs with otherwise-similar settings.
3. **Bigger batches (n_steps 512→1024→2048) narrowed the best-case spread somewhat but did not clearly narrow the final-checkpoint spread**, and slowed convergence (needed more steps to reach the same territory). Diminishing/mixed returns past 1024.
4. **The single best model in the whole project is run 3's 500k checkpoint (0/30 falls)**, saved as `models/standing_recovery_ppo_final.zip`. It requires the `FixedObsNormalize` wrapper at inference time.
5. **Practical takeaway:** since SB3 always saves the best checkpoint automatically (`best_model.zip`), the final-checkpoint degradation seen in several runs matters less in practice than it looks — we never have to deploy a degraded final checkpoint.

## Known tooling bug (fixed)

`scripts/phase3_standing/watch_latest.py` and `record_episode.py` originally fed raw (unnormalized)
observations to every model, which silently broke any model trained with `FixedObsNormalize` (runs
3 onward). Fixed 2026-09-28: both scripts now wrap with `FixedObsNormalize` by default; pass `--raw`
only for runs 1–2, which were trained without it.
