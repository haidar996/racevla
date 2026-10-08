# Scripts

One folder per project phase. Within a folder, scripts are numbered in the order they were written (gaps in the numbering are scripts for experiments that were dropped). Prefix meaning: `test_*` check an environment, `train_*` train a policy, `eval_*` / `compare_*` evaluate, `view_*` open the MuJoCo viewer (add `--no-window` for a headless check).

| Folder | Content |
|---|---|
| `phase1_foundation` | verify the Go1 model, observation / action / PD / reset / termination / reward tests, push test |
| `phase3_standing` | standing + push recovery: PPO, SAC, TD3 training, multi-seed evaluation, checkpoint selection, recording and viewers |
| `phase4_walking` | walking: three gait options (clock+schedule, sine reference + residual, reward-only), evaluation, comparison |
| `phase5_running` | running 1.5 → 2.5 m/s: leg probe, gait-table validation, training pipeline, evaluation, viewer |
| `phase6_terrain` | terrain generator test, baselines, mixed-curriculum training, stairs/slope tour viewer, lift test, scan experiments |
| `phase7_skills` | hurdle step-over (train, evaluate, view); `01`/`02` are an *experimental, paused* fall-recovery environment |
| `make_demos.py` | renders the README GIFs |

Training scripts write to `outputs/<run-name>/` (git-ignored; the learning curves of all runs are copied to `../experiments/runs/`; the released policies are in `../models/`, evaluation tables in `../outputs/analysis/`). Everything runs on CPU.
