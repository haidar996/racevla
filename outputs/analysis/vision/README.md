# Vision results

Written 2026-10-09. Write-up: [`../../../docs/vision.md`](../../../docs/vision.md). Scripts: [`scripts/phase10_vision/`](../../../scripts/phase10_vision). Model card: [`../../../models/terrain_classifier_final.md`](../../../models/terrain_classifier_final.md).

| File | What |
|---|---|
| `camera_views.png`, `camera_bar_run.gif` | what the onboard camera sees (V1): six terrains; a run and a step-over of a 7 cm bar |
| `v2_map_accuracy.md`, `v2_bar_trials.md` | elevation map error and bar trials with the camera map (V2) |
| `v3_classifier_round1.md`, `_round2.md`, `_round3.md`, `v3_confusion_round2.png`, `v3_confusion_round3.png` | classifier on held-out frames, per round |
| `v3_pipeline_round1.md`, `_round2.md`, `_round3.md` | closed-loop one-patch trials (classifier hint + camera map), 30 trials per cell, oracle for comparison |
| `v3_classifier.md`, `v3_confusion.png` | copies of the **round-2** results (written before the rounds were suffixed; kept so that links keep working) |
| `v3_pipeline.md` | copy of the **round-1** closed-loop results (same reason) |

Round 1 (83.1 % on frames) is inflated by a speed shortcut in the data, see `docs/vision.md`.
