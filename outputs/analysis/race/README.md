# Race results

Written 2026-10-09. Write-up: [`../../../docs/race.md`](../../../docs/race.md). Scripts: [`scripts/phase8_race/`](../../../scripts/phase8_race). 40 random 4-section courses, seeds 4000-4039.

| File | What |
|---|---|
| `summary_oracle.md`, `races_oracle.csv` | oracle hint (read from the course layout): 92 % finished |
| `summary_vision.md`, `races_vision.csv` | the terrain classifier's hint: 58 % finished |
| `summary_vision_hold10.md`, `summary_vision_hold25.md`, `races_vision_hold*.csv` | classifier hint smoothed over 10 / 25 steps: 62 % / 58 % |
| `summary_diag.md`, `races_diag.csv`, `diag_diag.json` | the vision run again with every step logged (it reproduces `summary_vision.md` exactly; `races_diag.csv` is identical to `races_vision.csv`); input of the diagnosis |
| `diagnose_diag.md`, `diagnose_confusion_diag.png` | per-step diagnosis: truth vs what the classifier said, by skill, by distance to the next section |
| `courses.png` | side views of six random courses |

CSV columns: seed, target speed, hint, layout, finish line, success / fell / left / stuck / timeout, end skill and x, time, sections passed, the section it ended on, hint agreement counts, skill steps, switches.
