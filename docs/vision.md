# Vision (Phase 10): onboard depth camera, elevation map, terrain classifier

Written 2026-10-09. Code: [`racevla/vision/`](../racevla/vision) · scripts: [`scripts/phase10_vision/`](../scripts/phase10_vision) · results: [`outputs/analysis/vision/`](../outputs/analysis/vision) · model card: [`models/terrain_classifier_final.md`](../models/terrain_classifier_final.md).

Until this phase the terrain skills got their information from the simulator: the step-over skill read a height scan from the simulator's height field, and the supervisor was told the terrain type by an **oracle**. An *oracle*, in machine-learning language, is a source of the true answer that the real system would not have: here the simulator, which knows exactly where each patch starts and ends. Vision replaces both with something a real robot could have: a depth camera.

Decisions made with the project owner: depth only (no colour), a classifier **trained from scratch on my own simulator data** (no downloaded model), a single frame (no history).

## V1: the onboard camera

- A camera `onboard` on the Go1 trunk ([`go1.xml`](../assets/robots/unitree_go1/go1.xml)): 0.32 m ahead of the trunk centre, 0.04 m up, pitched 25° down, 58° vertical field of view. It has no physics (the existing tests and reset / termination checks still pass).
- [`OnboardCamera`](../racevla/vision/camera.py): renders a 64 x 48 depth image (metres along the viewing axis, clipped to 5 m; about 4 ms per frame with EGL).
- The picture is rigidly attached to the trunk, so it tilts and shakes with the body, as a real camera would.
- **A bug worth knowing:** MuJoCo's renderer copies a height field to the graphics card *once*, when the renderer is created. The physics saw each new terrain, but the camera kept drawing the old (flat) one. `OnboardCamera.render` now re-uploads the height field whenever its data changes (`refresh_hfield`).
- View it: `python scripts/phase10_vision/01_show_camera.py [--gif]` → [`camera_views.png`](../outputs/analysis/vision/camera_views.png), [`camera_bar_run.gif`](../outputs/analysis/vision/camera_bar_run.gif).

## V2: elevation map from depth (geometry, no training)

[`HeightMapper`](../racevla/vision/heightmap.py) back-projects every depth pixel to a 3-D point (pinhole model, focal length from the field of view), moves it into the world with the camera pose, and keeps the highest point of every 4 cm map cell; a cell keeps its newest observation. `query(x, y)` returns the nearest observed cell within 2 cells, else a fallback (the mean height of the two lowest feet). `attach()` makes the body use this map instead of the simulator's height field, for **both** the 24-number far scan and the step-over skill's foot-lift rule.

Still taken from the simulator (a real robot would estimate it): the camera pose in the world, and the foot heights used for the fallback.

Map error while driving ([`v2_map_accuracy.md`](../outputs/analysis/vision/v2_map_accuracy.md)): scan RMS **0.5-2.3 cm** (10 cm bar 2.1, stairs down 1.3, everything else ≤ 0.7), 95th percentile ≤ 5 cm, near-field (0-0.8 m ahead, the camera's blind zone) RMS 0.3-1.3 cm. A 96 x 72 image gave no gain.

Bar trials with the oracle `bar` hint, 40 per cell ([`v2_bar_trials.md`](../outputs/analysis/vision/v2_bar_trials.md)): success with the simulator's map / the 64 x 48 camera map / the 96 x 72 camera map:

| Bar | Simulator | Camera 64 x 48 | Camera 96 x 72 |
|---|---|---|---|
| 3 cm | 92 % | 90 % | 95 % |
| 5 cm | 85 % | 85 % | 80 % |
| 7 cm | 62 % | 50 % | 42 % |
| 10 cm | 30 % | 22 % | 25 % |

The camera map is about as good up to 5 cm and mildly worse above; with n = 40 the differences at 7 and 10 cm are not statistically clear.

## V3: a CNN that names the terrain

[`TerrainCNN`](../racevla/vision/classifier.py): about 400k weights, input one 48 x 64 depth image (+ the body's gravity vector, round 3), output one of 7 classes: flat, rough, slope up, slope down, stairs up, stairs down, bar. The class becomes the hint of the rule supervisor. The label of a training frame uses the same windows the oracle used in the patch trials: a patch counts as "ahead" from 1.0 m before its start to 0.3 m after its end; the bar from 2.0 m before to 1.2 m after.

Training ([`04_train_classifier.py`](../scripts/phase10_vision/04_train_classifier.py)): split by *episode* seed (seed % 10: 0 test, 1 validation, the rest training) so no episode is in two sets; class-balanced cross-entropy (flat is by far the largest class); Adam 1e-3 with cosine decay; augmentation = Gaussian depth noise (σ = 1 cm x depth / 1 m) and 1 % of the pixels losing their return. 16 epochs take about 30 min on the laptop CPU. Data ([`03_collect_depth_data.py`](../scripts/phase10_vision/03_collect_depth_data.py)): 1400 episodes (200 per kind), about 93,000 frames, git-ignored.

### Three rounds, and what each one taught

| Round | Data / change | Test frames, 7 classes | Closed loop, all cells (1 frame / vote of 5) |
|---|---|---|---|
| 1 | bar episodes always run, all others walked | 83.1 % (inflated) | 68 % / 70 % |
| 2 | every kind both walked **and** run | 69.4 % | 66 % / 70 % |
| 3 | + the gravity vector (3 IMU numbers) as a second input | **73.4 %** (3 groups 80.7 %) | **73 % / 74 %** |
| (oracle hint) | | | 84 % |

Closed loop = the one-patch trials with the classifier hint and the camera map, 30 trials per cell ([`v3_pipeline_round1.md`](../outputs/analysis/vision/v3_pipeline_round1.md), [`round2`](../outputs/analysis/vision/v3_pipeline_round2.md), [`round3`](../outputs/analysis/vision/v3_pipeline_round3.md)). Frame results: [`v3_classifier_round1.md`](../outputs/analysis/vision/v3_classifier_round1.md), [`round2`](../outputs/analysis/vision/v3_classifier_round2.md), [`round3`](../outputs/analysis/vision/v3_classifier_round3.md).

- **Round 1 looked great (83 %) and was wrong.** In the first dataset the bar episodes were always run and everything else walked, so the network partly learned "running = bar": flat frames at running speed were 74.6 % flat / 12.9 % bar (walking: 89.4 % / 3.3 %). In the loop it called flat ground a bar and the supervisor started the step-over skill 3 m early. Lesson: check what a classifier could use as a shortcut.
- **A rule deadlock showed up at the same time** (see [`skill_switching.md`](skill_switching.md)): a robot stopped at a bar could never reach the 0.4 m/s that walk → run needs. Fixed with the 50-step stall escape.
- **Round 2 (honest numbers):** walking frames 80.7 % (3 groups 88.5 %), running frames only 52.3 % (62.2 %). A single frame while running is weak: body pitch and bob make a tilted camera on flat ground look like a slope or a step.
- **Round 3:** the body's gravity vector, which the robot already has, tells the network how the camera is tilted. Biggest effect: slope up 15° went from 13 % to 73 % (93 % with the vote). The frame accuracy rose by 4 points (73.4 %), and the closed loop from 66 % to 73 %.

Round-3 closed loop per cell (30 trials, oracle / vision 1 frame / vision vote of 5):

| Patch | Oracle | 1 frame | Vote of 5 |
|---|---|---|---|
| rough 6 cm | 100 % | 97 % | 97 % |
| rough 10 cm | 67 % | 33 % | 43 % |
| slope up 9° / 15° | 100 / 100 % | 100 / 73 % | 100 / 93 % |
| slope down 15° | 100 % | 100 % | 100 % |
| stairs up 4 cm / down 6 cm | 100 / 100 % | 100 / 100 % | 100 / 100 % |
| bar 3 / 5 cm | 90 / 87 % | 77 / 70 % | 77 / 73 % |
| bar 7 / 10 cm | 57 / 23 % | 40 / 13 % | 20 / 7 % |
| **all cells** | **84 %** | **73 %** | **74 %** |

### Known limits

- **The bar is the weak point**: the hint agrees with the oracle on only 22-40 % of steps near the bar (a thin edge in a 64 x 48 image, visible briefly). The majority vote delays the hint and hurts at the bar (7 cm: 20 % instead of 40 %).
- Rough ground at 10 cm: 33-43 %, against 67 % for the oracle (the skill itself is weak there too).
- **Long courses break it**: the classifier was trained on short episodes in which the robot stays near the centre line. In races it is wrong on most flat ground once the robot drifts sideways, see [`race.md`](race.md). The single-patch numbers above do **not** carry over to a long course.
- Simulation only: depth is rendered (plus added noise), not from a real sensor; the camera pose is the exact simulator pose; the sensitivity of the map to pose error was not tested; one frame, no memory.
- The closed-loop seeds were used to compare the three rounds, so differences of a few points between rounds are within noise.

## Reproduce

```bash
export MUJOCO_GL=egl                                   # off-screen rendering; set BEFORE python imports mujoco
python scripts/phase10_vision/01_show_camera.py --gif
python scripts/phase10_vision/02_eval_height_map.py    # map accuracy and bar trials
python scripts/phase10_vision/03_collect_depth_data.py --per-kind 200 --out data/vision3
python scripts/phase10_vision/04_train_classifier.py --epochs 16 --data data/vision3 --out outputs/vision3 --gravity
python scripts/phase10_vision/05_eval_classifier.py --model outputs/vision3/terrain_cnn.pt --data data/vision3 --gravity --tag _round3
python scripts/phase10_vision/06_eval_vision_pipeline.py --trials 30 --model models/terrain_classifier_cnn_round3.pt --gravity --tag _round3
```

Collecting the 1400 episodes took about an hour on 4 cores (while another job shared the CPU). The released weights are [`models/terrain_classifier_cnn_round3.pt`](../models/terrain_classifier_cnn_round3.pt); the data is not in the repository.
