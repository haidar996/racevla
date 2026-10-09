# Terrain classifier from one depth image (Phase 10, V3) -- written 2026-10-09

**`terrain_classifier_cnn_round3.pt`** = PyTorch state dict of `TerrainCNN(use_gravity=True)` (`racevla/vision/classifier.py`, about 400k weights). It looks at ONE 48 x 64 depth image from the onboard camera (`racevla/vision/camera.py`, max range 5 m) plus the body's gravity vector (3 IMU numbers) and outputs one of 7 classes: flat, rough, slope_up, slope_down, stairs_up, stairs_down, bar. The class is turned into the terrain hint that the rule supervisor (`racevla/skills/rules.py`) used to get from the simulator ("oracle").
Trained from scratch on my own simulator data (no pretrained network): 1400 episodes (200 per kind), 3 rounds of data (round 3 = the released one), split by episode seed (seed % 10: 0 test, 1 validation, rest training), class-balanced loss, depth noise + dropout augmentation, 16 epochs (about 30 min on the laptop CPU).

## How to run it
```python
from racevla.vision.classifier import TerrainClassifier, to_hint
clf = TerrainClassifier("models/terrain_classifier_cnn_round3.pt", use_gravity=True)
cls = clf.predict(depth_48x64, gravity_xyz)   # class index; to_hint(cls) -> supervisor hint
```
Gravity = obs[24:27] of the skill observation. The model must be loaded with `use_gravity=True`. Full pipeline: `scripts/phase10_vision/06_eval_vision_pipeline.py --model models/terrain_classifier_cnn_round3.pt --gravity`. Training: `scripts/phase10_vision/04_train_classifier.py --gravity`; data: `03_collect_depth_data.py`.

## Results
**Single frames** (8348 frames of 140 episodes never used for training or checkpoint choice): 7 classes **73.4 %** (round 2 without gravity: 69.4 %), 3 groups (flat / terrain patch / bar) 80.7 %. Weakest: rough 61 % (often called bar), stairs_up 65 %. Frames 1-2 m before a patch are the hardest (54-58 %). Extra depth noise up to 5 % of the range costs 2 points.

**Closed loop** (rule supervisor + classifier hint + camera elevation map, 30 trials per cell, one patch per trial; the oracle column uses the simulator's hint):

| patch | oracle | vision, 1 frame | vision, majority of 5 frames |
|---|---|---|---|
| rough 0.06 | 100 % | 97 % | 97 % |
| rough 0.1 | 67 % | 33 % | 43 % |
| slope_up 15 | 100 % | 73 % | 93 % |
| slope_down 15 | 100 % | 100 % | 100 % |
| stairs_up 0.04 / stairs_down 0.06 | 100 % | 100 % | 100 % |
| bar 3 / 5 cm | 90 / 87 % | 77 / 70 % | 77 / 73 % |
| bar 7 / 10 cm | 57 / 23 % | 40 / 13 % | 20 / 7 % |
| **all cells** | **84 %** | **73 %** | **74 %** |

Earlier rounds, same test: round 1 68 / 70 %, round 2 66 / 70 %. Tables: `outputs/analysis/vision/v3_pipeline_round*.md`, `v3_classifier_round*.md`.

## Known limits
- **Long courses (the main limit):** in 40 random 20-30 m race courses the classifier was right on only 35 % of steps (7 classes; 52 % for 3 groups) against 73 % / 81 % on single patches, and on flat ground it said "flat" only 12 % of the time. Cause (diagnosed, not fixed): the robot drifts sideways, the camera then sees the side edge of the 3 m wide height field as a drop, and training episodes never showed that. Races finish 58 % with this classifier against 92 % with the oracle hint ([`docs/race.md`](../docs/race.md)).
- The bar is the weak point: the hint agrees with the oracle on only 22-40 % of steps near the bar. It looks like a thin edge in a 64 x 48 image and is visible for a short time.
- Majority voting helps slopes but delays the hint and hurts at the bar (7 cm: 20 % vs 40 %).
- Rough ground 10 cm is only 33-43 % (the oracle with the same skill gets 67 %).
- Simulator only: the depth images are rendered, not from a real sensor; the camera pose is the exact simulator pose (sensitivity to pose error not tested). Single frame, no memory.
- The numbers use the checkpoint chosen on the validation split; the closed-loop seeds were not used for training, but the same seeds were used to compare the three rounds, so small differences between rounds (a few points) are within noise.

## Earlier rounds (kept for the record)
`terrain_classifier_cnn_round1.pt`: first dataset where the bar episodes were always run; frame accuracy 83.1 % was inflated by a speed shortcut (see [`docs/vision.md`](../docs/vision.md)); closed loop 68 / 70 %. `terrain_classifier_cnn_round2.pt`: both gaits in every class, no gravity input; 69.4 %; closed loop 66 / 70 %. Both load with `TerrainClassifier(path, use_gravity=False)`. Use round 3.

Files: [`docs/vision.md`](../docs/vision.md) (design, all rounds, reproduction), [`docs/race.md`](../docs/race.md) (the long-course failure).
