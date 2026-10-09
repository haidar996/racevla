Test set: 8348 frames from 140 episodes never used in training.

- 7-class accuracy 69.4 %, balanced (mean per-class recall) 68.0 %
- 3 groups (flat / terrain patch / bar): accuracy 78.0 %

| class | frames | recall % | most common mistake |
|---|---|---|---|
| flat | 4500 | 73.4 | bar (7.4 %) |
| rough | 1117 | 55.5 | bar (23.1 %) |
| slope_up | 844 | 68.2 | bar (10.0 %) |
| slope_down | 445 | 65.4 | stairs_down (22.5 %) |
| stairs_up | 624 | 58.0 | slope_up (18.6 %) |
| stairs_down | 462 | 81.6 | slope_down (8.2 %) |
| bar | 356 | 73.6 | flat (9.6 %) |

Accuracy against the distance from the robot's base to the patch start (patch kinds only, flat episodes excluded; negative = already on the patch):

| distance (m) | frames | 7-class acc % | 3-group acc % |
|---|---|---|---|
| -9 to -1 | 3798 | 71.4 | 81.3 |
| -1 to 0 | 1130 | 70.1 | 84.0 |
| 0 to 0.5 | 515 | 73.2 | 92.2 |
| 0.5 to 1 | 387 | 65.9 | 90.4 |
| 1 to 1.5 | 663 | 43.7 | 43.7 |
| 1.5 to 2 | 288 | 39.9 | 39.9 |
| 2 to 3 | 148 | 97.3 | 97.3 |
| 3 to 9 | 463 | 81.4 | 81.4 |

Robustness: extra depth noise on the test images (sigma = noise x depth, plus 1 % lost pixels):

| extra noise | 7-class acc % | 3-group acc % |
|---|---|---|
| 0 | 69.4 | 78.0 |
| 0.01 | 68.3 | 77.6 |
| 0.02 | 68.3 | 77.7 |
| 0.05 | 67.0 | 77.3 |
