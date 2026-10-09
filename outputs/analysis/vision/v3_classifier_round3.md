Test set: 8348 frames from 140 episodes never used in training.

- 7-class accuracy 73.4 %, balanced (mean per-class recall) 73.0 %
- 3 groups (flat / terrain patch / bar): accuracy 80.7 %

| class | frames | recall % | most common mistake |
|---|---|---|---|
| flat | 4500 | 76.3 | bar (7.6 %) |
| rough | 1117 | 61.3 | bar (19.2 %) |
| slope_up | 844 | 72.7 | stairs_up (9.7 %) |
| slope_down | 445 | 71.7 | stairs_down (20.9 %) |
| stairs_up | 624 | 65.4 | slope_up (13.5 %) |
| stairs_down | 462 | 81.6 | slope_down (10.6 %) |
| bar | 356 | 81.7 | flat (8.1 %) |

Accuracy against the distance from the robot's base to the patch start (patch kinds only, flat episodes excluded; negative = already on the patch):

| distance (m) | frames | 7-class acc % | 3-group acc % |
|---|---|---|---|
| -9 to -1 | 3798 | 73.6 | 83.0 |
| -1 to 0 | 1130 | 76.8 | 87.4 |
| 0 to 0.5 | 515 | 77.7 | 92.8 |
| 0.5 to 1 | 387 | 75.7 | 90.4 |
| 1 to 1.5 | 663 | 58.1 | 58.1 |
| 1.5 to 2 | 288 | 53.8 | 53.8 |
| 2 to 3 | 148 | 92.6 | 92.6 |
| 3 to 9 | 463 | 80.3 | 80.3 |

Robustness: extra depth noise on the test images (sigma = noise x depth, plus 1 % lost pixels):

| extra noise | 7-class acc % | 3-group acc % |
|---|---|---|
| 0 | 73.4 | 80.7 |
| 0.01 | 72.8 | 80.3 |
| 0.02 | 72.7 | 80.2 |
| 0.05 | 71.5 | 79.6 |
