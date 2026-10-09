Classifier output during 40 races (55316 steps), no smoothing.

Accuracy over all race steps: 7 classes 35 %, 3 groups (flat / terrain / bar) 52 %. (Single patches on a flat floor, held-out frames: 73 % / 81 %.)

Truth (rows, with the share of all steps) -> what the classifier said (% of that row):

| truth | steps % | flat | rough | slope_up | slope_down | stairs_up | stairs_down | bar |
|---|---|---|---|---|---|---|---|---|
| flat | 36 | 12 | 22 | 6 | 29 | 3 | 18 | 10 |
| rough | 18 | 11 | 44 | 1 | 13 | 2 | 18 | 11 |
| slope_up | 18 | 3 | 4 | 62 | 2 | 16 | 2 | 10 |
| slope_down | 3 | 0 | 0 | 0 | 71 | 1 | 26 | 1 |
| stairs_up | 12 | 6 | 11 | 14 | 13 | 40 | 5 | 11 |
| stairs_down | 4 | 4 | 3 | 1 | 19 | 0 | 72 | 2 |
| bar | 9 | 11 | 14 | 1 | 39 | 1 | 23 | 10 |

Group accuracy by the skill that was driving:

| | steps | group accuracy % |
|---|---|---|
| stand | 1040 | 30 |
| walk | 1452 | 48 |
| run | 8585 | 46 |
| terrain | 41304 | 56 |
| stepover | 2935 | 28 |

Truth = flat (36 % of steps): what the classifier said:

| says | % of the flat steps |
|---|---|
| flat | 12 |
| rough | 22 |
| slope_up | 6 |
| slope_down | 29 |
| stairs_up | 3 |
| stairs_down | 18 |
| bar | 10 |

False alarms (says patch or bar although truth = flat) by the distance to the next section's start (m; the oracle hint starts 1 m ahead of a patch, 2 m ahead of a bar):

| | steps | correct (says flat) % |
|---|---|---|
| 1 | 6135 | 14 |
| 2 | 5189 | 8 |
| 3 | 2471 | 7 |
| 4 | 2769 | 25 |
| 6 | 3106 | 7 |

Truth = rough (9783 steps): says rough 44 %, stairs_down 18 %, slope_down 13 %, flat 11 %

Truth = slope_up (10171 steps): says slope_up 62 %, stairs_up 16 %, bar 10 %, rough 4 %

Truth = slope_down (1871 steps): says slope_down 71 %, stairs_down 26 %, stairs_up 1 %, bar 1 %

Truth = stairs_up (6534 steps): says stairs_up 40 %, slope_up 14 %, slope_down 13 %, rough 11 %

Truth = stairs_down (2423 steps): says stairs_down 72 %, slope_down 19 %, flat 4 %, rough 3 %

Truth = bar (4864 steps): says slope_down 39 %, stairs_down 23 %, rough 14 %, flat 11 %

finished races: group accuracy 50 % over 36451 steps

unfinished races: group accuracy 56 % over 18865 steps
