| terrain | level | control: 8 cm lift, 600k steps | retrained with 12 cm lift, 600k steps | seed 2 final (2M steps), lift 8 cm | same model, lift raised to 10 cm (no retraining) |
|---|---|---|---|---|---|
| rough | 0 cm | 100% (0f 0s 0l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) |
| rough | 8 cm | 95% (1f 0s 0l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) |
| rough | 10 cm | 55% (6f 0s 3l) | 90% (2f 0s 0l) | 80% (2f 0s 2l) | 85% (3f 0s 0l) |
| slope_up | 15 deg | 100% (0f 0s 0l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) |
| stairs_up | 4 cm | 100% (0f 0s 0l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) |
| stairs_up | 6 cm | 65% (2f 0s 5l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) |
| stairs_up | 8 cm | 0% (19f 1s 0l) | 70% (3f 1s 2l) | 35% (12f 1s 0l) | 75% (4f 1s 0l) |
| stairs_up | 10 cm | 0% (10f 10s 0l) | 0% (13f 7s 0l) | 0% (16f 1s 3l) | 0% (7f 13s 0l) |
| stairs_up | 12 cm | 0% (2f 18s 0l) | 0% (9f 11s 0l) | 0% (17f 3s 0l) | 0% (13f 7s 0l) |
| stairs_down | 6 cm | 100% (0f 0s 0l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) |
| stairs_down | 10 cm | 20% (16f 0s 0l) | 95% (0f 1s 0l) | 100% (0f 0s 0l) | 100% (0f 0s 0l) |

Success of 20 episodes at 0.6 m/s with heading hold (f = fell, s = stuck, l = left the course).
