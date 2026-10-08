| terrain | level | in scope | baseline | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|---|---|
| rough | 0 cm | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| rough | 1 cm | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| rough | 2 cm | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| rough | 3 cm | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| rough | 4 cm | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| rough | 5 cm | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| rough | 6 cm | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| rough | 8 cm | yes | 85 / 100 | 100 / 100 | 100 / 95 | 95 / 100 |
| rough | 10 cm | yes | 10 / 40 | 100 / 65 | 60 / 65 | 90 / 80 |
| slope_up | 3 deg | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| slope_up | 6 deg | yes | 20 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| slope_up | 9 deg | yes | 0 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| slope_up | 12 deg | yes | 0 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| slope_up | 15 deg | yes | 0 / 15 | 100 / 100 | 100 / 100 | 100 / 100 |
| slope_down | 3 deg | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| slope_down | 6 deg | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| slope_down | 9 deg | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| slope_down | 12 deg | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| slope_down | 15 deg | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| stairs_up | 2 cm | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| stairs_up | 4 cm | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| stairs_up | 6 cm | no | 0 / 0 | 100 / 100 | 40 / 65 | 100 / 100 |
| stairs_up | 8 cm | no | 0 / 0 | 5 / 35 | 0 / 0 | 0 / 35 |
| stairs_up | 10 cm | no | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| stairs_up | 12 cm | no | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| stairs_up | 15 cm | no | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| stairs_down | 2 cm | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| stairs_down | 4 cm | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| stairs_down | 6 cm | yes | 100 / 100 | 100 / 100 | 100 / 100 | 100 / 100 |
| stairs_down | 8 cm | no | 85 / 30 | 100 / 100 | 100 / 100 | 100 / 100 |
| stairs_down | 10 cm | no | 30 / 55 | 95 / 100 | 100 / 95 | 100 / 100 |
| stairs_down | 12 cm | no | 10 / 45 | 40 / 45 | 85 / 45 | 95 / 60 |
| stairs_down | 15 cm | no | 0 / 0 | 15 / 10 | 35 / 5 | 45 / 15 |

| average over levels (0.4 / 0.6 m/s) | baseline | seed 0 | seed 1 | seed 2 |
|---|---|---|---|---|
| in scope (rough to 10 cm, slopes to 15 deg, stairs up to 4 cm, stairs down to 6 cm) | 80% / 94% | 100% / 99% | 98% / 98% | 99% / 99% |
| out of scope | 14% / 14% | 39% / 43% | 40% / 34% | 49% / 46% |

Each cell: success % of 20 episodes at 0.4 / 0.6 m/s (x reaches 6 m within 20 s, heading hold). Seeds = 2M-step terrain-trained models.
