Map error in cm (estimate minus truth) while the robot drives over the patch. Scan rows = 0.2 ... 1.6 m ahead of the base (mean over the 3 columns); near = 0 ... 0.8 m ahead of the base.

| patch | camera | scan RMS (all rows) | scan |error| 95th pct | scan RMS row 0.2 m | row 0.8 m | row 1.6 m | near-field RMS 0-0.8 m | near-field bias |
|---|---|---|---|---|---|---|---|---|
| hurdle 0.05 | 64x48 | 0.6 | 1.2 | 0.5 | 0.6 | 0.7 | 0.3 | +0.0 |
| hurdle 0.05 | 96x72 | 0.5 | 0.0 | 0.3 | 0.5 | 0.6 | 0.3 | +0.0 |
| hurdle 0.1 | 64x48 | 2.1 | 4.9 | 1.9 | 2.2 | 2.1 | 1.2 | +0.2 |
| hurdle 0.1 | 96x72 | 2.3 | 5.0 | 2.1 | 2.4 | 2.4 | 1.3 | +0.3 |
| stairs_up 0.04 | 64x48 | 0.7 | 1.6 | 0.6 | 0.7 | 0.7 | 0.5 | +0.0 |
| stairs_up 0.04 | 96x72 | 0.7 | 1.7 | 0.6 | 0.8 | 0.7 | 0.5 | +0.0 |
| stairs_down 0.06 | 64x48 | 1.3 | 2.7 | 0.6 | 1.4 | 1.9 | 0.7 | +0.0 |
| stairs_down 0.06 | 96x72 | 1.3 | 2.7 | 0.6 | 1.4 | 1.9 | 0.7 | +0.0 |
| rough 0.06 | 64x48 | 0.6 | 1.2 | 0.5 | 0.6 | 0.6 | 0.4 | +0.2 |
| rough 0.06 | 96x72 | 0.6 | 1.2 | 0.5 | 0.6 | 0.6 | 0.4 | +0.2 |
| slope_up 9 | 64x48 | 0.5 | 1.2 | 0.4 | 0.6 | 0.6 | 0.4 | +0.1 |
| slope_up 9 | 96x72 | 0.5 | 1.2 | 0.4 | 0.5 | 0.6 | 0.4 | +0.1 |
