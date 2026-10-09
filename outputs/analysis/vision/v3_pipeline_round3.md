Patch trials (30 per cell), success % (falls / stuck), and agreement of the hint with the oracle hint (share of steps):

| patch | oracle | vision x1 | vision x5 |
|---|---|---|---|
| rough 0.06 | 100 % (0 / 0) | 97 % (1 / 0), hint agrees 69 % | 97 % (0 / 1), hint agrees 68 % |
| rough 0.1 | 67 % (8 / 2) | 33 % (15 / 5), hint agrees 69 % | 43 % (13 / 4), hint agrees 63 % |
| slope_up 9 | 100 % (0 / 0) | 100 % (0 / 0), hint agrees 81 % | 100 % (0 / 0), hint agrees 80 % |
| slope_up 15 | 100 % (0 / 0) | 73 % (8 / 0), hint agrees 83 % | 93 % (2 / 0), hint agrees 85 % |
| slope_down 15 | 100 % (0 / 0) | 100 % (0 / 0), hint agrees 75 % | 100 % (0 / 0), hint agrees 70 % |
| stairs_up 0.04 | 100 % (0 / 0) | 100 % (0 / 0), hint agrees 56 % | 100 % (0 / 0), hint agrees 55 % |
| stairs_down 0.06 | 100 % (0 / 0) | 100 % (0 / 0), hint agrees 59 % | 100 % (0 / 0), hint agrees 72 % |
| hurdle 0.03 | 90 % (0 / 1) | 77 % (0 / 0), hint agrees 22 % | 77 % (0 / 0), hint agrees 23 % |
| hurdle 0.05 | 87 % (2 / 1) | 70 % (4 / 2), hint agrees 28 % | 73 % (2 / 2), hint agrees 26 % |
| hurdle 0.07 | 57 % (6 / 6) | 40 % (6 / 5), hint agrees 30 % | 20 % (11 / 5), hint agrees 26 % |
| hurdle 0.1 | 23 % (10 / 11) | 13 % (15 / 9), hint agrees 38 % | 7 % (19 / 9), hint agrees 40 % |
| **all cells** | **84 %** | **73 %** | **74 %** |
