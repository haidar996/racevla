Patch trials (30 per cell), success % (falls / stuck), and agreement of the hint with the oracle hint (share of steps):

| patch | oracle | vision x1 | vision x5 |
|---|---|---|---|
| rough 0.06 | 100 % (0 / 0) | 90 % (3 / 0), hint agrees 63 % | 97 % (1 / 0), hint agrees 62 % |
| rough 0.1 | 67 % (8 / 2) | 33 % (15 / 5), hint agrees 57 % | 43 % (13 / 4), hint agrees 61 % |
| slope_up 9 | 100 % (0 / 0) | 97 % (0 / 1), hint agrees 72 % | 100 % (0 / 0), hint agrees 76 % |
| slope_up 15 | 100 % (0 / 0) | 13 % (25 / 1), hint agrees 74 % | 37 % (18 / 1), hint agrees 73 % |
| slope_down 15 | 100 % (0 / 0) | 100 % (0 / 0), hint agrees 74 % | 100 % (0 / 0), hint agrees 70 % |
| stairs_up 0.04 | 100 % (0 / 0) | 100 % (0 / 0), hint agrees 52 % | 100 % (0 / 0), hint agrees 56 % |
| stairs_down 0.06 | 100 % (0 / 0) | 100 % (0 / 0), hint agrees 57 % | 100 % (0 / 0), hint agrees 65 % |
| hurdle 0.03 | 90 % (0 / 1) | 80 % (0 / 0), hint agrees 21 % | 83 % (0 / 1), hint agrees 28 % |
| hurdle 0.05 | 87 % (2 / 1) | 60 % (4 / 1), hint agrees 23 % | 67 % (3 / 0), hint agrees 25 % |
| hurdle 0.07 | 57 % (6 / 6) | 33 % (10 / 4), hint agrees 25 % | 33 % (12 / 7), hint agrees 24 % |
| hurdle 0.1 | 23 % (10 / 11) | 23 % (13 / 6), hint agrees 34 % | 7 % (13 / 11), hint agrees 35 % |
| **all cells** | **84 %** | **66 %** | **70 %** |
