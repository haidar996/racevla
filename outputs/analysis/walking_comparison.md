| Option | seeds done | falls /50 (per seed) | fwd error m/s | speed reached | turn error rad/s | turn achieved vs 0.40 | feet on floor (mean of 4) | foot slip | lift cm | steps to learn walking well |
|---|---|---|---|---|---|---|---|---|---|---|
| A  clock + schedule reward | 3 | 0, 0, 0 | 0.034 | 95% | 0.060 | 0.37 | 0.65 | 0.12 | 7.6 | 500k, 450k, 450k |
| B  sine reference + residual | 3 | 0, 0, 0 | 0.034 | 95% | 0.057 | 0.36 | 0.68 | 0.11 | 7.4 | 200k, 200k, 250k |
| C  reward-only fixes | 3 | 19, 11, 0 | 0.114 | 80% | 0.228 | 0.22 | 0.43 | 0.28 | 17.3 | never, never, never |

Time each foot (FR FL RR RL) is on the floor, per seed (a trot is about 0.6 for all four):
- A  clock + schedule reward: 0.65 0.66 0.70 0.65 | 0.64 0.60 0.69 0.65 | 0.64 0.66 0.66 0.67
- B  sine reference + residual: 0.65 0.69 0.72 0.67 | 0.67 0.67 0.67 0.70 | 0.68 0.66 0.69 0.69
- C  reward-only fixes: 0.49 0.76 0.76 0.06 | 0.91 0.12 0.22 0.89 | 0.35 0.06 0.08 0.47
