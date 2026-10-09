Round 1 (first dataset, data/vision: bar episodes always run, every other kind walked; 92,465 frames, 1120 episodes), test set 9487 frames from 112 unseen episodes:
- 7-class accuracy 83.1 %, balanced 80.2 %; 3 groups (flat / terrain patch / bar) 89.7 %.
- Recall: flat 88.5, rough 66.7, slope_up 84.7, slope_down 82.5, stairs_up 67.8, stairs_down 82.2, bar 88.7.
- Flat frames while running (bar episodes before the window): 74.6 % predicted flat, 12.9 % bar; while walking 89.4 % / 3.3 % -> the network used the speed (pose) as a cue for 'bar'.
- Closed loop (v3_pipeline_round1.md): oracle 84 %, vision x1 68 %, vision x5 70 % over all cells.
