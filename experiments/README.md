# Experiments

Every experiment behind the released skills, in the order they were run, **including the ones that failed or were stopped**. Raw learning curves and logs are in [`runs/`](runs) (one folder per training run) and [`logs/`](logs) (console logs of the longer jobs). Checkpoints are not stored (size); the released policies are in [`../models`](../models) and the evaluation tables in [`../outputs/analysis`](../outputs/analysis).

Conventions: PPO / SAC / TD3 from Stable-Baselines3, 4 CPU environments, 50 Hz policy; "falls /N" = episodes with a fall out of N evaluation episodes (deterministic policy); success = the rule given per phase. Counts from different evaluation sets are not directly comparable, the set size is always stated.

Contents: [Phase 3](#phase-3-standing-and-push-recovery) · [Phase 4](#phase-4-walking) · [Phase 5](#phase-5-running) · [Phase 6](#phase-6-terrain) · [Phase 7](#phase-7-obstacles-and-recovery) · [Skill switching](#skill-switching-phase-7-stages-1-4) · [Vision](#phase-10-vision) · [Race](#phase-8-race-course) · [Dropped](#dropped-experiments-negative-results) · [Lessons](#lessons-that-held-across-phases)

---

## Phase 3: standing and push recovery

Task: stay upright against random horizontal pushes (0.4-1.2 m/s every 1-3 s, 20 s episodes). Doing nothing falls 61 % of episodes. Full PPO table with every hyperparameter: [`../docs/ppo_experiments_log.md`](../docs/ppo_experiments_log.md); full analysis with figures: [`../docs/phase3_standing_analysis.pdf`](../docs/phase3_standing_analysis.pdf).

| Experiment | Runs (data in `runs/`) | Result | Takeaway |
|---|---|---|---|
| PPO, action range 0.25 → 0.5 rad, exploration noise, 500k-2M steps | `ppo_push_500k`, `ppo_push_500k_logstd-1`, `ppo_push_1M_scale0.5_logstd-0.8` | best 27 % falls (0.25 rad) → 3 % (0.5 rad), /30 | the wider action range was the biggest single lever |
| PPO seed sweep, same settings | `ppo_1M_final_seed0/1/2` | best 0 / 23 / 27 %, final 7 / 37 / 33 %, /30 | large seed variance |
| PPO variance reduction (batch 4096, clip 0.1, 5 epochs, LR decay) | `ppo_1M_variance_test_seed0/1/2` | best 13 / 23 / 7 %, final 20 / 40 / 7 % | spread did not clearly narrow; a 90 % collapse seen in the sweep did not recur |
| PPO bigger batch + 2M steps | `ppo_2M_final_seed3` | best 17 %, final 40 % | no breakthrough; PPO tuning stopped after 9 runs |
| SAC, 1M steps | `sac_push_1M_seed0` | best 7 % at 450-500k, final 53 %, /30 | fastest to a good policy, but oscillates strongly late in training |
| SAC recipe v2, shorter training episodes v3 | `sac_push_1M_v2_seed0/1`, `sac_push_1M_v3_ep250_seed0` | v2 seed 1 and v3 stopped early (laptop battery); held-out 15-21 % | not better than the first SAC (p < 0.01 worse) |
| PPO trained on wider pushes (to 1.6 m/s) | `ppo_wide1p6_seed0/1/2` | held-out 200 seeds: 1.0 / 32.5 / 0.5 % | wider training pushes helped two of three seeds |
| TD3, 1M steps, 2 seeds | `td3_push_1M_seed0/1` | held-out 200 seeds: 4.5 % and 2.0 % | strongest family; with a 0.2 s passive warm-up **0 / 200** |
| Held-out comparison of all of the above | `eval_heldout` | on 200 fresh seeds: SAC 5.5 %, PPO best 9.5 % (not significant), v2/v3 worse | the 30-seed numbers are optimistic (winner's curse), always re-test on fresh seeds |

Further findings (see the analysis PDF): most "start-up falls" were shallow 3-10 mm calf touches, a passive warm-up of zero action fixes them for near-home-stance policies, SAC is more push-robust than PPO beyond the training range (1.4 m/s: PPO 48 % falls, SAC 5-17 %).
**Released:** `models/standing_policy_td3_seed0.zip` (with warm-up).

## Phase 4: walking

Task: follow a velocity command (vx 0-0.6 m/s, yaw rate ±0.5 rad/s), no pushes, 3 seeds × 1M steps per option, 50 fresh evaluation episodes. Table: [`../docs/walking_options_comparison.md`](../docs/walking_options_comparison.md).

| Option | Runs | Falls /50 per seed | Steps to learn walking | Verdict |
|---|---|---|---|---|
| First reward only (tracking, slip penalty, capped air-time bonus), no gait structure | `walk_ppo_1M_seed0`, `walk2_ppo_1M_seed0` | 1 and 0 falls /16 (training evaluation) | | the starting point the three options improve on |
| A: gait clock + contact-schedule reward | `walk3_clock_1M_seed0/1/2` | 0, 0, 0 | 450-500k | works |
| **B: sine reference foot path + small learned residual** | `walk4_sine_1M_seed0/1/2` | 0, 0, 0 | **200-250k** | **released**, half the steps of A |
| C: reward-only gait fixes | `walk5_gait_1M_seed0/1/2` | 19, 11, 0 | never learned a trot | 1-2 feet almost never touched the ground; the return hid this |

**Released:** `models/walking_policy_sine_seed1.zip`.

## Phase 5: running

A probe first: leg L1 = L2 = 0.213 m; a linear Jacobian is off by 4-10 cm at strides of 0.12-0.20 m so exact two-link IK is needed; PD torque saturates at 0.24 / 0.36 rad error; the reference alone tracks 1.0-2.0 m/s but with no flight. Adding a 4 cm stance push-off and a speed-dependent gait table gave 23-36 % flight with the reference alone ([`runs/`](runs): `../outputs/analysis/running/reference_sweep*.csv`).
Pipeline, 3 seeds each, warm-started: stage 1 (0-1.5 m/s, 1M steps) → stage 2 (0-2.0, 600k) → stage 3 (0-2.5, 600k). Table: [`../docs/running_options_comparison.md`](../docs/running_options_comparison.md).

| Stage | Runs | Falls | Speed reached | Flight share |
|---|---|---|---|---|
| 1.5 m/s | `run1_1p5_seed0/1/2` | 0-1 /50 | 93-96 % | |
| 2.0 m/s | `run2_2p0_seed0/1/2` | 0-1 /50 | 93-98 % | |
| 2.5 m/s | `run2_2p5_seed0/1/2` | 1, 3, 4 /50 | 94-96 % | 28-35 % |

**Released:** `models/running_policy_2p5_seed0.zip` (+ the gait table, which is part of the policy). Not covered: pushes, uneven ground, > 2.5 m/s.

## Phase 6: terrain

Generator: 4 cm height field (rough ≤ 10 cm, slopes ±3-15°, stairs 2-15 cm, hurdles). Baselines on the walking policy (20 episodes per cell, success = x ≥ 6 m): rough fine to 6 cm, 10 cm breaks; slope up to 9-12° at 0.6 m/s, 15° falls; stairs up blocked from 6 cm; stairs down fall from 8 cm ([`../outputs/analysis/terrain/baseline.md`](../outputs/analysis/terrain/baseline.md)). A first baseline without heading hold was distorted by sideways drift (log: `logs/terrain_baseline_nohold_partial.log`).

| Experiment | Runs | Result |
|---|---|---|
| **Skill A**: blind walking, mixed terrain with a per-type curriculum, 500k pilot | `terr_pilot_seed0` | no cell worse than the walking policy; slope up 15° 0 → 100 % |
| Skill A, 2M steps × 3 seeds | `terr_2M_seed0/1/2` | in-scope success 100 / 98 / 99 % (0.4 m/s) vs walking policy 80 %; stairs up ≥ 8 cm unsolved; only weak in-scope cell is rough 10 cm |
| Foot-lift test (8 / 10 / 12 cm) | `terr_lift12_seed2`, `../outputs/analysis/terrain/lift_*` | higher lift speeds learning; stairs up 8 cm 0 → 70 %; stairs ≥ 10 cm still 0 |
| **Skill B**: add a height scan to the observation | `scan_pilot_seed2` | did not learn tall stairs (≥ 8 cm: 0 %), the policy barely used the scan |
| Skill B, scan-aware reference gait | `scan_gait_pilot_seed2`, `../outputs/analysis/terrain/scan_gait_reference*.csv` | first version exceeded the leg's reach (calf joint limit at ≈ 19 cm); fixed version: stairs up 8 cm 60-100 %, but slope up 15° fell to 0 %. Not better overall, line stopped |

**Released:** `models/terrain_policy_seed2.zip` (+ the 12 cm lift variant).

## Phase 7: obstacles and recovery

**Hurdle while running.** Target: 10-25 cm full-width bars (8 cm thick) at 1.3-1.7 m/s.

| Experiment | Runs / data | Result |
|---|---|---|
| Scripted jump from a run (height-only feasibility, then real crossing) | `logs/hurdle_search_015_20.log` | clears 24-43 cm in the height-only test, but flies only 0.05-0.5 m; the 0.65 m long robot needs ≈ 0.8-1 m of flight, so it hit the bar |
| Learned trigger for the scripted jump, 10 cm start | `hurdle_pilot_seed0`, `hurdle_curr_seed0` | 0 successes at 10 cm; with a low-bar curriculum stuck at 3 cm (100 → 60 → 80 %); two environment bugs found and fixed on the way (false "hit" on the flat floor, random triggers) |
| **Step over instead of jumping** (high-stepping reference + far scan) | `stepover_seed0`, `stepover_cont_seed0`, `models/hurdle_stepover_eval.csv` | 3 cm 100 %, 5 cm 90 %, **7 cm 80 %**, 10 cm 35 %, ≥ 15 cm ≈ 0 % (20 episodes each); a 2M-step continuation was worse |

**Recovery from a bad start** (paused on purpose: a race does not need a full get-up). `rec1_4M_seed0`: baseline "go to the home pose" succeeds 97 / 69 / 45 % for the three start classes; the first run learned to flip on purpose (additive penalties), after the fix PPO's noise collapsed and success stayed at baseline, so the skill was paused.

## Skill switching (Phase 7, stages 1-4)

Task: one supervisor hands control between the five skills while the robot keeps moving. Design, rules and every table: [`../docs/skill_switching.md`](../docs/skill_switching.md). Raw output: [`skills/switching_transitions.log`](skills/switching_transitions.log), [`skills/switching_patch_trials.log`](skills/switching_patch_trials.log), tables in [`../outputs/analysis/switching*`](../outputs/analysis).

| Experiment | Result | Takeaway |
|---|---|---|
| Stage 1: flat hand-overs stand / walk / run, 50 episodes each | all pairs 0 falls except **run → stand: 20 / 50** (warm-up 0), **39 / 50** (warm-up 10) | the stand warm-up makes it worse; falls within 0.3-0.5 s |
| Stage 2: run → stand through a walk brake | walk told to stop (0.0): 16-20 / 50 falls; **walk told 0.4 m/s until < 0.6 m/s: 0 / 50** (also 2 / 150 on fresh seeds) | a walk policy told to *stop* from running speed falls like stand; told to walk slowly it brakes safely |
| Stage 3: rule supervisor, random schedules | first test 3 / 300 falls, fixed with a 0.4 m/s command floor → 0 / 300; **fresh seeds 2 / 450** | the brake itself still tips about 1 in 150 at about 2.4 m/s; targets reached 97-99 % |
| Stage 4b: terrain and step-over hand-overs | 0 falls except walk → step-over 1 / 50 and **step-over → stand 5 / 50** | step-over → stand needs the brake route like run |
| Stage 4c: one-patch trials, oracle hint (20 per cell) | supervisor = terrain-only on terrain patches (both 100 % except rough 10 cm 75 / 65 %), walk-only 0 % on slope up 15°; bars: 85 / 80 / 50 / 30 % at 3 / 5 / 7 / 10 cm | switching helps against the plain walk policy, not against the terrain skill; failures at the bar are mostly "stuck", not falls |

## Phase 10: vision

Task: replace the simulator's height scan and the oracle terrain hint with an onboard depth camera, an elevation map and a learned classifier. Details and every table: [`../docs/vision.md`](../docs/vision.md). Raw logs in [`skills/`](skills) (`vision_round*_*.log`), tables in [`../outputs/analysis/vision/`](../outputs/analysis/vision), data (git-ignored) in `data/vision{,2,3}`.

| Experiment | Result | Takeaway |
|---|---|---|
| V1: camera on the trunk | works; a stale-terrain bug found (the renderer uploads the height field once) | re-upload after each terrain change |
| V2: elevation map from depth (no training), bar trials 40 per cell | map RMS 0.5-2.3 cm; bar success camera vs simulator: 90 / 85 / 50 / 22 % vs 92 / 85 / 62 / 30 % at 3 / 5 / 7 / 10 cm; 96 x 72 pixels no better | the camera map is about as good up to 5 cm |
| V3 round 1: CNN, bars always run, others walked | test frames 83.1 %; closed loop 68 % (oracle 84 %) | **inflated**: the network used "running = bar" as a shortcut |
| V3 round 2: every kind walked and run | test frames 69.4 % (running frames 52 %); closed loop 66 % | honest number; a single frame is weak while running |
| V3 round 3: + body gravity vector as input | test frames **73.4 %**; closed loop **73 %** (vote of 5: 74 %) | best; the bar and rough 10 cm remain weak |

## Phase 8: race course

Task: the supervisor drives a random 4-section course (20-30 m) from standing to a finish line. Details: [`../docs/race.md`](../docs/race.md). Raw logs: [`race/`](race), tables in [`../outputs/analysis/race/`](../outputs/analysis/race).

| Experiment | Result (40 courses, seeds 4000-4039) | Takeaway |
|---|---|---|
| Oracle hint | **92 %** finished, 28.5 s mean, 1 fall / 2 stuck | the supervisor and skills are fine with a correct hint |
| Vision hint | **58 %** finished (8 falls, 6 left the course, 3 stuck) | the single-patch result (73 %) does not carry over |
| Vision hint smoothed over 10 / 25 steps | 62 % / 58 % | **no gain**: the hint flickering was not the cause |
| Per-step diagnosis of the classifier in races | flat truth called flat only **12 %** of the time; flat floor is called "flat" 82 % of the time within 0.2 m of the centre line but only 8 % at 0.6-0.8 m off it | **root cause**: sideways drift shows the camera the side edge of the 3 m wide field (a "drop"); training never saw it |
| Probe with a perfectly level standing robot | said "stairs down" everywhere | not informative (out-of-distribution pose), kept as a record |

**Not done:** centre the robot on the course; retrain the classifier with sideways and heading offsets; a wider field.

## Dropped experiments (negative results)

- **Standing-start RL jump** (literature recipe: projectile-densified reward, reference state initialisation, vertical → forward → obstacle curriculum, 5 Hz filtered actions). Run 1 hopped repeatedly to farm the in-flight reward (my design flaw; the literature gives one jump per episode). Run 2 with one jump per episode took off to the right height (7-12 cm apex vs a 10 cm target in all 8 test episodes) but landed crumpled or tilted (success ≈ 2 % after 1.3M steps, curve flat). Landing terms from the Jumper paper (orientation, desired joint pose, catch landing) were added in a third run that was stopped before producing results. The code was removed from the repository; this is the whole result.
- **Scripted jump primitive** (`racevla/envs/jump_primitive.py`, parameters in `models/jump_primitive_hurdle.json`): kept because the hurdle environment imports it, not used by the released skill.
- **Published Go1 jumping and parkour results use 4096 GPU environments;** this project used 4 CPU environments, which is the likely reason these targets were out of reach.

## Lessons that held across phases

1. A reference motion plus a small learned residual beat pure reward shaping for gaits (walking: 200-250k steps vs never; running and step-over reuse it).
2. Always evaluate on fresh seeds that were not used to select the checkpoint: 30-episode numbers overstated results (standing: winner's curse).
3. Compare on several seeds before drawing conclusions; single-seed differences were often noise (PPO, SAC, terrain, step-over).
4. Several "failures" were bugs, not learning problems: raw instead of normalised observations, random initial headings in speed measurements, a hit rule that counted the flat floor, a reference that exceeded the leg's reach. Check the environment before tuning the algorithm.
5. Report the failures: they mark the real limits (7 cm step-over, no tall stairs, no jump).
6. A learned classifier can use a shortcut: the first terrain classifier scored 83 % because bar episodes were always run (speed gave the answer); it fell to 69 % once both gaits were in every class. Check what correlates with the label besides the thing you want.
7. Test in the conditions of use: the classifier was good on short single-patch episodes (73 %) and bad on long races (35 % on 7 classes) because the robot drifts sideways there. A result on one setup says little about another.
8. Keep an oracle baseline: running the same task with the true hint (race: 92 %) showed that the failures were vision, not the supervisor or the skills, and ruled out a long list of other suspects.
9. Test the first explanation before building on it: smoothing the hint was a cheap fix for a plausible cause and did nothing; only logging every step found the real one.

Reproduce a run: see the exact command in the matching model card in [`../models`](../models) or the script docstring under [`../scripts`](../scripts).
