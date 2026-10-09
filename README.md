# RaceVLA

[![tests](https://github.com/haidar996/racevla/actions/workflows/tests.yml/badge.svg)](https://github.com/haidar996/racevla/actions/workflows/tests.yml) ![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green) ![sim](https://img.shields.io/badge/sim-MuJoCo-orange)

**Learning a locomotion skill library for the Unitree Go1 quadruped in MuJoCo, as the foundation for vision-language guided skill switching.**

Five reinforcement-learning skills, trained with PPO / TD3 on a single laptop CPU (4 cores, no GPU), all on the same simulated Go1:

| | | |
|:--:|:--:|:--:|
| ![standing](docs/media/standing.gif) | ![walking](docs/media/walking.gif) | ![running](docs/media/running.gif) |
| **1. Stand + push recovery** | **2. Walk (velocity commands)** | **3. Run up to 2.5 m/s, with flight** |
| ![terrain](docs/media/terrain.gif) | ![stepover](docs/media/stepover.gif) | |
| **4. Blind terrain walking** | **5. Step over a bar while running** | |

On top of the skills there is now a **rule-based supervisor that switches between them**, an **onboard depth camera with a learned terrain classifier**, and a **race course** that tests all of it together:

![race](docs/media/race.gif)

*A race with the camera pipeline (depth image → elevation map and terrain classifier → supervisor): slope up, rough ground, a 5 cm bar, stairs down, finished in 27 s. This is a success case; with the classifier's hint about 58 % of random races finish, with a perfect hint 92 % (see [Switching, vision and the race](#switching-vision-and-the-race)).*

The long-term goal (see [Roadmap](#roadmap)) is a robot that is told in language what to do ("run to the bar, step over it, stop") and a vision-language model that picks the right skill. **The language part is not built yet.** This repository contains the skills, the supervisor, the vision pipeline, the race, the environments, the trained policies and every number behind them, including what did **not** work.

## Results at a glance

All numbers are simulation, deterministic policies, on fresh random episodes (standing: 200 seeds never used for checkpoint selection; walking 50, running 50, terrain 20 per cell, step-over 20 per bar height). The model cards in [`models/`](models) have the full tables.

| Skill | Policy | Result | Scope / honest limits |
|---|---|---|---|
| **Stand + push recovery** | TD3, 1M steps, [`standing_policy_td3_seed0.zip`](models/standing_policy_td3_seed0.zip) | **0 / 200 falls** (20 s episodes, pushes 0.4-1.2 m/s every 1-3 s; doing nothing falls 61 %) | needs a 0.2 s passive warm-up; never trained to get up after a fall |
| **Walking** | PPO, 1M steps, sine gait reference + learned residual, [`walking_policy_sine_seed1.zip`](models/walking_policy_sine_seed1.zip) | 0 / 50 falls, speed error 0.03 m/s, 96 % of commanded speed reached, turning 0.36 of 0.40 rad/s asked | 0-0.6 m/s, flat ground, no pushes |
| **Running** | PPO, 3-stage speed curriculum 1.5 → 2.0 → 2.5 m/s, [`running_policy_2p5_seed0.zip`](models/running_policy_2p5_seed0.zip) | 1 / 50 falls, 94 % of commanded speed, **28 % of the time all four feet are off the ground** (commands ≥ 1 m/s) | flat ground, no pushes; the 2.5 m/s gait table row is extrapolated |
| **Blind terrain walking** | PPO, 2M steps on a mixed curriculum, [`terrain_policy_seed2.zip`](models/terrain_policy_seed2.zip) | 99 % mean success in scope: rough ground ≤ 10 cm, slopes ±15°, stairs up ≤ 4 cm, stairs down ≤ 6 cm (the plain walking policy: 80 % at 0.4 m/s, 94 % at 0.6 m/s) | proprioception only: it cannot see steps, so tall stairs up (≥ 8 cm) are not solved; no pushes |
| **Step over a bar while running** | PPO with a far height scan, [`hurdle_stepover_seed0.zip`](models/hurdle_stepover_seed0.zip) | at 1.3-1.7 m/s: 3 cm 100 %, 5 cm 90 %, **7 cm 80 %**, 10 cm 35 %, ≥ 15 cm ~0 % (20 episodes each) | reliable to about 7 cm; the original 10-25 cm target was **not** reached (see below); the checkpoint was picked on the same 20-episode evaluation, so these numbers are slightly optimistic |

### Beyond single skills (simulation, deterministic; full tables in [`docs/`](docs))

| Part | Result | Honest limits |
|---|---|---|
| **Skill switching** ([`docs/skill_switching.md`](docs/skill_switching.md)) | rule supervisor over stand / walk / run / terrain / step-over: **0 falls in 300** flat random schedules, 2 in 450 on fresh seeds; targets reached 97-99 % | run → stand needs a walk "brake" (direct: 20-39 / 50 falls); the brake still tips about 1 in 150 at about 2.4 m/s; on terrain patches the terrain skill alone is as good as the supervisor |
| **Elevation map from the camera** ([`docs/vision.md`](docs/vision.md)) | 64 x 48 depth image → 4 cm map, error 0.5-2.3 cm; step-over with the camera map: 3 / 5 / 7 / 10 cm bar 90 / 85 / 50 / 22 % (simulator's map: 92 / 85 / 62 / 30 %) | camera pose and foot heights still come from the simulator; slightly worse above 5 cm |
| **Terrain classifier from one depth image + gravity vector** ([`models/terrain_classifier_final.md`](models/terrain_classifier_final.md)) | held-out frames **73.4 %** (7 classes); one-patch trials **73 %** average success vs **84 %** with the simulator's hint | trained from scratch on my own simulator data; weak at the bar and at rough 10 cm; **does not carry over to long courses** (below) |
| **Race course** ([`docs/race.md`](docs/race.md)) | 40 random 4-section courses (20-30 m): **92 %** finished with the oracle hint, **58 %** with the classifier's hint | the vision failures are caused by sideways drift (the camera sees the field's side edge as a drop); not fixed |

## Skill interfaces

Each skill is a separate policy with its own command input. This is what a supervisor has to provide (the rule supervisor in [`racevla/skills/rules.py`](racevla/skills/rules.py) does it):

| Skill | Command it takes | Observation | Action | Notes |
|---|---|---|---|---|
| Stand + push recovery | none | 45 | 12 joint offsets | 0.2 s passive warm-up first |
| Walking | forward speed 0-0.6 m/s, yaw rate ±0.5 rad/s | 49 (45 + command + 2 gait-clock values) | 12 | stands still when told 0 |
| Running | forward speed 0-2.5 m/s, yaw rate ±0.5 rad/s | 49 | 12 | needs `models/running_gait_table.json` |
| Blind terrain walking | forward 0.4-0.8 m/s + heading hold | 49 | 12 | 20 % flat, rest rough / slopes / stairs in training |
| Step over a bar | forward 1.3-1.7 m/s + far height scan | 75 (49 + 24 scan + 2 unused) | 12 | the scan comes from the simulator height map  or, in the vision pipeline, from the camera elevation map |

Observations are normalised with `FixedObsNormalize`; a raw observation makes any policy meaningless. The skills also need their own simulator settings (action scale, leg reference, gait table); [`racevla/skills/`](racevla/skills) does the switching of those settings, see [`docs/skill_switching.md`](docs/skill_switching.md).

## How it works

- **Robot and simulator.** Unitree Go1 (the MuJoCo model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), Unitree's BSD-3 license; `scene_terrain.xml` is added here), 12 actuated joints, 500 Hz physics, 50 Hz policy, PD joint control (Kp 100, Kd 0.5) with the real torque limits.
- **Actions and observations.** The policy outputs joint-angle offsets; the 45-number robot observation (joint positions/velocities, gravity vector, body velocities, last action) is scaled by *fixed* physical limits ([`FixedObsNormalize`](racevla/envs/wrappers.py)) instead of running statistics, so policies from different seeds see identical inputs.
- **Reference + residual for gaits.** Pure reward shaping never produced a clean trot (option C in [`docs/walking_options_comparison.md`](docs/walking_options_comparison.md): 0 of 3 seeds learned it). What worked: a gait clock drives a sine-shaped foot trajectory (exact two-link leg inverse kinematics), and the network learns a small correction on top. It learned to walk in about 200-250k steps instead of never.
- **Running = the same idea, with a speed-dependent gait table** (step rate 2.5 → 4 Hz, longer swing, 4 cm stance push-off) and a speed curriculum warm-started from the walking policy. The flight phase appears as a by-product, it is not rewarded as a goal.
- **Terrain.** A MuJoCo height field generator (rough ground, slopes, stairs up/down, hurdles) and a per-terrain-type curriculum: each difficulty level unlocks at ≥ 80 % success. All terrain types are mixed within one training run.
- **Seeing a bar.** The step-over skill gets a coarse far scan of the ground height (8 rows, 0.2-1.6 m ahead, max-pooled so a thin bar is never skipped) and a reference gait that lifts each swing foot to the highest ground within 0.32 m ahead.

- **Switching.** A `SwitchBody` is one simulated Go1 whose mode (stand / walk / run / terrain / step-over) changes while the episode runs: the action scale, leg reference and observation layout change, joints, velocity and gait clock carry on. A `RuleSupervisor` takes a target (speed, turn rate) and an optional terrain hint and picks the skill with hysteresis, a 0.5 s minimum dwell and only routes that were tested (run and step-over reach stand through a walk brake at 0.4 m/s). [`racevla/skills/`](racevla/skills).
- **Vision.** A depth camera on the trunk (64 x 48, 5 m); pixels become 3-D points and a 4 cm elevation map that replaces the simulator's height scan for the step-over skill; a 400k-weight CNN on one depth frame plus the body's gravity vector names the terrain (flat, rough, slope up / down, stairs up / down, bar) and thereby replaces the oracle hint. [`racevla/vision/`](racevla/vision).
- **Race.** Random courses of 4 sections on a 46 m height field, each section starting at the previous one's height ([`racevla/envs/race.py`](racevla/envs/race.py)); score = finished / time / sections passed ([`scripts/phase8_race/`](scripts/phase8_race)).

## Switching, vision and the race

What the tests show, in short (every table, with the failures, is in [`docs/skill_switching.md`](docs/skill_switching.md), [`docs/vision.md`](docs/vision.md) and [`docs/race.md`](docs/race.md)):

- Switching works with a correct terrain hint: 37 of 40 random race courses finished (92 %, 28.5 s on average).
- With the learned classifier's hint only 23 of 40 finished (58 %). The classifier scored 73 % on single patches but 35 % (7 classes) during races: as the robot drifts sideways the camera sees the side edge of the 3 m wide field, which looks like stairs down. Smoothing the hint did not help (58 / 62 / 58 %), because flicker was not the cause.
- Not done: keeping the robot centred, retraining the classifier with sideways offsets, a wider field.

## What did not work (kept in the repo on purpose)

- **PPO on standing recovery** had large seed-to-seed variance (0-27 % falls at the best checkpoint across 9 runs); the full log with every hyperparameter is in [`docs/ppo_experiments_log.md`](docs/ppo_experiments_log.md). TD3 and SAC were compared next; SAC (one seed completed) reached the best score fastest but oscillated strongly late in training.
- **Jumping over 10-25 cm hurdles while running.** A scripted jump clears the *height* (24-43 cm) but flies only 0.05-0.5 m forward, and the 0.65 m long robot needs about 0.8-1 m of flight to pass over a bar. A learned trigger for it stalled at 3 cm. I replaced it with the step-over skill above (reliable to 7 cm). A standing-start RL jump following the literature recipe (projectile-densified reward, reference state initialisation, staged curriculum) took off to the right height but never landed upright, so I dropped it for now.
- **A height-scan-aware gait for tall stairs** gave limited gain and was stopped.
- **Run → stand directly** fell in 20-39 of 50 hand-overs (the stand warm-up made it worse); the walk-brake route fixed it, but it still tips about 1 time in 150.
- **The first terrain classifier looked like 83 %** and was a shortcut: bar episodes were always run, so speed gave the answer. With both gaits in every class the honest number was 69 % (73 % after adding the gravity vector).
- **Smoothing the hint** (a new terrain group must be asked for 10 or 25 steps in a row) did not improve the race: the flicker was not the cause.
- **A standing-pose probe of the classifier** said "stairs down" everywhere: a level standing robot is not what it was trained on, so the probe was uninformative (kept in `scripts/phase8_race/03_flat_probe.py` as a record).

The published Go1 results use 4096 parallel GPU environments; this project trains on 4 CPU environments, which is the main reason the jump targets were out of reach.

## Quick start

```bash
git clone https://github.com/haidar996/racevla.git && cd racevla
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# watch a skill in the MuJoCo viewer
python scripts/phase4_walking/03_view_walk.py models/walking_policy_sine_seed1.zip --sine
python scripts/phase5_running/06_view_running.py models/running_policy_2p5_seed0.zip
python scripts/phase6_terrain/06_view_stairs_tour.py --terrains others
python scripts/phase7_skills/18_view_stepover.py

# watch a whole race: the supervisor with the camera pipeline (seed 4015 finishes)
python scripts/phase8_race/05_view_race.py --seed 4015            # --hint oracle, --slow 2

# (viewers need a display; the GIF scripts render headless with EGL)
# regenerate the GIFs above
python scripts/make_demos.py
python scripts/phase8_race/06_record_race.py                       # docs/media/race.gif
```

Switching, vision and the race (off-screen rendering needs `export MUJOCO_GL=egl` before Python imports mujoco):

```bash
export MUJOCO_GL=egl
python scripts/phase7_skills/21_test_supervisor.py                 # random stand / walk / run schedules with the rule supervisor
python scripts/phase7_skills/24_test_patch_trials.py               # one-patch trials with the oracle hint
python scripts/phase10_vision/01_show_camera.py --gif              # what the onboard camera sees
python scripts/phase10_vision/06_eval_vision_pipeline.py --trials 30 --model models/terrain_classifier_cnn_round3.pt --gravity --tag _check
python scripts/phase8_race/01_run_races.py --races 40 --hint oracle --tag _oracle    # or --hint vision [--hold 10]
```

The classifier training data (about 93,000 depth frames per round) is not in the repository; `scripts/phase10_vision/03_collect_depth_data.py` regenerates it (about an hour on 4 cores), see [`docs/vision.md`](docs/vision.md).

Train a skill (each phase folder has the exact scripts used; 4 envs, no GPU needed):

```bash
python scripts/phase4_walking/02_train_walk_ppo.py --env sine --run-name walk_sine_seed1 --seed 1 --steps 1000000
bash scripts/phase5_running/run_pipeline.sh            # 3-stage running curriculum
python scripts/phase6_terrain/05_train_terrain_ppo.py --help   # see 'models/terrain_policy_final.md' for the exact command
```

## Tests

```bash
pip install -e ".[dev]" && pytest -q     # 17 smoke tests, ~10 s: environments, terrain generator, every released policy loads and behaves, race course builder, hint filter, supervisor rules, the 46 m race body, the packaged classifier, the camera elevation map
```

The camera test needs off-screen rendering (EGL / OSMesa) and is skipped where it is not available.

The same tests run in GitHub Actions on every push.

## Repository layout

```
racevla/envs/             Gymnasium environments: standing, walking (clock / sine / reward-only), running, terrain, scan, hurdle, step-over
racevla/envs/terrain.py   height-field terrain generator (rough, slopes, stairs, hurdles); the mixin also takes a ready-made height field
racevla/envs/race.py      random multi-section race course on a 46 m height field
racevla/envs/wrappers.py  FixedObsNormalize
racevla/controllers/      PD joint controller
racevla/skills/           skill library, switching body, rule supervisor, hint filter (stand / walk / run / terrain / step-over)
racevla/vision/           onboard depth camera, elevation map, terrain classifier
assets/robots/unitree_go1 MuJoCo model (with the onboard depth camera) + terrain scene + 46 m race scene
scripts/                  per-phase tests, training, evaluation and viewer scripts (see scripts/README.md)
models/                   released policies and the terrain classifier + a model card (.md) for each: config, how to run, results, known limits
experiments/              catalogue of all experiments + raw learning curves (runs/) and console logs (logs/)
docs/                     experiment logs, option comparisons, switching / vision / race write-ups, figures, demo media
outputs/analysis/         evaluation tables and figures behind the numbers above
tests/                    smoke tests (pytest)
```

## Further reading

- [`experiments/`](experiments): **every experiment in order, including failures**, with learning curves and logs for all 53 run folders
- Model cards (exact configs, commands, full result tables): [`models/*_final.md`](models)
- [`docs/skill_switching.md`](docs/skill_switching.md), [`docs/vision.md`](docs/vision.md), [`docs/race.md`](docs/race.md): the switching supervisor, the camera pipeline and the race, with every table and the failures
- [`docs/ppo_experiments_log.md`](docs/ppo_experiments_log.md): all nine PPO runs on standing recovery
- [`docs/walking_options_comparison.md`](docs/walking_options_comparison.md), [`docs/running_options_comparison.md`](docs/running_options_comparison.md): the design alternatives that were compared, over three seeds each
- Figures: [`docs/figures/`](docs/figures) and [`outputs/analysis/`](outputs/analysis)

## Roadmap

- [x] Phase 1-3: simulator, environments, standing and push recovery (PPO vs SAC vs TD3)
- [x] Phase 4: walking · Phase 5: running with flight · Phase 6: terrain · Phase 7: hurdle step-over
- [x] **Skill switching**: a rule supervisor that hands over between stand / walk / run / terrain / step-over (`racevla/skills/`, [`docs/skill_switching.md`](docs/skill_switching.md))
- [x] **Vision**: onboard depth camera, elevation map and a learned terrain classifier replace the simulator's height scan and terrain hint (`racevla/vision/`, [`docs/vision.md`](docs/vision.md)); works on single patches (73 % vs 84 % with the oracle)
- [x] **Race course**: random multi-section courses driven by the supervisor ([`docs/race.md`](docs/race.md)): 92 % finished with the oracle hint, 58 % with the vision hint
- [ ] **Vision on long courses**: keep the robot centred and / or retrain the classifier with sideways and heading offsets (the diagnosed cause of the 58 %)
- [ ] **VLA** (not started): a language model that turns an instruction ("run to the bar, step over it, then stop") into a plan of speeds, events and stops for the supervisor, evaluated on the race course. The design discussed: a small pretrained instruction model run locally, the plan as JSON, tested first with the oracle hint so that only the language part is measured

## Limitations

Simulation only, one robot, no domain randomization (friction, mass, motor strength), single seed for the step-over skill, and each skill was trained and tested in its own scope.

- **Switching** uses hand-written thresholds and was tested on flat schedules, single patches and 40 random courses; the run-to-stand brake still tips about 1 time in 150. On terrain patches the terrain skill alone is as good as the supervisor.
- **Vision**: depth is rendered (plus added noise), the camera pose and foot heights are exact simulator values (their error was not tested), the classifier sees one frame, and its training data is short single-patch episodes. It solves 73-74 % of single-patch trials (84 % with the simulator's hint) but only **58 % of races** (92 % with the hint), because the robot drifts sideways and the camera then sees the side edge of the field. The bar and rough 10 cm are its weak points. Sample sizes (30 trials per cell, 40 courses) are modest, and the closed-loop seeds were reused to compare classifier rounds.
- **No language model yet**, and nothing was run on a real robot.

See [`models/terrain_classifier_final.md`](models/terrain_classifier_final.md) and [`docs/race.md`](docs/race.md).

## Acknowledgements

[MuJoCo](https://github.com/google-deepmind/mujoco) and [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) (Go1 model), [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (PPO, SAC, TD3), [Gymnasium](https://gymnasium.farama.org/). All training was done on a laptop CPU (i7-6600U, 4 cores).

## License

MIT for the code (see [LICENSE](LICENSE)); the Go1 model keeps its own BSD-3-Clause license (see `assets/robots/unitree_go1/LICENSE`).
