# Skill switching (Phase 7, stages 1-4)

Written 2026-10-09. Code: [`racevla/skills/`](../racevla/skills) · tests and trials: [`scripts/phase7_skills/20-24`](../scripts/phase7_skills) · raw results: [`outputs/analysis/switching*/`](../outputs/analysis).

The five skills (stand, walk, run, blind terrain, step-over) were trained separately. This phase asks: **can one supervisor hand control from one skill to another while the robot keeps moving, and decide which skill to use?** Everything here is simulation, deterministic policies, and (until the vision phase) a terrain hint read from the simulator ("oracle", see [`docs/vision.md`](vision.md)).

## Why a hand-over is not trivial

A skill is more than its network. Each one expects its own *simulator settings*:

| Skill | Action scale | Leg reference | Observation | Extras |
|---|---|---|---|---|
| Stand (TD3) | 0.5 rad | none | 45 | 0.2 s passive warm-up |
| Walk (PPO) | 0.25 rad residual | sine foot path, linear Jacobian, 2.5 Hz, 8 cm lift | 49 (45 + command + gait clock) | |
| Run (PPO) | 0.25 rad residual | gait table, exact leg IK, push-off | 49 | |
| Terrain (PPO) | 0.25 rad residual | the walking reference | 49 | heading hold (turn = -2 x yaw error, clipped ±0.5) |
| Step-over (PPO) | 0.25 rad residual | running reference + foot lift from the height map (look-ahead 0.32 m, cap 17.5 cm) | 75 (49 + 24 far height scan + 2 jump flags, always 0) | heading hold |

## What was built

- [`SwitchBody`](../racevla/skills/body.py): one simulated Go1 whose mode (`stand / walk / run / terrain / stepover`) can change while the episode runs. Only the settings above change; joint angles, velocity and the gait clock carry on, so a switch is a real hand-over. `FlatSwitchBody` (infinite floor, scan = zeros), `TerrainSwitchBody` (the 10 m terrain course) and `RaceSwitchBody` (the 46 m race course, see [`docs/race.md`](race.md)) add the terrain modes.
- [`Skill` / `Supervisor`](../racevla/skills/library.py): a skill = network + its simulator mode + how to build its observation + its warm-up. The supervisor asks the active skill for an action every step and can hand over at any moment. `Supervisor.go()` is the safe route to stand (below). `Supervisor.vision` is the hook for the camera map ([`docs/vision.md`](vision.md)).
- [`RuleSupervisor`](../racevla/skills/rules.py): you give it a target (forward speed, turn rate) and optionally a terrain hint at any time; it decides which skill is active, using only rules (no learning).

## The rules

- Stand when the target is about zero (below 0.10 m/s; leaving stand needs 0.15). Walk for small targets, run above 0.70 m/s (back to walk below 0.50): a gap between "on" and "off" so a target hovering near a threshold does not flip the skill.
- At least 25 steps (0.5 s) between two switches.
- A terrain hint (`rough`, `slope_up`, `slope_down`, `stairs_up`, `stairs_down`) selects the terrain skill (0.4-0.8 m/s). The hint `bar` selects the step-over skill (1.3-1.7 m/s), **entered only from run**.
- Routes that were tested and are safe: stand ↔ walk, walk ↔ run, run → walk, walk / stand / run ↔ terrain, run ↔ step-over, step-over → walk. **Run → stand and step-over → stand only through walk** (the brake route). Stand → run goes through walk; step-over → terrain goes through walk.
- Stall escape: when the step-over skill is wanted but the robot is stopped (a robot standing at a bar can never reach the 0.4 m/s that walk → run needs), after 50 steps it goes to step-over directly. Found when the vision hint deadlocked a robot in front of the bar.
- On a bounded course, `hold_heading=True` gives walk and run the same heading hold the terrain skills have (the run policy otherwise drifts sideways, see [`docs/race.md`](race.md)).

## Results

All numbers are falls or successes in simulation, deterministic policies. "Falls after switch /50" means episodes where the robot fell after the hand-over; no episode fell before the switch.

### Stage 1: flat ground, stand / walk / run (50 episodes per pair, seeds 7000+)

[`outputs/analysis/switching/transitions.md`](../outputs/analysis/switching/transitions.md)

| Hand-over | Falls after switch /50 | Settle median (s) |
|---|---|---|
| stand → walk | 0 | 0.62 |
| walk → stand | 0 (warm-up 0 and 10) | 0.32 / 0.40 |
| walk → run | 0 | 0.43 |
| run → walk | 0 | 0.72 |
| controls (same skill) | 0 | 0 |
| **run → stand** | **20** (warm-up 0), **39** (warm-up 10) | 0.93 / 1.04 |

Run → stand breaks, and the stand warm-up makes it *worse* (the robot falls inside 0.3-0.5 s, reason: tilt). The fall rate rises with the speed at the switch. A likely reason is that the stand policy never saw body speeds like a run's, but this was not tested separately.

### Stage 2: braking route for run → stand

[`outputs/analysis/switching_brake/transitions.md`](../outputs/analysis/switching_brake/transitions.md): the walk policy takes over first and is asked for **0.4 m/s** until the speed has stayed below 0.6 m/s for 5 steps; then stand takes over.

| Route | Falls /50 |
|---|---|
| direct | 20 |
| via walk, command 0.0, until < 0.6 m/s | 16 |
| via walk, command 0.0, until < 0.3 m/s | 20 |
| **via walk, command 0.4, until < 0.6 m/s** | **0** (also with warm-up 10) |

The lesson: a walk policy told to *stop* from running speed falls just like stand does; told to *walk slowly* it brakes safely. Brake time median 0.36 s (max 0.58 s). On 150 fresh seeds (7050-7199) the 0.4 route fell 2 times (the two falls were not examined).

### Stage 3: rule supervisor on flat ground

[`outputs/analysis/switching_supervisor/`](../outputs/analysis/switching_supervisor) and [`_fresh/`](../outputs/analysis/switching_supervisor_fresh): random schedules (slow = 6 segments of 3 s, fast = 10 of 1.5 s, turns) with targets stand / walk U(0.2, 0.6) / run U(1, 2.5).

- First test (seeds 8000+, 100 episodes per set): 3 falls in 300, all "run at 2.2-2.45 m/s → walk with a target of 0.2-0.29 m/s" (walk brakes too hard). Fix: while walking faster than 0.6 m/s, the command has a floor of 0.4. Same seeds afterwards: **0 falls in 300**.
- Fresh seeds (9000+, 150 episodes per set): **2 falls in 450** (slow 1, fast 0, turns 1), both "run at about 2.4 m/s, then target stand": the brake itself tips (tilt 44° at the moment stand takes over). Target reached at the end of the segment: 99 % (slow), 98 % (fast), 99 % (turns); 7.2 switches per episode in the slow set, 12.2 in the fast set.

### Stage 4a: do the terrain skills behave inside the switching body?

[`22_check_terrain_skills.py`](../scripts/phase7_skills/22_check_terrain_skills.py), 20 episodes per cell: the terrain skill matches its model card (rough 6 / 8 / 10 cm: 100 / 95 / 90 %, slopes ±15°: 100 %, stairs up 4 cm / down 6 cm: 100 %); the step-over skill gets 100 / 95 / 95 / 85 / 30 % for no bar / 3 / 5 / 7 / 10 cm (card: 100 / 100 / 90 / 80 / 35).

### Stage 4b: hand-overs involving the terrain and step-over skills (flat floor, 50 episodes)

[`outputs/analysis/switching_terrain/transitions.md`](../outputs/analysis/switching_terrain/transitions.md): 0 falls for walk ↔ terrain, stand → terrain, terrain → stand / run, run → terrain, run ↔ step-over, step-over → walk / run. Exceptions: walk → step-over 1 / 50, and **step-over → stand 5 / 50** (so step-over → stand uses the brake route like run). Note: the "never settled" counts for the terrain skill (up to 11 / 50, also for terrain → terrain) are an artefact of the ±20 % settle test at low commands, not a stability problem.

### Stage 4c: one-patch trials with an oracle terrain hint (20 episodes per cell, seeds 3000+, robot starts at x = -0.5 m)

[`outputs/analysis/switching_patches/summary.md`](../outputs/analysis/switching_patches/summary.md)

| Patch | Supervisor | Walk only | Terrain only |
|---|---|---|---|
| rough 6 cm | 100 % | 100 % | 100 % |
| rough 10 cm | 75 % | 45 % | 65 % |
| slope up 9° | 100 % | 60 % | 100 % |
| slope up 15° | 100 % | **0 %** (20 falls) | 100 % |
| slope down 15°, stairs up 4 cm, stairs down 6 cm | 100 % | 100 % | 100 % |

| Bar | Supervisor | Run only | Step-over only |
|---|---|---|---|
| 3 cm | 85 % | 90 % | 100 % |
| 5 cm | 80 % | 80 % | 70 % |
| 7 cm | 50 % | 20 % | 40 % |
| 10 cm | 30 % | 5 % | 35 % |

What this shows, honestly:
- Switching beats the plain walking policy on slopes, which is what a supervisor is for.
- **The terrain skill alone is as good as the supervisor on terrain patches**: no extra benefit of switching was shown there.
- On the bar the supervisor is about as good as the step-over skill alone; most failures are "stuck at the bar" (the robot stops at x ≈ 5.0-5.2 m), not falls.
- The step-over skill is weaker in these trials than its model card: the card's seeds (2000-2019) were the checkpoint-selection seeds (winner's curse) and the trials start 0.5 m further back. At the card's start with its own seeds: 5 cm 20 / 20, 7 cm 18 / 20; at the trials' start with fresh seeds: 14 / 20 and 8 / 20.

## Known limits

- Run → stand through the brake still falls about 1 time in 150 at about 2.4 m/s. Ideas not tried: decelerate inside the run policy before handing to walk; a tilt guard before the stand takeover.
- Seed 3015 never left the start in the bar trials (a run stall that was not investigated).
- All trials use a single patch on a flat floor; long courses are in [`docs/race.md`](race.md).
- The rules are hand-written thresholds; nothing here was tuned beyond the fixes listed.

## Reproduce

```bash
python scripts/phase7_skills/20_test_transitions.py --set direct|brake|terrain    # stages 1, 2, 4b
python scripts/phase7_skills/21_test_supervisor.py --seed-base 9000               # stage 3
python scripts/phase7_skills/22_check_terrain_skills.py                           # stage 4a
python scripts/phase7_skills/24_test_patch_trials.py --episodes 20                # stage 4c
```
