"""Go1WalkingClockEnv: Phase 4, option A -- gait CLOCK + contact schedule + swing-height target. Built on Go1WalkingEnv (same commands, physics, fall rule, tracking/slip/smoothness rewards).
The clock is a metronome: it does NOT move the joints. It (1) is shown to the policy as two numbers (sin, cos of the phase) and (2) is used to score the feet: a trot schedule says, at every moment,
which feet should be in the air (swing) and which on the floor (stance); the reward pays for matching it and for lifting the swing foot along a smooth half-sine arc of height SWING_HEIGHT.
The policy still outputs all 12 joint targets itself (learned). The touchdown air-time bonus of Go1WalkingEnv is removed: the schedule replaces it.
Clock runs only while the robot is asked to move; when told to stand it is off (inputs are 0) and all four feet should be on the floor."""
import numpy as np
from gymnasium import spaces

from racevla.envs.go1_walking import Go1WalkingEnv, WALK_OBS_DIM, MOVE_THRESHOLD, W_AIR_TIME

CLOCK_OBS_DIM = WALK_OBS_DIM + 2          # 47 + (sin, cos) = 49

# --- gait (trot) ----------------------------------------------------------------------------------------------
GAIT_FREQUENCY = 2.5                       # cycles per second  (one cycle = 0.4 s = 20 policy steps)
SWING_FRACTION = 0.4                       # share of the cycle a foot is in the air (the other 0.6 it is on the floor)
FOOT_PHASE_OFFSETS = np.array([0.0, 0.5, 0.5, 0.0])   # foot order FR, FL, RR, RL: diagonal pairs (FR+RL) and (FL+RR) are half a cycle apart
SCHEDULE_MARGIN = 0.05                     # no scoring within +-0.05 of the cycle around a swing/stance switch (one policy step = 0.05)
SWING_HEIGHT = 0.08                        # m, peak height of the swing arc

# --- extra reward weights (first guesses) ----------------------------------------------------------------------------
W_SCHEDULE = 0.5                           # fraction of feet whose contact matches the schedule (max +0.5 per step)
W_SWING_HEIGHT = -25.0                     # squared shortfall of the swing foot below its half-sine target arc (a foot on the floor in mid-swing costs about 0.16)


class Go1WalkingClockEnv(Go1WalkingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = spaces.Box(-np.inf, np.inf, (CLOCK_OBS_DIM,), np.float32)
        self.phase = 0.0
        self.foot_z0 = np.array([self.data.geom_xpos[g, 2] for g in self.foot_ids])      # foot centre height at the home pose = "on the floor"

    # ---- clock ------------------------------------------------------------------------------------------------
    def _moving(self) -> bool: return bool(abs(self.command[0]) > MOVE_THRESHOLD or abs(self.command[1]) > MOVE_THRESHOLD)

    def _clock(self):
        return np.array([np.sin(2 * np.pi * self.phase), np.cos(2 * np.pi * self.phase)], np.float32) if self._moving() else np.zeros(2, np.float32)

    def _obs(self, base_obs): return np.concatenate([super()._obs(base_obs), self._clock()]).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)       # (the command is sampled inside; the clock inputs in obs are fixed up below)
        self.phase = float(self.np_random.random()); obs[-2:] = self._clock(); info["phase"] = self.phase
        return obs, info

    def step(self, action):
        if self._moving(): self.phase = (self.phase + GAIT_FREQUENCY * 0.02) % 1.0      # the metronome only ticks while asked to move
        return super().step(action)

    # ---- schedule -----------------------------------------------------------------------------------------------
    def foot_phases(self): return (self.phase + FOOT_PHASE_OFFSETS) % 1.0

    def desired_swing(self): return self.foot_phases() < SWING_FRACTION

    def _foot_heights(self): return self.data.geom_xpos[self.foot_ids, 2] - self.foot_z0

    # ---- reward -------------------------------------------------------------------------------------------------
    def _compute_reward(self, action):
        _, terms = super()._compute_reward(action)                   # tracking, slip, smoothness, ... (also updates self.in_contact / self.air_time)
        total = terms["r_raw"] - W_AIR_TIME * terms["r_air"]         # drop the touchdown bonus: the schedule replaces it
        contact, h = self.in_contact, self._foot_heights()
        if self._moving():
            ph = self.foot_phases(); swing = ph < SWING_FRACTION
            near_switch = np.minimum.reduce([np.abs(ph - 0.0), np.abs(ph - 1.0), np.abs(ph - SWING_FRACTION)]) < SCHEDULE_MARGIN
            care = ~near_switch
            r_sched = float(np.sum((contact == ~swing) & care)) / max(1, int(care.sum()))
            target = SWING_HEIGHT * np.sin(np.pi * np.clip(ph / SWING_FRACTION, 0.0, 1.0))          # half-sine arc: 0 at lift-off, SWING_HEIGHT mid-swing, 0 at touchdown
            p_swing = float(np.sum((np.maximum(0.0, target - h) ** 2) * (swing & care)))
        else:
            r_sched, p_swing = float(np.mean(contact)), 0.0            # told to stand: all four feet should be on the floor
        total += W_SCHEDULE * r_sched + W_SWING_HEIGHT * p_swing
        terms.update(r_sched=r_sched, p_swing=p_swing, r_raw=float(total), phase=float(self.phase), max_foot_h=float(h.max()))
        return float(max(total, 0.0)), terms
