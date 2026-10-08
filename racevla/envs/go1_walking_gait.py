"""Go1WalkingGaitRewardEnv: Phase 4, option C -- REWARD-ONLY gait fixes. No clock, no reference motion, the standard 47-D observation; only the reward changes, using nothing but the feet's contact states and heights:
  * foot clearance: reward for a foot that is in the air, in proportion to its height up to CLEARANCE_TARGET (so feet must really lift)
  * trot pairing: reward when the diagonal feet (FR,RL) and (FL,RR) share the same contact state
  * alternation: reward when exactly one diagonal pair is down and the other pair is up (a proper trot swing, not a pronk or a stand)
  * stuck-foot penalty: a foot that stays on the floor, or in the air, longer than OVERTIME_AFTER seconds while the robot is asked to move is penalised (kills the 'three feet down, one leg held up' stance)
  * told to stand: reward for all four feet on the floor
These terms are added to Go1WalkingEnv's reward (tracking, slip penalty, smoothness, capped air-time bonus)."""
import numpy as np

from racevla.envs.go1_walking import Go1WalkingEnv, MOVE_THRESHOLD

CLEARANCE_TARGET = 0.08     # m, foot height that earns the full clearance reward
OVERTIME_AFTER = 0.6        # s, a foot may stay down (or up) this long while moving before it is penalised
OVERTIME_CAP = 1.0          # s, the penalty stops growing after this much overtime per foot
W_CLEARANCE = 1.0
W_TROT_PAIRS = 0.3
W_ALTERNATE = 0.3
W_STUCK = -0.3              # per second of (capped) overtime, summed over feet: worst case about -1.2 per step, so a standing robot still scores a little above 0
W_STAND_FEET = 0.5          # told to stand: fraction of feet on the floor


class Go1WalkingGaitRewardEnv(Go1WalkingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.foot_z0 = np.array([self.data.geom_xpos[g, 2] for g in self.foot_ids])       # foot centre height at the home pose = "on the floor"
        self.contact_time = np.zeros(4)

    def _foot_heights(self): return self.data.geom_xpos[self.foot_ids, 2] - self.foot_z0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options); self.contact_time[:] = 0.0
        return obs, info

    def _moving(self) -> bool: return bool(abs(self.command[0]) > MOVE_THRESHOLD or abs(self.command[1]) > MOVE_THRESHOLD)

    def _compute_reward(self, action):
        _, terms = super()._compute_reward(action)                       # base walking reward; also updates self.in_contact and self.air_time
        total, contact, h = terms["r_raw"], self.in_contact, self._foot_heights()
        self.contact_time = np.where(contact, self.contact_time + 0.02, 0.0)
        r_clear = r_trot = r_alt = p_stuck = r_stand = 0.0
        if self._moving():
            r_clear = float(np.sum((~contact) * np.clip(h, 0.0, CLEARANCE_TARGET) / CLEARANCE_TARGET)) / 4.0
            r_trot = 0.5 * (float(contact[0] == contact[3]) + float(contact[1] == contact[2]))            # foot order FR, FL, RR, RL: diagonals are (FR, RL) and (FL, RR)
            r_alt = abs(int(contact[0]) + int(contact[3]) - int(contact[1]) - int(contact[2])) / 2.0
            over_down = np.minimum(np.maximum(self.contact_time - OVERTIME_AFTER, 0.0), OVERTIME_CAP)
            over_up = np.minimum(np.maximum(self.air_time - OVERTIME_AFTER, 0.0), OVERTIME_CAP)
            p_stuck = float(np.sum(over_down) + np.sum(over_up))
            total += W_CLEARANCE * r_clear + W_TROT_PAIRS * r_trot + W_ALTERNATE * r_alt + W_STUCK * p_stuck
        else:
            r_stand = float(np.mean(contact)); total += W_STAND_FEET * r_stand
        terms.update(r_clear=r_clear, r_trot=r_trot, r_alt=r_alt, p_stuck=p_stuck, r_stand=r_stand, r_raw=float(total), max_foot_h=float(h.max()))
        return float(max(total, 0.0)), terms
