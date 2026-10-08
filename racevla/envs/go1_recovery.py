"""Go1RecoveryEnv: Phase 7 recovery skill -- stand up from a very random start. The robot ALWAYS starts right side up (never on its side or upside down): random joint angles, a random tilt and a short
drop, followed by a 1 s settle with the legs held at the sampled angles; starts whose body ended up tilted more than START_TILT_MAX, or with robot parts pushed into each other, are rejected and redrawn.
Start classes: 0 = near the standing pose (like the standing task), 1 = crouched (legs moderately off home, tilt up to 0.35 rad), 2 = sprawled / flat on the belly (legs far off home, tilt up to 0.7 rad before the drop).
Action: target joint angle = home + RECOVERY_ACTION_SCALE * action (+-1.5 rad: wide enough to reach the folded leg poses). Observation: the standard 45 numbers (incl. the previous action).
Episode: 500 policy steps (10 s). Fails (and ends, reward -10) if the body tilts more than FLIP_TILT (flipped onto its side or back). No floor-contact fall rule: lying on the belly is the start state.
Reward (per step, no constant alive bonus):  progress * exp(-min(penalty, 3))  [+ FLIP_PENALTY on the step that flips]
  progress = W_UP k(tilt, 1.0) + W_HEIGHT clip(h / 0.265) cos+ + W_POSE k(e_pose, 1.0) cos+ + W_STAND k(tilt, 0.3) k(h - 0.265, 0.05) k(e_pose, 0.7)
  penalty  = 0.2 sum (target change)^2 + 1e-5 sum torque^2 + 2.5e-7 sum joint_acc^2 + 1e-4 sum joint_vel^2 + 0.05 (roll rate^2 + pitch rate^2),   e_pose = distance of the 12 joints from the home pose.
The penalties DISCOUNT the progress (like the standing reward) instead of being subtracted: an additive version (first try) made every step negative under exploration noise, so ending the episode by flipping paid better than
surviving, and the policy learned to flip within 30 steps.
info['stand_ok'] = tilt < 30 deg and body height > 0.22 m (success = stand_ok during the last 2 s of the episode)."""
import mujoco
import numpy as np
from racevla.envs.action import N_ACTIONS
from racevla.envs.go1_standing import Go1StandingEnv, STAND_HEIGHT, _kernel, _quat_mul, _axis_quat
from racevla.envs.observation import build_obs

RECOVERY_ACTION_SCALE = 1.5          # rad
MAX_STEPS = 500                      # 10 s
START_TILT_MAX = np.deg2rad(50.0)    # starts more tilted than this (after the settle) are rejected -> no side / upside-down starts
FLIP_TILT = np.deg2rad(80.0)         # failure: flipped onto its side or back
FLIP_PENALTY = -5.0
PENALTY_CAP = 3.0                    # the penalty discount never goes below exp(-3) = 0.05
SETTLE_STEPS = 500                   # physics steps (1 s) of drop + settle with PD holding the sampled joint angles
DROP_HEIGHT = 0.5                    # m, base height at the start of the drop
SELF_PENETRATION = 0.005             # m, robot parts pressed into each other deeper than this -> start rejected
STAND_TILT_OK, STAND_HEIGHT_OK = np.deg2rad(30.0), 0.22

# per start class: (joint half-ranges below / above home: hip, thigh, calf), tilt range (rad) of roll and pitch before the drop
CLASSES = {0: dict(lo=(0.2, 0.2, 0.2), hi=(0.2, 0.2, 0.2), tilt=0.2),
           1: dict(lo=(0.3, 0.3, 0.4), hi=(0.3, 0.7, 0.4), tilt=0.35),
           2: dict(lo=(0.6, 0.6, 0.8), hi=(0.6, 1.4, 0.8), tilt=0.7)}

W_UP, W_HEIGHT, W_POSE, W_STAND = 0.3, 0.4, 0.1, 0.3
W_SMOOTH, W_TORQUE, W_ACC, W_JVEL, W_SPIN = 0.2, 1e-5, 2.5e-7, 1e-4, 0.05


class Go1RecoveryEnv(Go1StandingEnv):
    def __init__(self, class_probs=(0.2, 0.4, 0.4)):
        super().__init__(push=False)
        self.action_scale = RECOVERY_ACTION_SCALE
        self.max_steps = MAX_STEPS
        self.class_probs = np.array(class_probs, float) / np.sum(class_probs)
        self.prev_target = self.home_joints.copy()
        self.start_class, self.rejected = 0, 0

    def set_class_probs(self, probs): self.class_probs = np.array(probs, float) / np.sum(probs)

    # ---- start state ----------------------------------------------------------------------------------------
    def _sample_start(self, cls):
        rng, spec = self.np_random, CLASSES[cls]
        lo, hi = np.tile(spec["lo"], 4), np.tile(spec["hi"], 4)                       # per joint: hip, thigh, calf x 4 legs
        joints = np.clip(self.home_joints + rng.uniform(-lo, hi), self.joint_lo, self.joint_hi)
        roll, pitch = rng.uniform(-spec["tilt"], spec["tilt"], 2); yaw = rng.uniform(-np.pi, np.pi)
        quat = _quat_mul(_axis_quat([0, 0, 1], yaw), _quat_mul(_axis_quat([0, 1, 0], pitch), _axis_quat([1, 0, 0], roll)))
        return joints, quat

    def _self_penetration(self):
        d = self.data
        for c in d.contact[:d.ncon]:
            if c.dist < -SELF_PENETRATION and self.floor_id not in (c.geom1, c.geom2): return True
        return False

    def reset(self, seed=None, options=None):
        super(Go1StandingEnv, self).reset(seed=seed)                                  # seeds self.np_random (skips the standing reset)
        m, d = self.model, self.data
        forced = None if not options else options.get("class")
        for attempt in range(50):
            cls = int(forced) if forced is not None else int(self.np_random.choice(3, p=self.class_probs))
            if attempt >= 30: cls = 0                                                 # last resort: a standing-like start (never expected)
            joints, quat = self._sample_start(cls)
            mujoco.mj_resetData(m, d); d.qpos[7:] = joints; d.qpos[3:7] = quat; d.qpos[0:2] = 0.0; d.qpos[2] = DROP_HEIGHT; d.qvel[:] = 0.0; d.ctrl[:] = 0.0
            mujoco.mj_forward(m, d)
            for _ in range(SETTLE_STEPS): self.pd.apply(d, joints); mujoco.mj_step(m, d)
            tilt = np.arccos(np.clip(d.xmat[1].reshape(3, 3)[2, 2], -1.0, 1.0))
            if tilt <= START_TILT_MAX and not self._self_penetration() and np.isfinite(d.qpos).all(): break
            self.rejected += 1
        d.ctrl[:] = 0.0; mujoco.mj_forward(m, d)
        self.start_class, self.prev_action, self.prev_target = cls, np.zeros(N_ACTIONS), self.home_joints.copy()
        self.step_count, self.termination_reason = 0, None
        return build_obs(d, self.home_joints, self.prev_action), {"base_height": float(d.qpos[2]), "start_class": cls, "start_tilt": float(tilt)}

    # ---- step / reward / termination -----------------------------------------------------------------------------
    def step(self, action):
        obs, reward, terminated, _, info = super().step(action)
        return obs, reward, terminated, (not terminated) and self.step_count >= self.max_steps, info

    def _tilt(self): return float(np.arccos(np.clip(self.data.xmat[1].reshape(3, 3)[2, 2], -1.0, 1.0)))

    def _compute_reward(self, action):
        d = self.data; cos = float(d.xmat[1].reshape(3, 3)[2, 2]); tilt = self._tilt(); h = float(d.qpos[2]); gate = max(cos, 0.0)
        e_pose = float(np.linalg.norm(d.qpos[7:] - self.home_joints))
        target = np.clip(self.home_joints + self.action_scale * np.clip(action, -1.0, 1.0), self.joint_lo, self.joint_hi)
        smooth = float(np.sum((target - self.prev_target) ** 2)); self.prev_target = target
        r_up = W_UP * _kernel(tilt, 1.0); r_h = W_HEIGHT * float(np.clip(h / STAND_HEIGHT, 0.0, 1.0)) * gate; r_pose = W_POSE * _kernel(e_pose, 1.0) * gate
        r_stand = W_STAND * _kernel(tilt, 0.3) * _kernel(h - STAND_HEIGHT, 0.05) * _kernel(e_pose, 0.7)
        pen = (W_SMOOTH * smooth + W_TORQUE * float(np.sum(d.ctrl ** 2)) + W_ACC * float(np.sum(d.qacc[6:] ** 2)) + W_JVEL * float(np.sum(d.qvel[6:] ** 2)) + W_SPIN * float(d.qvel[3] ** 2 + d.qvel[4] ** 2))
        flipped = tilt > FLIP_TILT
        progress = r_up + r_h + r_pose + r_stand
        reward = progress * float(np.exp(-min(pen, PENALTY_CAP))) + (FLIP_PENALTY if flipped else 0.0)         # never negative except the flip step: dying early never pays
        terms = {"progress": progress, "r_up": r_up, "r_height": r_h, "r_pose": r_pose, "r_stand": r_stand, "p_total": pen, "p_smooth": smooth, "tilt": tilt, "e_pose": e_pose, "height": h,
                 "stand_ok": float(tilt < STAND_TILT_OK and h > STAND_HEIGHT_OK)}
        return float(reward), terms

    def _is_terminated(self) -> bool:
        d = self.data
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()) or any(d.warning[w].number > 0 for w in self._bad_state_warnings):
            self.termination_reason = "bad_state"; return True
        if self._tilt() > FLIP_TILT: self.termination_reason = "flipped"; return True
        return False
