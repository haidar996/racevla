"""Go1StandingEnv. Step 10 = __init__ + reset(); step() comes in Step 11."""
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from racevla.controllers.pd import PDController
from racevla.envs.action import ACTION_SCALE, N_ACTIONS, action_to_target
from racevla.envs.observation import OBS_DIM, build_obs
from racevla.robots.go1 import LEGS, ctrl_limits, load_model, reset_home

# --- timing ------------------------------------------------------------------------------------
DECIMATION = 10              # physics steps (0.002 s each) per policy step  ->  policy runs at 50 Hz
MAX_EPISODE_STEPS = 1000     # policy steps per episode = 20 s

# --- reset randomization ("large", option C) -------------------------------------------------
RESET_JOINT_NOISE = 0.2      # rad,   uniform +/- around the home pose (then clipped to joint limits)
RESET_TILT_NOISE = 0.2       # rad,   uniform +/- on base roll and pitch (yaw is fully random)
RESET_LIN_VEL_NOISE = 0.3    # m/s,   uniform +/- on base linear velocity
RESET_ANG_VEL_NOISE = 0.3    # rad/s, uniform +/- on base angular velocity
RESET_JOINT_VEL_NOISE = 0.3  # rad/s, uniform +/- on joint velocities
FOOT_CLEARANCE = 0.01        # m, lowest foot starts this far above the floor

# --- termination -------------------------------------------------------------------------------
MAX_TILT = np.deg2rad(60.0)  # rad, angle between body 'up' and world 'up' beyond which the robot has fallen
BODY_CONTACT_DEPTH = 0.003   # m, non-foot floor penetration beyond which the robot has fallen (brushes stay < 1.6 mm)

# --- pushes (only when Go1StandingEnv(push=True)) ----------------------------------------------
PUSH_GRACE_STEPS = 50        # policy steps (1 s) with no push, so the landing after reset can finish
PUSH_INTERVAL_STEPS = (50, 150)   # steps between pushes, uniform integer in [50, 150) = 1-3 s
PUSH_KICK_RANGE = (0.4, 1.2)      # m/s, horizontal base velocity added by one push (uniform)

# --- reward (option C: weighted sum of kernels, discounted by smoothness penalties) ------------
STAND_HEIGHT = 0.265         # m, base height the robot settles at holding the home pose (keyframe says 0.270; PD sags 5 mm)
UPRIGHT_SCALE = 0.5          # rad, tilt at which the upright kernel drops to 1/e
HEIGHT_SCALE = 0.05          # m,   height error at which the height kernel drops to 1/e
POSE_SCALE = 1.0             # rad, joint-pose distance (12-D norm) at which the pose kernel drops to 1/e
ALIVE_BONUS = 0.2            # constant part of the bracket: the floor of the reward
W_UPRIGHT, W_HEIGHT, W_POSE = 0.4, 0.3, 0.1   # with ALIVE_BONUS these sum to 1.0
ACTION_RATE_COEF = 0.05     # discount strength for jerky actions
JOINT_VEL_COEF = 0.0005      # discount strength for fast joint motion


def _kernel(error, scale):
    """1.0 at zero error, falling smoothly to 0; equals 1/e at error == scale."""
    return float(np.exp(-(error / scale) ** 2))


def _quat_mul(a, b):
    out = np.zeros(4); mujoco.mju_mulQuat(out, a, b); return out


def _axis_quat(axis, angle):
    q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, np.asarray(axis, float), angle); return q


class Go1StandingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, push=False, push_kick_range=None, scene_xml=None):
        self.push = push                                          # random velocity kicks to the base during the episode
        self.push_kick_range = tuple(push_kick_range) if push_kick_range else PUSH_KICK_RANGE   # m/s (min, max); default = the standard 0.4-1.2
        self.next_push_step = None
        self.model, self.data = load_model(scene_xml) if scene_xml else load_model()      # scene_xml: another scene (e.g. with terrain); default = flat floor
        self.ground_height = 0.0                                  # height of the ground under the start position (terrain scenes change it)
        self.pd = PDController(self.model)                       # switches actuators to torque mode
        self.joint_lo, self.joint_hi = ctrl_limits(self.model)
        reset_home(self.model, self.data)
        self.home_joints = self.data.qpos[7:].copy()
        self.home_qpos = self.data.qpos.copy()
        self.foot_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, leg) for leg in LEGS]
        self.foot_radius = self.model.geom_size[self.foot_ids[0], 0]
        self.floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.termination_reason = None
        self._bad_state_warnings = [mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC]

        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (N_ACTIONS,), np.float32)
        self.prev_action = np.zeros(N_ACTIONS)
        self.step_count = 0
        self.action_scale = ACTION_SCALE                          # rad per unit of action; subclasses with a reference motion can use a smaller residual

    def _reference_offset(self):
        """Joint-angle offset (rad, scalar or (12,)) added to the home pose before the action is applied. Zero for standing."""
        return 0.0

    def _lowest_foot_bottom(self):
        return min(self.data.geom_xpos[i, 2] for i in self.foot_ids) - self.foot_radius

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)              # seeds self.np_random: ALL randomness must come from it
        rng, m, d = self.np_random, self.model, self.data
        mujoco.mj_resetData(m, d)

        # joints: home pose + noise, kept inside the joint limits
        joints = self.home_joints + rng.uniform(-RESET_JOINT_NOISE, RESET_JOINT_NOISE, 12)
        d.qpos[7:] = np.clip(joints, self.joint_lo, self.joint_hi)

        # base orientation: random yaw, small random roll/pitch   (q = Rz(yaw) * Ry(pitch) * Rx(roll))
        roll, pitch = rng.uniform(-RESET_TILT_NOISE, RESET_TILT_NOISE, 2)
        yaw = rng.uniform(-np.pi, np.pi)
        if options and options.get("yaw") is not None: yaw = float(options["yaw"])      # (drawn above either way, so the random stream is unchanged)
        d.qpos[3:7] = _quat_mul(_axis_quat([0, 0, 1], yaw), _quat_mul(_axis_quat([0, 1, 0], pitch), _axis_quat([1, 0, 0], roll)))
        d.qpos[0:2] = 0.0

        # base height: put the lowest foot FOOT_CLEARANCE above the floor (never start inside the ground)
        d.qpos[2] = self.home_qpos[2]
        mujoco.mj_forward(m, d)
        d.qpos[2] += self.ground_height + FOOT_CLEARANCE - self._lowest_foot_bottom()

        # velocities
        d.qvel[0:3] = rng.uniform(-RESET_LIN_VEL_NOISE, RESET_LIN_VEL_NOISE, 3)
        d.qvel[3:6] = rng.uniform(-RESET_ANG_VEL_NOISE, RESET_ANG_VEL_NOISE, 3)
        d.qvel[6:] = rng.uniform(-RESET_JOINT_VEL_NOISE, RESET_JOINT_VEL_NOISE, 12)
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)               # refresh derived quantities (rotation matrix, contacts) before observing

        self.prev_action = np.zeros(N_ACTIONS)
        self.step_count = 0
        self.termination_reason = None
        if self.push:                         # drawn last, and only if pushing, so push=False resets are unchanged
            self.next_push_step = PUSH_GRACE_STEPS + self._draw_push_interval()
        return build_obs(d, self.home_joints, self.prev_action), {"base_height": float(d.qpos[2])}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = action_to_target(action, self.home_joints + self._reference_offset(), self.joint_lo, self.joint_hi, scale=self.action_scale)
        kick = self._maybe_push()                     # before the physics loop, so this step's obs already shows the reaction

        for _ in range(DECIMATION):                   # PD + physics run at 500 Hz, policy at 50 Hz
            self.pd.apply(self.data, target)
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = build_obs(self.data, self.home_joints, action)   # this action becomes next step's "previous action"
        reward, reward_terms = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self.step_count >= MAX_EPISODE_STEPS       # time limit, not a failure
        self.prev_action = action
        info = {"base_height": float(self.data.qpos[2]), "push_kick": kick, **reward_terms}
        if terminated:
            info["termination_reason"] = self.termination_reason
        return obs, reward, terminated, truncated, info

    def _draw_push_interval(self) -> int:
        return int(self.np_random.integers(*PUSH_INTERVAL_STEPS))

    def _maybe_push(self) -> float:
        """If a push is due, add a horizontal velocity kick to the base. Returns the kick size in m/s (0.0 if none)."""
        if not self.push or self.step_count < self.next_push_step:
            return 0.0
        size = float(self.np_random.uniform(*self.push_kick_range))
        angle = float(self.np_random.uniform(0.0, 2.0 * np.pi))
        self.data.qvel[0] += size * np.cos(angle)                 # qvel[0:2] is the base's world-frame x, y velocity
        self.data.qvel[1] += size * np.sin(angle)
        self.next_push_step = self.step_count + self._draw_push_interval()
        return size

    def _compute_reward(self, action):
        """Returns (reward, terms): terms holds every ingredient, so it can be logged and the coefficients tuned."""
        d = self.data
        tilt = np.arccos(np.clip(d.xmat[1].reshape(3, 3)[2, 2], -1.0, 1.0))
        upright = _kernel(tilt, UPRIGHT_SCALE)
        height = _kernel(d.qpos[2] - STAND_HEIGHT, HEIGHT_SCALE)
        pose = _kernel(np.linalg.norm(d.qpos[7:] - self.home_joints), POSE_SCALE)
        bracket = ALIVE_BONUS + W_UPRIGHT * upright + W_HEIGHT * height + W_POSE * pose      # in [0.2, 1.0]

        action_rate = np.sum((action - self.prev_action) ** 2)
        joint_vel_sq = np.sum(d.qvel[6:] ** 2)
        discount = np.exp(-(ACTION_RATE_COEF * action_rate + JOINT_VEL_COEF * joint_vel_sq))  # in (0, 1]
        reward = float(bracket * discount)                     # always > 0, so dying early never pays
        terms = {"r_upright": upright, "r_height": height, "r_pose": pose, "r_bracket": float(bracket),
                 "r_action_rate": float(action_rate), "r_joint_vel_sq": float(joint_vel_sq), "r_discount": float(discount)}
        return reward, terms

    def _is_terminated(self) -> bool:
        d = self.data
        # D) numerical guard: MuJoCo auto-resets on bad values (and counts a warning), so check both
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()) or \
           any(d.warning[w].number > 0 for w in self._bad_state_warnings):
            self.termination_reason = "bad_state"; return True
        # B) tilt: cos(angle between body z-axis and world z-axis) = R[2, 2]
        if np.arccos(np.clip(d.xmat[1].reshape(3, 3)[2, 2], -1.0, 1.0)) > MAX_TILT:
            self.termination_reason = "tilt"; return True
        # A) any part other than a foot pressed into the floor deeper than BODY_CONTACT_DEPTH (dist < 0 means
        #    penetration; landing right after reset brushes up to ~1.5 mm, real collapse reaches 5+ mm)
        for c in d.contact[:d.ncon]:
            g1, g2 = c.geom1, c.geom2
            if c.dist < -BODY_CONTACT_DEPTH and self.floor_id in (g1, g2):
                other = g2 if g1 == self.floor_id else g1
                if other not in self.foot_ids:
                    self.termination_reason = "body_contact"; return True
        return False
