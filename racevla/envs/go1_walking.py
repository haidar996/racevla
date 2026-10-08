"""Go1WalkingEnv: velocity-command walking (Phase 4). Built on Go1StandingEnv -- same robot, PD controller, reset randomisation, 50 Hz policy and optional pushes;
what changes is (1) the observation gets the 2 command numbers, (2) the reward scores command tracking, (3) the fall rule only counts tilt and TRUNK contact.

Command = (forward speed vx in m/s, turn rate wz in rad/s), resampled every COMMAND_INTERVAL_STEPS; sometimes zero ("stand still"). No sideways command (vy = 0).
Reward = ALIVE + weighted tracking terms + feet-air-time bonus - penalties (incl. foot slip), clipped at >= 0 (so ending an episode early never pays; lesson from standing)."""
import mujoco
import numpy as np
from gymnasium import spaces

from racevla.envs.go1_standing import Go1StandingEnv, MAX_TILT, BODY_CONTACT_DEPTH
from racevla.envs.observation import OBS_DIM

# --- commands ------------------------------------------------------------------------------------
CMD_VX_RANGE = (0.0, 0.6)         # m/s forward (no backwards walking at first)
CMD_WZ_RANGE = (-0.5, 0.5)        # rad/s turn rate (counter-clockwise positive)
COMMAND_INTERVAL_STEPS = 200      # resample every 4 s
STAND_PROBABILITY = 0.15          # chance that a new command is exactly zero (stand still)
MOVE_THRESHOLD = 0.1              # |command| above this counts as "asked to move"
WALK_OBS_DIM = OBS_DIM + 2        # 45 + (vx, wz) = 47

# --- reward weights (first guesses, to be tuned in the smoke test) ----------------------------------
ALIVE_BONUS = 0.1
W_LIN_TRACK, LIN_SIGMA2 = 1.0, 0.25     # exp(-|v_xy - cmd_xy|^2 / sigma^2)
W_YAW_TRACK, YAW_SIGMA2 = 0.5, 0.25     # exp(-(wz - cmd_wz)^2 / sigma^2)
W_AIR_TIME, AIR_TIME_TARGET = 1.0, 0.3  # reward (min(air_time, AIR_TIME_CAP) - target) at each touchdown, only when asked to move
AIR_TIME_CAP = 0.5                      # v2: a touchdown pays at most (0.5 - 0.3) = +0.2, however long the foot was held up
W_FOOT_SLIP = -0.3                      # v2: horizontal speed of every foot that touches the floor (stops skating on dragged feet)
W_VZ = -2.0                             # vertical body velocity^2 (no bouncing)
W_ANG_XY = -0.05                        # roll / pitch rate^2
W_TILT = -2.0                           # gravity-xy^2: body tilt
W_TORQUE = -1e-5                        # sum of torque^2
W_JOINT_ACC = -2.5e-7                   # sum of joint acceleration^2
W_ACTION_RATE = -0.01                   # sum of (action - previous action)^2
W_STAND_POSE = -0.3                     # joint distance from home pose, only when commanded to stand still

# --- fall rule: only trunk contact and tilt (the 3 mm lower-leg rule of standing is dropped, legs are near the floor while walking) ---
TRUNK_CONTACT_DEPTH = BODY_CONTACT_DEPTH


class Go1WalkingEnv(Go1StandingEnv):
    def __init__(self, push=False, push_kick_range=None, vx_range=CMD_VX_RANGE, wz_range=CMD_WZ_RANGE, scene_xml=None):
        super().__init__(push=push, push_kick_range=push_kick_range, scene_xml=scene_xml)
        self.vx_range, self.wz_range = tuple(vx_range), tuple(wz_range)
        self.observation_space = spaces.Box(-np.inf, np.inf, (WALK_OBS_DIM,), np.float32)
        m = self.model
        self.trunk_geoms = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name == "trunk"}
        self.command = np.zeros(2)                       # (vx, wz)
        self.air_time = np.zeros(4); self.in_contact = np.ones(4, bool)

    # ---- commands -----------------------------------------------------------------------------
    def _sample_command(self):
        if self.np_random.random() < STAND_PROBABILITY: self.command = np.zeros(2)
        else: self.command = np.array([self.np_random.uniform(*self.vx_range), self.np_random.uniform(*self.wz_range)])

    def _obs(self, base_obs): return np.concatenate([base_obs, self.command]).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._sample_command(); self.air_time[:] = 0.0; self.in_contact = self._foot_contacts()
        info.update(cmd_vx=float(self.command[0]), cmd_wz=float(self.command[1]))
        return self._obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if not (terminated or truncated) and self.step_count % COMMAND_INTERVAL_STEPS == 0: self._sample_command()   # next obs shows the new command
        info.update(cmd_vx=float(self.command[0]), cmd_wz=float(self.command[1]))
        return self._obs(obs), reward, terminated, truncated, info

    # ---- contacts ---------------------------------------------------------------------------
    def _foot_contacts(self):
        d, touching = self.data, np.zeros(4, bool)
        for c in d.contact[:d.ncon]:
            if c.dist > 0.0: continue
            for i, f in enumerate(self.foot_ids):
                if (c.geom1 == f and c.geom2 == self.floor_id) or (c.geom2 == f and c.geom1 == self.floor_id): touching[i] = True
        return touching

    # ---- reward -------------------------------------------------------------------------------
    def _compute_reward(self, action):
        d = self.data; R = d.xmat[1].reshape(3, 3)
        v_body = R.T @ d.qvel[0:3]; wz = d.qvel[5]; cmd_v, cmd_w = self.command
        asked_to_move = abs(cmd_v) > MOVE_THRESHOLD or abs(cmd_w) > MOVE_THRESHOLD
        lin_track = float(np.exp(-((v_body[0] - cmd_v) ** 2 + v_body[1] ** 2) / LIN_SIGMA2))
        yaw_track = float(np.exp(-((wz - cmd_w) ** 2) / YAW_SIGMA2))
        contact = self._foot_contacts(); touchdown = contact & ~self.in_contact
        air_bonus = float(np.sum(np.minimum(self.air_time[touchdown], AIR_TIME_CAP) - AIR_TIME_TARGET)) if asked_to_move else 0.0
        slip, vel6 = 0.0, np.zeros(6)
        for i, g in enumerate(self.foot_ids):
            if contact[i]: mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_GEOM, g, vel6, 0); slip += float(np.linalg.norm(vel6[3:5]))   # world-frame horizontal speed of a foot that is on the floor
        self.air_time = np.where(contact, 0.0, self.air_time + 0.02); self.in_contact = contact
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        pen_vz, pen_angxy = float(v_body[2] ** 2), float(d.qvel[3] ** 2 + d.qvel[4] ** 2)
        pen_tilt, pen_torque = float(gravity[0] ** 2 + gravity[1] ** 2), float(np.sum(d.ctrl ** 2))
        pen_acc, pen_rate = float(np.sum(d.qacc[6:] ** 2)), float(np.sum((action - self.prev_action) ** 2))
        pen_pose = 0.0 if asked_to_move else float(np.linalg.norm(d.qpos[7:] - self.home_joints))
        total = (ALIVE_BONUS + W_LIN_TRACK * lin_track + W_YAW_TRACK * yaw_track + W_AIR_TIME * air_bonus + W_VZ * pen_vz + W_ANG_XY * pen_angxy + W_TILT * pen_tilt
                 + W_TORQUE * pen_torque + W_JOINT_ACC * pen_acc + W_ACTION_RATE * pen_rate + W_STAND_POSE * pen_pose + W_FOOT_SLIP * slip)
        terms = {"r_lin_track": lin_track, "r_yaw_track": yaw_track, "r_air": air_bonus, "p_vz": pen_vz, "p_angxy": pen_angxy, "p_tilt": pen_tilt, "p_torque": pen_torque,
                 "p_acc": pen_acc, "p_rate": pen_rate, "p_pose": pen_pose, "p_slip": slip, "vx": float(v_body[0]), "wz": float(wz), "r_raw": float(total)}
        return float(max(total, 0.0)), terms

    # ---- fall rule ------------------------------------------------------------------------------
    def _is_terminated(self) -> bool:
        d = self.data
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()) or any(d.warning[w].number > 0 for w in self._bad_state_warnings):
            self.termination_reason = "bad_state"; return True
        if np.arccos(np.clip(d.xmat[1].reshape(3, 3)[2, 2], -1.0, 1.0)) > MAX_TILT:
            self.termination_reason = "tilt"; return True
        for c in d.contact[:d.ncon]:
            if c.dist < -TRUNK_CONTACT_DEPTH and self.floor_id in (c.geom1, c.geom2) and ((c.geom2 if c.geom1 == self.floor_id else c.geom1) in self.trunk_geoms):
                self.termination_reason = "trunk_contact"; return True
        return False
