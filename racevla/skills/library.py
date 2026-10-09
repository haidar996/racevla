"""Skill library + supervisor (Phase 7, skill switching, stage 1: stand / walk / run on flat ground).
A Skill = a trained network + the three things that go with it: (1) which simulator mode it needs (body.set_mode), (2) how to build ITS input from the shared robot state, (3) a warm-up (steps of zero action after taking over).
The Supervisor holds one SwitchBody and the skills, asks the ACTIVE skill for an action every step, and can hand control to another skill at any moment."""
import json, pathlib
import numpy as np
from stable_baselines3 import PPO, TD3

from racevla.envs.wrappers import OBS_SCALE
from racevla.skills.body import SwitchBody, FlatSwitchBody, TerrainSwitchBody, RaceSwitchBody

ROOT = pathlib.Path(__file__).resolve().parents[2]
OBS_BASE = OBS_SCALE.shape[0]            # 45 robot numbers


def normalize(obs):
    """What FixedObsNormalize does (it wraps an env; here the supervisor builds the observation itself)."""
    return (obs / np.concatenate([OBS_SCALE, np.ones(obs.shape[-1] - OBS_BASE, np.float32)])).astype(np.float32)


class Skill:
    def __init__(self, name, mode, model, with_command, warmup=0, scan=False, hold=False):
        self.name, self.mode, self.model, self.with_command, self.warmup, self.scan, self.hold = name, mode, model, with_command, warmup, scan, hold

    def observe(self, body, base):
        """45 robot numbers (stand) or 45 + command (vx, wz) + clock (sin, cos) (walk, run, terrain); step-over: + 24 far height-scan numbers + 2 jump numbers (always 0, no jump)."""
        obs = np.concatenate([base, body.command, body._clock()]) if self.with_command else base
        if self.scan: obs = np.concatenate([obs, body.scan(), np.zeros(2)])
        return normalize(obs.astype(np.float32))

    def act(self, body, base): return self.model.predict(self.observe(body, base), deterministic=True)[0]


def load_skills():
    m = ROOT / "models"
    return {"stand": Skill("stand", "stand", TD3.load(m / "standing_policy_td3_seed0.zip", device="cpu"), with_command=False, warmup=10),
            "walk": Skill("walk", "walk", PPO.load(m / "walking_policy_sine_seed1.zip", device="cpu"), with_command=True),
            "run": Skill("run", "run", PPO.load(m / "running_policy_2p5_seed0.zip", device="cpu"), with_command=True),
            "terrain": Skill("terrain", "terrain", PPO.load(m / "terrain_policy_seed2.zip", device="cpu"), with_command=True, hold=True),            # heading hold: it was trained with turn command = -2 x yaw error
            "stepover": Skill("stepover", "stepover", PPO.load(m / "hurdle_stepover_seed0.zip", device="cpu"), with_command=True, scan=True, hold=True)}


class Supervisor:
    def __init__(self, terrain=False, flat_skills=False, race=False):
        table = json.load(open(ROOT / "models/running_gait_table.json"))       # terrain=True: the 10 m terrain course (flat = ('rough', 0.0)); flat_skills=True: infinite flat floor but all five modes; neither: stand / walk / run only
        self.body = (RaceSwitchBody if race else TerrainSwitchBody if terrain else FlatSwitchBody if flat_skills else SwitchBody)(table)
        self.skills = load_skills(); self.active = None; self.warm = 0; self.base = None
        self.pending = None; self.calm = 0; self.handover_step = None; self.steps = 0
        self.vision = None                   # a racevla.vision.heightmap.HeightMapper (attached with .attach()): fed with the depth image before every action

    def reset(self, seed, name, command, warmup=None, options=None):
        obs, _ = self.body.reset(seed=seed, options=options); self.base = obs[:OBS_BASE]; self.switch(name, command, warmup)
        if self.vision is not None: self.vision.reset(); self.vision.update(self.body.data)
        return obs

    def switch(self, name, command, warmup=None):
        """Hand control to skill `name`: set the simulator mode, the new command and the warm-up (None = the skill's default)."""
        self.active = self.skills[name]; self.body.set_mode(self.active.mode); self.body.command = np.array(command, float)
        self.warm = self.active.warmup if warmup is None else warmup

    def go(self, name, command, brake_cmd=0.0, brake_speed=0.6, warmup=None):
        """Ask for skill `name` with the safe route: standing is NEVER entered directly from running. From run, walk takes over first with forward command `brake_cmd` (the brake); when the
        forward speed has stayed below `brake_speed` m/s for 5 steps, stand takes over. Same for step-over. Every other request is an immediate switch."""
        if self.active.mode in ("run", "stepover") and name == "stand":
            self.switch("walk", [brake_cmd, 0.0]); self.pending = (name, command, brake_speed, warmup); self.calm = 0; self.handover_step = None
        else: self.pending = None; self.switch(name, command, warmup)

    def step(self):
        if self.vision is not None: self.vision.update(self.body.data)                          # the camera map is updated with the newest picture
        if self.active.hold: self.body.command[1] = self.body.heading_hold()                    # terrain / step-over: the turn command is the heading hold, refreshed every step
        action = np.zeros(12) if self.warm > 0 else self.active.act(self.body, self.base)      # warm-up: hold the home pose (+ reference) with zero action
        self.warm = max(self.warm - 1, 0)
        obs, reward, terminated, truncated, info = self.body.step(action); self.base = obs[:OBS_BASE]; self.steps += 1
        if self.pending is not None:                                                            # braking: wait until slow enough, then hand over to the real target
            name, command, speed, warmup = self.pending; self.calm = self.calm + 1 if abs(info["vx"]) < speed else 0
            if self.calm >= 5: self.pending = None; self.handover_step = self.steps; self.switch(name, command, warmup)
        return terminated, info
