"""Fast smoke tests: environments build and step, the terrain generator works, and every packaged policy loads, matches its environment and keeps the robot upright for a short rollout.
Run: pytest -q"""
import json, pathlib
import numpy as np
import pytest
from stable_baselines3 import PPO, TD3

from racevla.envs.go1_running_sine import Go1RunningSineEnv
from racevla.envs.go1_standing import Go1StandingEnv
from racevla.envs.go1_stepover import Go1StepOverEnv
from racevla.envs.go1_terrain_walk import Go1TerrainWalkEnv
from racevla.envs.go1_walking_sine import Go1WalkingSineEnv
from racevla.envs.wrappers import FixedObsNormalize

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


@pytest.mark.parametrize("cls,obs_dim", [(Go1StandingEnv, 45), (Go1WalkingSineEnv, 49), (Go1RunningSineEnv, 49), (Go1TerrainWalkEnv, 49), (Go1StepOverEnv, 75)])
def test_env_builds_and_steps(cls, obs_dim):
    env = cls(); obs, _ = env.reset(seed=1)
    assert obs.shape == (obs_dim,) and np.isfinite(obs).all()
    obs, reward, term, trunc, info = env.step(env.action_space.sample() * 0)
    assert obs.shape == (obs_dim,) and np.isfinite(reward)


def test_terrain_generator_hurdle_and_stairs():
    env = Go1TerrainWalkEnv()
    for kind, level in [("rough", 0.06), ("slope_up", 12), ("stairs_up", 0.04), ("stairs_down", 0.06), ("hurdle", 0.05)]:
        env.set_terrain(kind, level, 0); env.reset(seed=0, options={"terrain": (kind, level), "terrain_seed": 0, "yaw": 0.0})
        assert np.isfinite(env.data.qpos).all()


def test_standing_td3_survives_pushes():
    model = TD3.load(MODELS / "standing_policy_td3_seed0.zip", device="cpu"); env = FixedObsNormalize(Go1StandingEnv(push=True)); obs, _ = env.reset(seed=3)
    for t in range(300):                                           # first 10 steps passive (see models/standing_policy_final.md)
        act = np.zeros(12, np.float32) if t < 10 else model.predict(obs, deterministic=True)[0]
        obs, _, term, trunc, _ = env.step(act); assert not term, f"fell at step {t}"


def test_walking_policy_walks_forward():
    model = PPO.load(MODELS / "walking_policy_sine_seed1.zip", device="cpu"); raw = Go1WalkingSineEnv(push=False); env = FixedObsNormalize(raw); raw._sample_command = lambda: None
    obs, _ = env.reset(seed=3000, options={"yaw": 0.0}); x0 = float(raw.data.qpos[0])
    for t in range(250):
        raw.command = np.array([0.5, 0.0]); obs[-4:-2] = raw.command; obs[-2:] = raw._clock()
        obs, _, term, _, _ = env.step(model.predict(obs, deterministic=True)[0]); assert not term
    assert raw.data.qpos[0] - x0 > 1.0


def test_running_policy_reaches_speed():
    table = json.load(open(MODELS / "running_gait_table.json")); model = PPO.load(MODELS / "running_policy_2p5_seed0.zip", device="cpu")
    raw = Go1RunningSineEnv(push=False, gait={"table": table}); env = FixedObsNormalize(raw); raw._sample_command = lambda: None; obs, _ = env.reset(seed=3000, options={"yaw": 0.0}); vx = []
    for t in range(300):
        raw.command = np.array([1.5, 0.0]); obs[-4:-2] = raw.command; obs[-2:] = raw._clock()
        obs, _, term, _, info = env.step(model.predict(obs, deterministic=True)[0]); assert not term; vx.append(info["vx"])
    assert np.mean(vx[150:]) > 1.2


def test_terrain_policy_climbs_slope():
    model = PPO.load(MODELS / "terrain_policy_seed2.zip", device="cpu"); raw = Go1TerrainWalkEnv(); env = FixedObsNormalize(raw); raw._sample_command = lambda: None
    obs, _ = env.reset(seed=1, options={"terrain": ("slope_up", 9), "terrain_seed": 1, "yaw": 0.0})
    for t in range(300):
        raw.command = np.array([0.6, raw.command[1]]); obs[-4:-2] = raw.command
        obs, _, term, _, _ = env.step(model.predict(obs, deterministic=True)[0]); assert not term
    assert raw.data.qpos[0] > 1.5


def test_stepover_policy_loads_and_runs():
    model = PPO.load(MODELS / "hurdle_stepover_seed0.zip", device="cpu"); env = FixedObsNormalize(Go1StepOverEnv())
    assert model.observation_space.shape == (75,)
    obs, _ = env.reset(seed=2000, options={"terrain": ("hurdle", 0.03), "terrain_seed": 2000, "yaw": 0.0})
    for t in range(150): obs, _, term, _, _ = env.step(model.predict(obs, deterministic=True)[0]); assert not term
