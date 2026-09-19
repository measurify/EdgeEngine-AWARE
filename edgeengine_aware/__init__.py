"""EdgeEngine AWARE - application- and energy-aware simulation environment
for reinforcement learning in energy-harvesting Edge IoT systems.

Quick start::

    import gymnasium as gym
    import edgeengine_aware  # registers "EdgeEngineAware-v0"

    env = gym.make("EdgeEngineAware-v0")
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

__version__ = "0.2.0"

from gymnasium.envs.registration import register

from .actions import Action, decode_action, describe_action, encode_action, flatten_action, unflatten_action
from .config import (
    AgricultureConfig,
    ApplicationConfig,
    CommunicationConfig,
    DomainRandomizationConfig,
    EdgeEngineAwareConfig,
    EnergyStorageConfig,
    HarvestingConfig,
    MCUConfig,
    ObservationConfig,
    RewardConfig,
    SensingConfig,
    TimeConfig,
    default_config,
)
from .env import EdgeEngineAwareEnv
from .interfaces import Policy
from .metrics import EpisodeMetrics
from .observation import OBSERVATION_FIELDS, NodeProfile, NodeState, ObservationBuilder
from .policies import AlwaysOnPolicy, PeriodicPolicy, RandomPolicy, RuleBasedParams, RuleBasedPolicy, run_episode
from .scenarios import SCENARIOS, get_scenario

register(
    id="EdgeEngineAware-v0",
    entry_point="edgeengine_aware.env:EdgeEngineAwareEnv",
    max_episode_steps=None,  # the environment truncates itself at config.time.max_steps
)

__all__ = [
    "__version__",
    "EdgeEngineAwareEnv",
    "EdgeEngineAwareConfig",
    "default_config",
    "TimeConfig",
    "EnergyStorageConfig",
    "MCUConfig",
    "HarvestingConfig",
    "SensingConfig",
    "CommunicationConfig",
    "AgricultureConfig",
    "ApplicationConfig",
    "RewardConfig",
    "ObservationConfig",
    "DomainRandomizationConfig",
    "Policy",
    "RuleBasedPolicy",
    "RuleBasedParams",
    "RandomPolicy",
    "PeriodicPolicy",
    "AlwaysOnPolicy",
    "run_episode",
    "SCENARIOS",
    "get_scenario",
    "EpisodeMetrics",
    "ObservationBuilder",
    "OBSERVATION_FIELDS",
    "NodeProfile",
    "NodeState",
    "Action",
    "encode_action",
    "decode_action",
    "flatten_action",
    "unflatten_action",
    "describe_action",
]
