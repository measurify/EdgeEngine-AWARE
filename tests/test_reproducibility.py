"""Seeding: identical seeds -> identical trajectories; different seeds -> different ones."""

from __future__ import annotations

import numpy as np

import edgeengine_aware as ea


def rollout(env, seed, policy, n=200):
    obs, _ = env.reset(seed=seed)
    policy.reset()
    observations, rewards = [obs.copy()], []
    for _ in range(n):
        obs, r, term, trunc, _ = env.step(policy.act(obs))
        observations.append(obs.copy())
        rewards.append(r)
    return np.array(observations), np.array(rewards)


def test_same_seed_same_trajectory():
    env = ea.EdgeEngineAwareEnv()
    o1, r1 = rollout(env, 123, ea.RuleBasedPolicy())
    o2, r2 = rollout(env, 123, ea.RuleBasedPolicy())
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(r1, r2)


def test_same_seed_same_trajectory_across_instances():
    e1, e2 = ea.EdgeEngineAwareEnv(), ea.EdgeEngineAwareEnv()
    o1, r1 = rollout(e1, 9, ea.PeriodicPolicy(3, 2))
    o2, r2 = rollout(e2, 9, ea.PeriodicPolicy(3, 2))
    np.testing.assert_array_equal(o1, o2)
    np.testing.assert_array_equal(r1, r2)


def test_different_seeds_differ():
    env = ea.EdgeEngineAwareEnv()
    o1, r1 = rollout(env, 1, ea.PeriodicPolicy(4, 2))
    o2, r2 = rollout(env, 2, ea.PeriodicPolicy(4, 2))
    assert not np.array_equal(o1, o2)
    assert not np.array_equal(r1, r2)


def test_ground_truth_differs_between_seeds():
    env = ea.EdgeEngineAwareEnv()
    env.reset(seed=1)
    m1 = [env.step([0, 0])[4]["ground_truth"]["soil_moisture"] for _ in range(50)]
    env.reset(seed=2)
    m2 = [env.step([0, 0])[4]["ground_truth"]["soil_moisture"] for _ in range(50)]
    assert not np.allclose(m1, m2)


def test_domain_randomisation_changes_physical_parameters_but_not_spaces():
    cfg = ea.default_config()
    cfg.randomization.enabled = True
    env = ea.EdgeEngineAwareEnv(cfg)
    env.reset(seed=1)
    p1 = (env.cfg.storage.capacity_j, env.cfg.communication.tx_energy_j, env.cfg.sensing.noise_std)
    env.reset(seed=2)
    p2 = (env.cfg.storage.capacity_j, env.cfg.communication.tx_energy_j, env.cfg.sensing.noise_std)
    assert p1 != p2
    assert env.observation_space.shape == (18,)
    # the base configuration is untouched
    assert env.base_config.storage.capacity_j == 300.0
    # randomised costs are what the node reports in its observation
    obs, _ = env.reset(seed=3)
    idx = env.obs_builder.index("tx_cost")
    assert obs[idx] == np.float32(env.cfg.communication.tx_energy_j / env.cfg.storage.capacity_j)


def test_randomisation_is_reproducible():
    cfg = ea.default_config()
    cfg.randomization.enabled = True
    e1, e2 = ea.EdgeEngineAwareEnv(cfg), ea.EdgeEngineAwareEnv(cfg)
    o1, r1 = rollout(e1, 42, ea.RuleBasedPolicy(), n=100)
    o2, r2 = rollout(e2, 42, ea.RuleBasedPolicy(), n=100)
    np.testing.assert_array_equal(o1, o2)
