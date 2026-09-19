"""Gymnasium API behaviour, spaces, termination and information leakage."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import edgeengine_aware as ea
from edgeengine_aware.actions import N_FLAT_ACTIONS, flatten_action, unflatten_action
from edgeengine_aware.observation import OBSERVATION_FIELDS


@pytest.fixture
def env():
    e = ea.EdgeEngineAwareEnv()
    yield e
    e.close()


def test_gymnasium_checker_passes(env):
    check_env(env, skip_render_check=True)


def test_registration_and_make():
    import gymnasium as gym

    e = gym.make("EdgeEngineAware-v0")
    obs, info = e.reset(seed=0)
    assert obs.shape == (len(OBSERVATION_FIELDS),)
    e.close()


def test_reset_and_step_signatures(env):
    out = env.reset(seed=0)
    assert isinstance(out, tuple) and len(out) == 2
    obs, info = out
    assert isinstance(obs, np.ndarray) and obs.dtype == np.float32
    assert isinstance(info, dict)
    out = env.step(env.action_space.sample())
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert "reward_components" in info and "metrics" in info


def test_observations_in_space_for_random_actions(env):
    obs, _ = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    for _ in range(300):
        a = env.action_space.sample()
        assert env.action_space.contains(a)
        obs, *_ = env.step(a)
        assert env.observation_space.contains(obs), obs


def test_observation_dimension_matches_documentation(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] == len(OBSERVATION_FIELDS) == env.obs_builder.dim == 17
    assert list(env.obs_builder.names) == [f.name for f in OBSERVATION_FIELDS]
    assert env.observation_space.shape == (17,)


def test_action_encoding_roundtrip():
    for i in range(N_FLAT_ACTIONS):
        assert flatten_action(unflatten_action(i)) == i
    with pytest.raises(ValueError):
        ea.decode_action([3, 0])
    with pytest.raises(ValueError):
        ea.decode_action([0, 2])


def test_truncation_at_expected_step(env):
    env.reset(seed=0)
    n = env.max_steps
    assert n == int(7 * 86400 / 900) == 672
    for k in range(n):
        _, _, terminated, truncated, _ = env.step([0, 0])
        assert not terminated
        if k < n - 1:
            assert not truncated
    assert truncated


def test_custom_horizon_and_timestep():
    cfg = ea.default_config()
    cfg.time.timestep_s = 3600.0
    cfg.time.episode_days = 2
    e = ea.EdgeEngineAwareEnv(cfg)
    e.reset(seed=0)
    assert e.max_steps == 48
    steps = 0
    truncated = False
    while not truncated:
        _, _, _, truncated, _ = e.step([1, 1])
        steps += 1
    assert steps == 48


def test_termination_on_depletion_option():
    cfg = ea.default_config()
    cfg.terminate_on_depletion = True
    cfg.storage.initial_soc = 0.03
    cfg.harvesting.max_power_w = 0.0
    e = ea.EdgeEngineAwareEnv(cfg)
    e.reset(seed=0)
    terminated = False
    for _ in range(e.max_steps):
        _, _, terminated, truncated, info = e.step([2, 1])
        if terminated:
            break
    assert terminated
    assert info["metrics"]["battery_depletion_events"] >= 1


def test_failed_transmission_consumes_energy_and_updates_nothing_at_app():
    cfg = ea.default_config()
    cfg.communication.min_success_prob = 0.0
    cfg.communication.channel_noise_std = 0.0
    cfg.communication.base_success_prob = 1e-9  # (practically) never delivers
    e = ea.EdgeEngineAwareEnv(cfg)
    e.reset(seed=0)
    e_before = e.storage.energy_j()
    _, _, _, _, info = e.step([2, 1])
    assert info["tx_attempted"] and info["tx_success"] is False
    assert info["reward_components"]["communication_cost"] > 0
    assert info["reward_components"]["application_utility"] == 0.0
    assert e.application.last_packet is None
    assert info["metrics"]["communication_energy_j"] == pytest.approx(cfg.communication.tx_energy_j)
    spent = e_before + info["harvested_energy_j"] - e.storage.energy_j()
    assert spent == pytest.approx(cfg.communication.tx_energy_j + cfg.sensing.energy_j[2] + cfg.mcu.baseline_power_w * 900.0)


def test_transmit_without_measurement_is_rejected(env):
    env.reset(seed=0)
    _, _, _, _, info = env.step([0, 1])
    assert "transmit:no_measurement" in info["rejected"]
    assert not info["tx_attempted"]
    assert info["metrics"]["communication_energy_j"] == 0.0


def test_infeasible_actions_are_rejected_not_executed():
    cfg = ea.default_config()
    cfg.storage.initial_soc = 0.021  # 6.3 J: reserve 6 J + baseline 0.18 J leave < 0.6 J
    cfg.harvesting.max_power_w = 0.0
    e = ea.EdgeEngineAwareEnv(cfg)
    e.reset(seed=0)
    _, _, _, _, info = e.step([2, 1])
    assert info["sensing_level"] == 0 and not info["tx_attempted"]
    assert set(info["rejected"]) == {"sensing:insufficient_energy", "transmit:no_measurement"}
    assert info["reward_components"]["rejection_penalty"] > 0


def test_observation_contains_no_privileged_information(env):
    """Perturb hidden ground truth: the observation must not change."""
    obs0, _ = env.reset(seed=7)
    for _ in range(10):
        env.step([2, 1])
    state_before = env.node_state()
    obs_a = env.obs_builder.build(state_before)
    # mutate every privileged quantity that a leaking builder could read
    env.field.moisture = 0.0
    env.field.temperature = 99.0
    env.source._power_w = 1.0
    env.application._priority = 2  # not yet delivered to the node
    obs_b = env.obs_builder.build(env.node_state())
    np.testing.assert_array_equal(obs_a, obs_b)
    names = env.obs_builder.names
    forbidden = {"true", "future", "clearness", "success_prob", "irradiance"}
    assert not any(any(f in n for f in forbidden) for n in names)


def test_no_future_harvest_in_observation():
    """The harvest observation must reflect the interval that already elapsed."""
    e = ea.EdgeEngineAwareEnv()
    e.reset(seed=11)
    idx = e.obs_builder.index("harvest_power")
    ref = e.cfg.observation.harvest_ref_power_w
    for _ in range(100):
        obs, _, _, _, info = e.step([0, 0])
        measured_last_interval = info["harvested_energy_j"] / e.timestep_s
        # noisy measurement (5 % relative noise) of the power of the elapsed interval
        assert obs[idx] == pytest.approx(min(1.0, measured_last_interval / ref), abs=0.25 * max(measured_last_interval / ref, 1e-3) + 1e-6)


def test_rgb_render_and_ansi_render_without_events():
    e = ea.EdgeEngineAwareEnv(render_mode="rgb_array")
    e.reset(seed=0)
    frame = e.render()  # nothing happened yet: must not fail
    assert frame.ndim == 3 and frame.shape[2] == 3
    for _ in range(5):
        e.step([0, 0])  # no sensing, no transmission
    frame = e.render()
    assert frame.shape[2] == 3
    e.close()
    e2 = ea.EdgeEngineAwareEnv(render_mode="ansi")
    e2.reset(seed=0)
    txt = e2.render()
    assert "battery SoC" in txt


def test_priority_on_uplink_mode_only_updates_after_ack():
    cfg = ea.default_config()
    cfg.communication.priority_update_mode = "on_uplink"
    e = ea.EdgeEngineAwareEnv(cfg)
    e.reset(seed=0)
    idx = e.obs_builder.index("app_priority")
    # starve the application: after aoi_elevated_s its priority rises, but the node
    # must not see it without an acknowledged uplink
    steps = int(cfg.application.aoi_elevated_s / cfg.time.timestep_s) + 2
    for _ in range(steps):
        obs, *_ = e.step([0, 0])
    assert e.application.priority() >= 1
    assert obs[idx] == 0.0


def test_metrics_are_consistent(env):
    res = ea.run_episode(env, ea.PeriodicPolicy(4, 2), seed=2)
    m = env.metrics
    assert m.n_successful_transmissions <= m.n_transmissions
    assert m.n_high_quality_sensing <= m.n_sensing
    assert m.total_consumed_energy_j == pytest.approx(m.baseline_energy_j + m.sensing_energy_j + m.communication_energy_j)
    assert 0.0 <= m.min_battery_soc <= m.average_battery_soc <= 1.0
    assert m.total_reward == pytest.approx(res.total_reward)
    assert m.steps == env.max_steps


def test_initial_harvest_observation_is_not_the_upcoming_interval():
    """At reset the node reports the interval that preceded the episode, not
    the energy it is about to harvest during step 0."""
    cfg = ea.default_config()
    cfg.time.start_hour = 12.0  # noon: harvesting is non-zero and changes between intervals
    cfg.harvesting.measurement_noise_std = 0.0
    e = ea.EdgeEngineAwareEnv(cfg)
    obs0, _ = e.reset(seed=4)
    idx = e.obs_builder.index("harvest_power")
    ref = e.cfg.observation.harvest_ref_power_w
    step0_power = e.source.true_power_w()  # power of the upcoming interval (privileged)
    assert obs0[idx] != pytest.approx(min(1.0, step0_power / ref))
    obs1, _, _, _, info = e.step([0, 0])
    assert obs1[idx] == pytest.approx(min(1.0, info["harvested_energy_j"] / e.timestep_s / ref), abs=1e-6)


def test_unconfirmed_uplinks_keep_link_quality_and_assume_delivery():
    cfg = ea.default_config()
    cfg.communication.ack_available = False
    cfg.communication.min_success_prob = 0.0
    cfg.communication.channel_noise_std = 0.0
    cfg.communication.base_success_prob = 1e-9  # nothing is ever delivered ...
    e = ea.EdgeEngineAwareEnv(cfg)
    e.reset(seed=0)
    obs, _, _, _, info = e.step([2, 1])
    assert info["tx_success"] is False
    ns = e.node_state()
    assert ns.has_reported  # ... but the node assumes it was
    assert ns.link_quality == 1.0
    assert obs[e.obs_builder.index("app_info_age")] < 0.1


def test_render_before_reset_does_not_crash():
    e = ea.EdgeEngineAwareEnv(render_mode="ansi")
    assert "battery SoC" in e.render()


def test_config_validation():
    cfg = ea.default_config()
    cfg.communication.min_success_prob = 0.95
    with pytest.raises(ValueError):
        ea.EdgeEngineAwareEnv(cfg)
    cfg = ea.default_config()
    cfg.sensing.energy_j = (0.0, 0.1)
    with pytest.raises(ValueError):
        ea.EdgeEngineAwareEnv(cfg)
    cfg = ea.default_config()
    cfg.time.start_hour = 24.0
    with pytest.raises(ValueError):
        ea.EdgeEngineAwareEnv(cfg)


def test_human_render_on_headless_backend_closes_figures():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    for _ in range(3):
        e = ea.EdgeEngineAwareEnv(render_mode="human")
        e.reset(seed=0)
        e.step([2, 1])
        e.render()
        e.close()
    assert len(plt.get_fignums()) == before
