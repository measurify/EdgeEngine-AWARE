"""Scenarios, RL wrappers, evaluation protocol and the numpy MLP runtime."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

import edgeengine_aware as ea
from edgeengine_aware.actions import N_FLAT_ACTIONS, unflatten_action
from edgeengine_aware.rl import NumpyMLPPolicy, evaluate, make_env, summarize
from edgeengine_aware.scenarios import SCENARIOS, get_scenario


def test_all_scenarios_build_valid_environments():
    for name in SCENARIOS:
        env = make_env(name, seed=0)
        obs, _, _, _, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        env.close()


def test_get_scenario_returns_fresh_copies():
    a, b = get_scenario("cloudy_week"), get_scenario("cloudy_week")
    a.storage.capacity_j = 1.0
    assert b.storage.capacity_j != 1.0
    with pytest.raises(KeyError):
        get_scenario("no_such_scenario")


def test_flat_action_wrapper_roundtrip():
    env = make_env("default", flat_actions=True)
    assert isinstance(env.action_space, spaces.Discrete) and env.action_space.n == N_FLAT_ACTIONS == 12
    env.reset(seed=0)
    _, _, _, _, info = env.step(10)  # = (2, 2): high-quality sensing + transmit in the standard mode
    assert info["sensing_level"] == 2 and info["tx_attempted"] and info["tx_mode"] == 1
    assert np.array_equal(unflatten_action(10), [2, 2])


def test_evaluate_and_summarize_shapes():
    rows = evaluate({"rule": ea.RuleBasedPolicy, "periodic": lambda: ea.PeriodicPolicy(4, 2)}, ["default", "tiny_battery"], seeds=range(1000, 1002))
    assert len(rows) == 2 * 2 * 2
    summ = summarize(rows, "reward")
    assert set(summ) == {"default", "tiny_battery"} and set(summ["default"]) == {"rule", "periodic"}
    d = rows[0].as_dict()
    assert "c_application_utility" in d and d["scenario"] == "default"


def test_numpy_mlp_policy_argmax_decoding():
    # 17 -> 4 (tanh) -> 5 logits: force sensing level 1 and transmit 1
    W1 = np.zeros((4, 17)); b1 = np.zeros(4)
    W2 = np.zeros((5, 4)); b2 = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
    model = {"type": "mlp", "input_dim": 17, "layers": [{"W": W1.tolist(), "b": b1.tolist(), "activation": "tanh"}, {"W": W2.tolist(), "b": b2.tolist(), "activation": "linear"}], "output": "multidiscrete_logits", "output_split": [3, 2]}
    pol = NumpyMLPPolicy(model)
    assert np.array_equal(pol.act(np.zeros(17, dtype=np.float32)), [1, 1])
    flat = {"type": "mlp", "input_dim": 17, "layers": [{"W": np.zeros((6, 17)).tolist(), "b": [0, 0, 0, 0, 1.0, 0], "activation": "linear"}], "output": "flat_q_values"}
    assert np.array_equal(NumpyMLPPolicy(flat).act(np.zeros(17)), unflatten_action(4))


def test_sb3_export_matches_model_if_available():
    sb3 = pytest.importorskip("stable_baselines3")
    from edgeengine_aware.rl import SB3Policy, export_sb3_mlp

    env = make_env("default")
    model = sb3.PPO("MlpPolicy", env, n_steps=64, batch_size=64, policy_kwargs=dict(net_arch=[16, 16]), seed=0, device="cpu", verbose=0)
    weights = export_sb3_mlp(model)
    runtime, ref = NumpyMLPPolicy(weights), SB3Policy(model)
    obs, _ = env.reset(seed=3)
    for _ in range(100):
        assert np.array_equal(runtime.act(obs), ref.act(obs))
        obs, *_ = env.step(env.action_space.sample())


def test_frame_stacker_matches_vec_frame_stack():
    sb3 = pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    from edgeengine_aware.rl import FrameStacker, make_env_fn

    venv = VecFrameStack(DummyVecEnv([make_env_fn("default", randomize=False, seed=0)]), n_stack=4)
    venv.seed(5)
    stacked = venv.reset()
    env = make_env("default")
    obs, _ = env.reset(seed=5)
    fs = FrameStacker(4, 18)
    assert np.allclose(stacked[0], fs.push(obs))
    for k in range(20):
        a = np.array([[2, 1]]) if k % 3 == 0 else np.array([[0, 0]])
        stacked, *_ = venv.step(a)
        obs, *_ = env.step(a[0])
        assert np.allclose(stacked[0], fs.push(obs))
    assert sb3 is not None


def test_stacked_policy_wraps_plain_policy():
    from edgeengine_aware.rl import StackedPolicy

    class Probe:
        def __init__(self):
            self.dims = []

        def reset(self):
            pass

        def act(self, o):
            self.dims.append(len(o))
            return np.array([0, 0])

    inner = Probe()
    pol = StackedPolicy(inner, 3)
    env = make_env("default")
    obs, _ = env.reset(seed=0)
    pol.reset()
    for _ in range(3):
        obs, *_ = env.step(pol.act(obs))
    assert inner.dims == [54, 54, 54]


def test_rule_based_v2_is_lazier_when_battery_is_low():
    from edgeengine_aware.observation import OBSERVATION_FIELDS

    idx = {f.name: i for i, f in enumerate(OBSERVATION_FIELDS)}
    pol = ea.RuleBasedPolicy()
    obs = np.zeros(len(OBSERVATION_FIELDS), dtype=np.float32)
    obs[idx["path_loss_est"]] = 1.0
    obs[idx["measurement_quality"]] = 0.8
    obs[idx["measurement"]] = obs[idx["reported_value"]] = 0.5
    obs[idx["measurement_age"]] = obs[idx["time_since_tx_success"]] = obs[idx["app_info_age"]] = 3.0 / 24  # 3 h
    obs[idx["link_quality"]] = 1.0
    obs[idx["battery_soc"]] = 0.8
    assert np.array_equal(pol.act(obs), [2, 2])  # normal mode: 3 h > 2 h -> report (standard radio mode)
    obs[idx["battery_soc"]] = 0.4
    assert np.array_equal(pol.act(obs), [0, 0])  # economy: interval is 4 h -> wait, and no checks
    obs[idx["battery_soc"]] = 0.1
    assert np.array_equal(pol.act(obs), [0, 0])  # deep economy: 8 h
    obs[idx["app_info_age"]] = 9.0 / 24
    assert np.array_equal(pol.act(obs), [2, 2])  # ... but still a high-quality report when due


def test_rule_based_picks_radio_mode_from_path_loss_estimate():
    from edgeengine_aware.observation import OBSERVATION_FIELDS, NodeProfile

    cfg = ea.default_config()
    prof = NodeProfile.from_config(cfg)
    idx = {f.name: i for i, f in enumerate(OBSERVATION_FIELDS)}
    pol = ea.RuleBasedPolicy(profile=prof)
    obs = np.zeros(len(OBSERVATION_FIELDS), dtype=np.float32)
    obs[idx["measurement_quality"]] = 0.8
    obs[idx["measurement"]] = obs[idx["reported_value"]] = 0.5
    obs[idx["measurement_age"]] = obs[idx["time_since_tx_success"]] = obs[idx["app_info_age"]] = 3.0 / 24
    obs[idx["link_quality"]] = 1.0
    obs[idx["battery_soc"]] = 0.8
    o = cfg.observation

    def with_pl(pl_db):
        obs[idx["path_loss_est"]] = (pl_db - o.path_loss_min_db) / (o.path_loss_max_db - o.path_loss_min_db)
        return pol.act(obs)

    obs[idx["path_loss_est"]] = 1.0  # no estimate -> reference (standard) mode
    assert np.array_equal(pol.act(obs), [2, 2])
    assert np.array_equal(with_pl(130.0), [2, 1])  # fast mode has +7 dB -> cheapest
    assert np.array_equal(with_pl(139.0), [2, 2])  # fast -2 dB, standard +4 dB -> standard
    assert np.array_equal(with_pl(146.0), [2, 3])  # only robust has >= 4 dB
    assert np.array_equal(with_pl(160.0), [2, 3])  # nothing qualifies -> most robust
    obs[idx["link_quality"]] = 0.5  # recent failures -> escalate one mode
    assert np.array_equal(with_pl(130.0), [2, 2])
