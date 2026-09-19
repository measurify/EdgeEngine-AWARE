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
    assert isinstance(env.action_space, spaces.Discrete) and env.action_space.n == N_FLAT_ACTIONS
    env.reset(seed=0)
    _, _, _, _, info = env.step(5)  # = (2, 1): high-quality sensing + transmit
    assert info["sensing_level"] == 2 and info["tx_attempted"]
    assert np.array_equal(unflatten_action(5), [2, 1])


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
