"""Sim-to-real scaffolding: shared observation builder, mock backend, export."""

from __future__ import annotations

import json

import numpy as np
import pytest

import edgeengine_aware as ea
from edgeengine_aware.actions import plan_execution
from edgeengine_aware.deployment import HardwareBackend, NodeController, PolicyBundle, export_policy, make_mock_backend
from edgeengine_aware.interfaces import Clock, EnergySource, EnergyStorage, Policy, Radio, RemoteApplication, Sensor
from edgeengine_aware.observation import OBSERVATION_FIELDS, NodeProfile


def test_mock_backend_implements_protocols():
    profile = NodeProfile.from_config(ea.default_config())
    hw = make_mock_backend(profile)
    assert isinstance(hw.clock, Clock)
    assert isinstance(hw.storage, EnergyStorage)
    assert isinstance(hw.source, EnergySource)
    assert isinstance(hw.sensor, Sensor)
    assert isinstance(hw.radio, Radio)
    assert isinstance(hw.application, RemoteApplication)
    assert isinstance(ea.RuleBasedPolicy(), Policy)


def test_simulated_subsystems_implement_protocols():
    env = ea.EdgeEngineAwareEnv()
    env.reset(seed=0)
    assert isinstance(env.clock, Clock)
    assert isinstance(env.storage, EnergyStorage)
    assert isinstance(env.source, EnergySource)
    assert isinstance(env.sensor, Sensor)
    assert isinstance(env.radio, Radio)
    assert isinstance(env.application, RemoteApplication)


def test_controller_runs_policy_on_mock_hardware():
    cfg = ea.default_config()
    profile = NodeProfile.from_config(cfg)
    hw = make_mock_backend(profile)
    ctrl = NodeController(hw, profile, ea.RuleBasedPolicy())
    n_tx = 0
    socs = []
    for _ in range(96):
        report = ctrl.run_cycle()
        assert report.observation.shape == (len(OBSERVATION_FIELDS),)
        assert 0.0 <= report.observation.min() and report.observation.max() <= 1.0
        n_tx += int(report.executed_transmit)
        hw.end_of_cycle(report)
        socs.append(hw.storage.energy_j() / hw.storage.capacity_j())
    assert n_tx > 0
    assert len(hw.radio.sent) == n_tx
    assert np.std(socs) > 0.0  # the mock power path actually moves the battery


class _Replay:
    """Drivers that replay the *measurable* readings recorded from a simulated
    episode (no ground truth is stored)."""

    def __init__(self, rec):
        self.rec = rec
        self.k = 0

    # Clock
    def now_s(self):
        return self.rec["time"][self.k]

    def time_of_day_s(self):
        return self.rec["time"][self.k] % 86400.0

    # EnergyStorage
    def capacity_j(self):
        return self.rec["capacity"]

    def energy_j(self):
        return self.rec["energy"][self.k]

    # EnergySource
    def measured_power_w(self):
        return self.rec["harvest"][self.k]

    # Sensor
    def read(self, level, now_s):
        m = self.rec["measurement"][self.k]
        assert m is not None and m.level == level
        return m

    def energy_cost_j(self, level):
        return self.rec["profile"].sensing_energy_j[level]

    def noise_std(self, level):
        return self.rec["profile"].sensing_noise_std[level]

    # Radio
    def transmit(self, packet, mode):
        return self.rec["ack"][self.k]

    def tx_energy_j(self, mode):
        return self.rec["profile"].tx_energy_j[mode]

    def n_modes(self):
        return self.rec["profile"].n_modes

    # RemoteApplication (downlink): what the node reads at wake-up in 'immediate'
    # mode, or the content of the downlink that came with the ACK in 'on_uplink' mode
    def priority(self):
        return self.rec["priority"][self.k] if self.rec["mode"] == "immediate" else self.rec["downlink"][self.k]


class _Scripted:
    def __init__(self, actions):
        self.actions, self.k = actions, 0

    def reset(self):
        self.k = 0

    def act(self, obs):
        a = self.actions[self.k]
        self.k += 1
        return a


@pytest.mark.parametrize("mode", ["immediate", "on_uplink"])
def test_firmware_loop_reproduces_simulator_observations(mode):
    """Record the driver-level readings of a simulated episode (energy, harvest,
    priority downlink, sensor samples, ACKs) and replay them through the
    firmware controller: the observation vectors must be identical step by
    step. This is the sim-to-real contract."""
    cfg = ea.default_config()
    cfg.communication.priority_update_mode = mode
    profile = NodeProfile.from_config(cfg)
    env = ea.EdgeEngineAwareEnv(cfg)
    env.action_space.seed(3)
    actions = [env.action_space.sample() for _ in range(200)]

    rec = {"time": [], "energy": [], "harvest": [], "priority": [], "downlink": [], "measurement": [], "ack": [], "capacity": 0.0, "profile": profile, "mode": mode}
    sim_obs = []
    obs, _ = env.reset(seed=5)
    # tap the drivers of the simulator (subsystems are rebuilt at reset, so tap afterwards)
    real_read, real_tx = env.sensor.read, env.radio.transmit
    last = {"m": None, "ack": None}

    def tapped_read(level, now_s):
        m = real_read(level, now_s)
        last["m"] = m
        return m

    def tapped_tx(packet, mode):
        res = real_tx(packet, mode)
        last["ack"] = res
        return res

    env.sensor.read, env.radio.transmit = tapped_read, tapped_tx
    rec["capacity"] = env.storage.capacity_j()
    for a in actions:
        sim_obs.append(obs.copy())
        rec["time"].append(env.clock.now_s())
        rec["energy"].append(env.storage.energy_j())
        rec["harvest"].append(env._last_measured_harvest_w)
        rec["priority"].append(env.application.priority())
        last = {"m": None, "ack": None}
        obs, *_ = env.step(a)
        rec["measurement"].append(last["m"])
        rec["ack"].append(last["ack"])
        rec["downlink"].append(env.tracker.priority)  # priority carried by the ACK (if any)

    drv = _Replay(rec)
    hw = HardwareBackend(clock=drv, storage=drv, source=drv, sensor=drv, radio=drv, application=drv)
    ctrl = NodeController(hw, profile, _Scripted(actions), priority_update_mode=mode)
    n_tx = 0
    for k, a in enumerate(actions):
        drv.k = k
        report = ctrl.run_cycle()
        np.testing.assert_array_equal(report.observation, sim_obs[k], err_msg=f"step {k}")
        assert report.requested_action == (int(a[0]), int(a[1]))
        n_tx += int(report.executed_transmit)
    assert n_tx > 0 and n_tx == sum(1 for x in rec["ack"] if x is not None)


def test_node_state_has_only_measurable_fields():
    env = ea.EdgeEngineAwareEnv()
    env.reset(seed=0)
    env.step([2, 1])
    ns = env.node_state()
    field_names = set(vars(ns).keys())
    for forbidden in ("true", "soil_moisture", "clearness", "success_prob", "future"):
        assert not any(forbidden in f for f in field_names)


def test_plan_execution_shared_rule():
    modes = (0.3, 0.6, 1.2)
    plan = plan_execution([2, 2], stored_energy_j=10.0, baseline_energy_j=0.2, reserve_energy_j=0.2, sensing_energy_j=(0.0, 0.1, 0.6), tx_energy_j=modes, has_measurement=False)
    assert plan.sensing_level == 2 and plan.transmit and plan.mode == 1 and plan.tx_energy_j == 0.6 and plan.rejected == ()
    plan = plan_execution([2, 3], stored_energy_j=1.5, baseline_energy_j=0.2, reserve_energy_j=0.2, sensing_energy_j=(0.0, 0.1, 0.6), tx_energy_j=modes, has_measurement=True)
    assert plan.sensing_level == 2 and not plan.transmit and plan.mode == -1 and plan.rejected == ("transmit:insufficient_energy",)
    plan = plan_execution([2, 1], stored_energy_j=1.5, baseline_energy_j=0.2, reserve_energy_j=0.2, sensing_energy_j=(0.0, 0.1, 0.6), tx_energy_j=modes, has_measurement=True)
    assert plan.transmit and plan.mode == 0 and plan.tx_energy_j == 0.3  # the cheap mode still fits
    plan = plan_execution([0, 1], stored_energy_j=10.0, baseline_energy_j=0.2, reserve_energy_j=0.2, sensing_energy_j=(0.0, 0.1, 0.6), tx_energy_j=modes, has_measurement=False)
    assert plan.rejected == ("transmit:no_measurement",) and plan.tx_energy_j == 0.0
    # a scalar energy is a single-mode radio with the binary action
    plan = plan_execution([1, 1], stored_energy_j=10.0, baseline_energy_j=0.2, reserve_energy_j=0.2, sensing_energy_j=(0.0, 0.1, 0.6), tx_energy_j=0.6, has_measurement=True)
    assert plan.transmit and plan.mode == 0
    with pytest.raises(ValueError):
        plan_execution([1, 2], stored_energy_j=10.0, baseline_energy_j=0.2, reserve_energy_j=0.2, sensing_energy_j=(0.0, 0.1, 0.6), tx_energy_j=0.6, has_measurement=True)


def test_policy_export_roundtrip(tmp_path):
    cfg = ea.default_config()
    profile = NodeProfile.from_config(cfg)
    bundle = export_policy(ea.RuleBasedPolicy(), profile, policy_type="rule_based", notes="test")
    path = bundle.save(tmp_path / "policy.json")
    loaded = PolicyBundle.load(path)
    assert loaded.observation_names == [f.name for f in OBSERVATION_FIELDS]
    assert loaded.observation_dim == 18
    assert loaded.action_encoding["nvec"] == [3, 4] and loaded.action_encoding["n_flat"] == 12
    assert loaded.model["soc_critical"] == ea.RuleBasedParams().soc_critical
    assert loaded.profile["tx_energy_j"] == [m.energy_j for m in cfg.communication.modes]
    json.loads(path.read_text())  # valid JSON


def test_export_of_mlp_like_parameters(tmp_path):
    profile = NodeProfile.from_config(ea.default_config())
    w = {"layers": [{"W": np.zeros((18, 8)).tolist(), "b": [0.0] * 8, "act": "relu"}, {"W": np.zeros((8, 12)).tolist(), "b": [0.0] * 12, "act": "linear"}]}

    class Dummy:
        def act(self, o):
            return np.array([0, 0])

        def reset(self):
            pass

    b = export_policy(Dummy(), profile, policy_type="mlp", model=w)
    b.save(tmp_path / "mlp.json")
    assert PolicyBundle.load(tmp_path / "mlp.json").model["layers"][1]["W"][0] == [0.0] * 12
