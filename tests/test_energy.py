"""Energy bookkeeping: bounds, conservation, units."""

from __future__ import annotations

import numpy as np
import pytest

import edgeengine_aware as ea
from edgeengine_aware.config import EnergyStorageConfig, HarvestingConfig, TimeConfig
from edgeengine_aware.energy import SimulatedClock, SimulatedEnergyStorage, SolarEnergySource


def test_storage_never_leaves_bounds_under_random_policy():
    e = ea.EdgeEngineAwareEnv()
    e.reset(seed=0)
    cap = e.storage.capacity_j()
    for _ in range(e.max_steps):
        e.step(e.action_space.sample())
        assert 0.0 <= e.storage.energy_j() <= cap + 1e-9


def test_storage_never_negative_when_starved():
    cfg = ea.default_config()
    cfg.storage.initial_soc = 0.01
    cfg.harvesting.max_power_w = 0.0
    e = ea.EdgeEngineAwareEnv(cfg)
    e.reset(seed=0)
    for _ in range(200):
        _, _, _, _, info = e.step([2, 1])
        assert e.storage.energy_j() >= 0.0
        assert info["battery_soc"] >= 0.0
    assert info["metrics"]["battery_depletion_events"] > 0


def test_energy_conservation_over_episode():
    """E_end = E_start + harvested_stored - consumed_delivered, with waste and
    brown-out shortfall accounted for."""
    e = ea.EdgeEngineAwareEnv()
    e.reset(seed=5)
    e_start = e.storage.energy_j()
    for _ in range(e.max_steps):
        e.step(e.action_space.sample())
    m = e.metrics
    consumed = m.total_consumed_energy_j
    harvested = m.total_harvested_energy_j
    wasted = m.wasted_harvest_energy_j
    # the storage is ideal: everything not wasted was stored, everything
    # requested was delivered unless the buffer was empty
    delivered = e_start + harvested - wasted - e.storage.energy_j()
    assert delivered <= consumed + 1e-6
    if m.battery_depletion_events == 0:
        assert delivered == pytest.approx(consumed, abs=1e-6)


def test_per_step_energy_balance():
    e = ea.EdgeEngineAwareEnv()
    e.reset(seed=3)
    for _ in range(300):
        before = e.storage.energy_j()
        wasted_before = e.storage.wasted_j
        _, _, _, _, info = e.step(e.action_space.sample())
        spent = e.cfg.mcu.baseline_power_w * e.timestep_s
        spent += e.cfg.sensing.energy_j[info["sensing_level"]]
        spent += e.cfg.communication.modes[info["tx_mode"]].energy_j if info["tx_attempted"] else 0.0
        wasted = e.storage.wasted_j - wasted_before
        expected = before - spent + info["harvested_energy_j"] - wasted
        assert e.storage.energy_j() == pytest.approx(min(expected, e.storage.capacity_j()), abs=1e-6)


def test_reserve_blocks_optional_operations_but_not_baseline():
    cfg = EnergyStorageConfig(capacity_j=100.0, initial_soc=0.03, reserve_soc=0.02)
    st = SimulatedEnergyStorage(cfg)
    st.reset(np.random.default_rng(0))
    assert st.can_afford(0.5)  # 3 - 0.5 = 2.5 >= 2
    assert not st.can_afford(1.5)  # 3 - 1.5 = 1.5 < 2
    assert st.discharge(5.0) == pytest.approx(3.0)  # baseline may drain to zero
    assert st.energy_j() == 0.0 and st.is_depleted()


def test_charge_clips_and_counts_waste():
    st = SimulatedEnergyStorage(EnergyStorageConfig(capacity_j=10.0, initial_soc=0.9))
    stored = st.charge(5.0)
    assert stored == pytest.approx(1.0)
    assert st.wasted_j == pytest.approx(4.0)
    assert st.energy_j() == pytest.approx(10.0)


def test_solar_profile_is_zero_at_night_and_peaks_at_noon():
    src = SolarEnergySource(HarvestingConfig(), timestep_s=900.0)
    assert src.solar_profile(0.0) == 0.0
    assert src.solar_profile(5.0 * 3600) == 0.0
    assert src.solar_profile(12.0 * 3600) == pytest.approx(1.0)
    assert 0.0 < src.solar_profile(9.0 * 3600) < 1.0


def test_solar_units_are_watts_and_joules():
    cfg = HarvestingConfig(max_power_w=0.010, efficiency=1.0, clearness_mean=1.0, clearness_std=0.0, cloud_noise_std=0.0, measurement_noise_std=0.0)
    src = SolarEnergySource(cfg, timestep_s=900.0)
    src.reset(np.random.default_rng(0), start_time_s=12.0 * 3600)
    assert src.true_power_w() == pytest.approx(0.010)
    assert src.harvested_energy_j() == pytest.approx(0.010 * 900.0)
    assert src.measured_power_w() == pytest.approx(0.010)


def test_harvesting_is_stochastic_between_days():
    cfg = HarvestingConfig()
    src = SolarEnergySource(cfg, timestep_s=900.0)
    src.reset(np.random.default_rng(1), start_time_s=0.0)
    daily = []
    for day in range(6):
        total = 0.0
        for k in range(96):
            t = day * 86400 + k * 900
            src.update(t)
            total += src.harvested_energy_j()
        daily.append(total)
    assert np.std(daily) > 0.0


def test_daily_harvest_is_in_plausible_range():
    """With defaults the node harvests tens of joules per day, i.e. roughly
    the order of magnitude of its daily consumption (see config docstring)."""
    e = ea.EdgeEngineAwareEnv()
    totals = []
    for seed in range(4):
        ea.run_episode(e, ea.PeriodicPolicy(4, 2), seed=seed)
        totals.append(e.metrics.total_harvested_energy_j / 7.0)
    assert 15.0 < np.mean(totals) < 150.0


def test_clock():
    c = SimulatedClock(TimeConfig(timestep_s=900.0, start_hour=23.5))
    assert c.hour_of_day() == pytest.approx(23.5)
    c.advance()
    c.advance()
    assert c.hour_of_day() == pytest.approx(0.0)
    assert c.day_index() == 1
