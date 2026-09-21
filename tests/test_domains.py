"""The three application domains: configuration, hidden-world models, shared
contract (same spaces and observation semantics) and cross-domain plumbing."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import edgeengine_aware as ea
from edgeengine_aware.config import QuantityConfig, ScheduleConfig
from edgeengine_aware.domains import DOMAIN_NAMES, build_world, domain_config, indoor_air_config, industrial_config
from edgeengine_aware.observation import ObservationBuilder
from edgeengine_aware.policies import PeriodicPolicy, run_episode
from edgeengine_aware.process import ActivitySchedule
from edgeengine_aware.rl import MixedScenarioEnv, make_env
from edgeengine_aware.scenarios import get_scenario, scenario_names, split_scenario_name

DAY = 86400.0
H = 3600.0


# ---------------------------------------------------------------------------
# QuantityConfig: direction-aware thresholds
# ---------------------------------------------------------------------------
def test_quantity_zones_both_directions():
    low = QuantityConfig(warning_threshold=0.35, critical_threshold=0.25, critical_is_upper=False)
    assert [low.zone(v) for v in (0.5, 0.3, 0.2)] == [0, 1, 2]
    assert low.beyond_critical(0.25) and not low.beyond_critical(0.26)
    up = QuantityConfig(warning_threshold=0.4, critical_threshold=0.7, critical_is_upper=True)
    assert [up.zone(v) for v in (0.2, 0.5, 0.8)] == [0, 1, 2]
    assert up.beyond_critical(0.7) and not up.beyond_critical(0.69)
    assert up.to_physical(up.to_normalised(1234.0)) == pytest.approx(1234.0)
    with pytest.raises(ValueError):
        QuantityConfig(warning_threshold=0.7, critical_threshold=0.4, critical_is_upper=True).validate()


def test_importance_follows_the_danger_side():
    ob_low = ObservationBuilder(ea.NodeProfile.from_config(domain_config("agriculture")))
    ob_up = ObservationBuilder(ea.NodeProfile.from_config(domain_config("indoor_air")))
    assert ob_low.importance(0.10, True) == 1.0 and ob_low.importance(0.90, True) < 0.01
    assert ob_up.importance(0.90, True) == 1.0 and ob_up.importance(0.05, True) < 0.05
    assert ob_up.importance(0.5, False) == 0.0


# ---------------------------------------------------------------------------
# Weekly schedule
# ---------------------------------------------------------------------------
def test_schedule_week_pattern():
    cfg = ScheduleConfig(p_day_off=0.0, p_extra_day=0.0, noise_std=0.0)
    sch = ActivitySchedule(cfg, start_weekday=0, horizon_days=9)
    sch.reset(np.random.default_rng(0), 0.0)
    assert sch.level(0 * DAY + 12 * H) > 0.5  # Monday noon
    assert sch.level(0 * DAY + 13 * H) < sch.level(0 * DAY + 11 * H)  # lunch dip
    assert sch.level(0 * DAY + 3 * H) == 0.0  # night
    assert sch.level(5 * DAY + 12 * H) == 0.0 and sch.level(6 * DAY + 12 * H) == 0.0  # weekend
    sch2 = ActivitySchedule(cfg, start_weekday=5, horizon_days=9)  # episode starts on a Saturday
    sch2.reset(np.random.default_rng(0), 0.0)
    assert sch2.level(12 * H) == 0.0 and sch2.level(2 * DAY + 12 * H) > 0.5


def test_schedule_exceptions_and_reproducibility():
    cfg = ScheduleConfig(p_day_off=1.0, p_extra_day=1.0)  # every weekday off, every weekend day on
    sch = ActivitySchedule(cfg, start_weekday=0, horizon_days=9)
    sch.reset(np.random.default_rng(1), 0.0)
    assert sch.level(0 * DAY + 12 * H) == 0.0 and sch.level(5 * DAY + 12 * H) > 0.0
    a = ActivitySchedule(ScheduleConfig(), 0, 9)
    b = ActivitySchedule(ScheduleConfig(), 0, 9)
    a.reset(np.random.default_rng(7), 0.0)
    b.reset(np.random.default_rng(7), 0.0)
    for _ in range(50):
        a.step()
        b.step()
    assert a.level(2 * DAY + 10 * H) == b.level(2 * DAY + 10 * H)


# ---------------------------------------------------------------------------
# Hidden-world models
# ---------------------------------------------------------------------------
def test_indoor_air_rises_with_people_and_lights_bring_energy():
    cfg = indoor_air_config()
    cfg.schedule.p_day_off = cfg.schedule.p_extra_day = 0.0
    cfg.schedule.noise_std = 0.0
    sch, proc, src = build_world(cfg, 900.0)
    rng = np.random.default_rng(0)
    sch.reset(rng, 0.0)
    proc.reset(rng, 0.0)
    src.reset(rng, 0.0)
    ppm = []
    power = []
    for k in range(96 * 7):
        t = k * 900.0
        sch.step()
        st = proc.step(t)
        src.update(t + 900.0)
        ppm.append(st.aux["co2_ppm"])
        power.append(src.true_power_w())
    ppm, power = np.array(ppm), np.array(power)
    hours = (np.arange(96 * 7) * 0.25) % 24
    days = np.arange(96 * 7) // 96
    workday_afternoon = (days < 5) & (hours > 14) & (hours < 17)
    night = hours < 5
    assert ppm[workday_afternoon].mean() > 900 and ppm[night].mean() < 600
    assert power[workday_afternoon].min() > 50e-6  # lights on
    assert power[night].max() == 0.0  # no light at all at night
    weekend_noon = (days >= 5) & (hours > 11) & (hours < 14)
    assert power[weekend_noon].max() < 50e-6  # only daylight through the window
    assert 0.0 <= proc.value <= 1.0


def test_industrial_thermal_and_teg_coupling():
    cfg = industrial_config()
    cfg.schedule.p_day_off = cfg.schedule.p_extra_day = 0.0
    cfg.schedule.noise_std = 0.0
    cfg.industrial.fault_onsets_per_day = 0.0
    sch, proc, src = build_world(cfg, 900.0)
    rng = np.random.default_rng(0)
    sch.reset(rng, 0.0)
    proc.reset(rng, 0.0)
    src.reset(rng, 0.0)
    temps, power, loads, dts = [], [], [], []
    for k in range(96 * 2):
        t = k * 900.0
        sch.step()
        st = proc.step(t)
        src.update(t + 900.0)
        temps.append(st.aux["temperature_c"])
        power.append(src.true_power_w())
        loads.append(st.aux["load"])
        dts.append(st.aux["temperature_c"] - st.aux["ambient_c"])
    temps, power, loads, dts = map(np.array, (temps, power, loads, dts))
    running = loads > 0.5
    assert temps[running].max() > 55.0  # warms up under load
    assert temps[~running].min() < 30.0  # cools down at night
    assert power[running].max() > 0.5e-3  # a few mW while running
    assert power[dts < cfg.thermoelectric.min_dt_c].max() == 0.0  # the converter needs a minimum temperature difference
    # a fault drives the temperature to the critical zone; maintenance eventually repairs it
    cfg2 = industrial_config()
    cfg2.industrial.fault_onsets_per_day = 50.0
    cfg2.industrial.initial_health_range = (0.2, 0.2)
    cfg2.industrial.maintenance_delay_mean_s = 3600.0
    sch, proc, _ = build_world(cfg2, 900.0)
    sch.reset(np.random.default_rng(1), 0.0)
    proc.reset(np.random.default_rng(1), 0.0)
    zones, maint = [], 0
    for k in range(96 * 5):
        sch.step()
        st = proc.step(k * 900.0)
        zones.append(st.zone)
        maint = st.aux["maintenance_events"]
    assert max(zones) == 2 and maint >= 1


# ---------------------------------------------------------------------------
# Environment contract across domains
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("domain", DOMAIN_NAMES)
def test_domain_env_passes_checker_and_shares_the_contract(domain):
    ref = ea.EdgeEngineAwareEnv(domain_config("agriculture"))
    env = ea.EdgeEngineAwareEnv(domain_config(domain))
    check_env(env, skip_render_check=True)
    assert env.observation_space == ref.observation_space
    assert env.action_space == ref.action_space
    obs, info = env.reset(seed=0)
    assert info["ground_truth"]["domain"] == domain
    assert 0.0 <= info["ground_truth"]["value"] <= 1.0
    if domain != "agriculture":
        assert info["ground_truth"]["activity"] is not None


@pytest.mark.parametrize("domain", ("indoor_air", "industrial"))
def test_domain_reproducible_and_priority_direction(domain):
    env = ea.EdgeEngineAwareEnv(domain_config(domain))
    pol = PeriodicPolicy(4, 2, 2)
    r1 = run_episode(env, pol, seed=5)
    m1 = env.metrics.as_dict()
    r2 = run_episode(env, pol, seed=5)
    assert r1.total_reward == pytest.approx(r2.total_reward)
    assert env.metrics.as_dict()["total_harvested_energy_j"] == pytest.approx(m1["total_harvested_energy_j"])
    assert m1["total_harvested_energy_j"] > 0
    # the application escalates when the (high-is-bad) quantity is beyond the thresholds
    env.reset(seed=0)
    q = env.quantity
    env.application.last_packet = None
    from edgeengine_aware.interfaces import Measurement, Packet

    env.application.last_packet = Packet(Measurement(q.critical_threshold + 0.05, 0.0, 2, 0.01), 0.0)
    env.application.last_received_at_s = 0.0
    assert env.application._compute_priority(0.0) == 2
    env.application.last_packet = Packet(Measurement(q.warning_threshold - 0.05, 0.0, 2, 0.01), 0.0)
    assert env.application._compute_priority(0.0) == 0


def test_scenario_names_and_mixtures():
    assert split_scenario_name("industrial:degrading") == ("industrial", "degrading")
    assert split_scenario_name("cloudy_week") == ("agriculture", "cloudy_week")
    names = scenario_names("all")
    assert "cloudy_week" in names and "indoor_air:no_window" in names and "industrial:continuous" in names
    assert get_scenario("indoor_air:no_window").indoor_light.daylight_lux == 0.0
    with pytest.raises(KeyError):
        get_scenario("indoor_air:nope")
    env = make_env("all")
    assert isinstance(env, MixedScenarioEnv)
    seen = set()
    for s in range(12):
        _, info = env.reset(seed=s)
        seen.add(info["ground_truth"]["domain"])
    assert len(seen) == 3
    env_dom = make_env("mixed:industrial")
    assert all(n.startswith("industrial:") for n in env_dom.scenario_names)


def test_agriculture_defaults_unchanged():
    cfg = ea.default_config()
    assert cfg.domain == "agriculture" and cfg.harvesting_source == "solar" and not cfg.uses_schedule
    env = ea.EdgeEngineAwareEnv(cfg)
    assert env.schedule is None
    env.reset(seed=0)
    assert env.field is env.process


def test_thermoelectric_requires_industrial_process():
    cfg = ea.default_config()
    cfg.harvesting_source = "thermoelectric"
    with pytest.raises(ValueError):
        cfg.validate()
