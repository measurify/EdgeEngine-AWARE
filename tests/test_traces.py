"""Trace-driven backends: loader semantics, replay correctness and the
invariance of the observation contract."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from edgeengine_aware import EdgeEngineAwareEnv, NodeProfile, RuleBasedParams, RuleBasedPolicy, default_config, run_episode
from edgeengine_aware.traces import INSTANT, PRECEDING_MEAN, Trace, TraceColumn, TraceDrivenEnv, TraceFieldEnvironment, TraceSolarEnergySource, TraceWindow

DEMO = Path(__file__).resolve().parents[1] / "data" / "traces" / "demo_liguria_2023_hourly.csv"
H = 3600.0
DAY = 86400.0


def _tiny_trace(n_days: int = 4) -> Trace:
    t = np.arange(0, n_days * 24 + 1) * H
    hours = (t / H) % 24
    ghi = np.clip(800 * np.sin(np.pi * (hours - 6) / 12), 0, None)
    ghi[hours <= 6] = 0.0
    ghi[hours >= 18] = 0.0
    moist = 0.30 - 0.0005 * (t / H)  # slow drying, m³/m³
    rain = np.zeros_like(t)
    rain[30] = 5.0  # 5 mm in hour 29→30
    return Trace(
        t,
        {
            "shortwave_radiation": TraceColumn(ghi, PRECEDING_MEAN, "W/m²"),
            "soil_moisture_0_to_7cm": TraceColumn(moist, INSTANT, "m³/m³"),
            "temperature_2m": TraceColumn(20 + 5 * np.cos(2 * np.pi * (hours - 15) / 24), INSTANT),
            "relative_humidity_2m": TraceColumn(np.full_like(t, 65.0), INSTANT),
            "precipitation": TraceColumn(rain, PRECEDING_MEAN, "mm"),
        },
        {"site": "unit test"},
    )


# ---------------------------------------------------------------------------
# Trace container
# ---------------------------------------------------------------------------
def test_instant_column_is_linearly_interpolated():
    tr = _tiny_trace()
    v0 = tr.value_at("soil_moisture_0_to_7cm", 0.0)
    v1 = tr.value_at("soil_moisture_0_to_7cm", H)
    assert tr.value_at("soil_moisture_0_to_7cm", 0.5 * H) == pytest.approx(0.5 * (v0 + v1))
    # exact integral of a linear function over an interval
    assert tr.interval_mean("soil_moisture_0_to_7cm", 0.0, H) == pytest.approx(0.5 * (v0 + v1))


def test_preceding_mean_column_is_piecewise_constant():
    tr = _tiny_trace()
    # the value stamped at 10:00 holds on (09:00, 10:00]
    v10 = tr.columns["shortwave_radiation"].values[10]
    assert tr.value_at("shortwave_radiation", 9.25 * H) == pytest.approx(v10)
    assert tr.value_at("shortwave_radiation", 10.0 * H) == pytest.approx(v10)
    assert tr.interval_mean("shortwave_radiation", 9.0 * H, 10.0 * H) == pytest.approx(v10)
    # a 15-minute interval fully inside the hour has exactly the hourly mean
    assert tr.interval_mean("shortwave_radiation", 9.25 * H, 9.5 * H) == pytest.approx(v10)
    # straddling two hours: time-weighted mean
    v11 = tr.columns["shortwave_radiation"].values[11]
    assert tr.interval_mean("shortwave_radiation", 9.5 * H, 10.5 * H) == pytest.approx(0.5 * (v10 + v11))


def test_interval_sum_conserves_precipitation():
    tr = _tiny_trace()
    total = sum(tr.interval_sum("precipitation", k * 900.0, (k + 1) * 900.0) for k in range(4 * 24 * 4))
    assert total == pytest.approx(5.0)
    assert tr.interval_sum("precipitation", 29 * H, 30 * H) == pytest.approx(5.0)
    assert tr.interval_sum("precipitation", 30 * H, 31 * H) == pytest.approx(0.0)


def test_daily_means_and_slicing():
    tr = _tiny_trace(4)
    dm = tr.daily_means("shortwave_radiation")
    assert dm.shape == (4,)
    assert np.allclose(dm, dm[0])  # identical days
    sub = tr.slice_days(1, 2)
    assert sub.n_days == 2
    assert sub.start_s == 0.0
    assert sub.value_at("soil_moisture_0_to_7cm", 0.0) == pytest.approx(tr.value_at("soil_moisture_0_to_7cm", DAY))


def test_csv_roundtrip(tmp_path):
    tr = _tiny_trace(2)
    tr.meta["start_local"] = "2023-06-01T00:00"
    p = tmp_path / "t.csv"
    tr.to_csv(p, time_format="iso")
    back = Trace.from_csv(p)
    assert back.meta["site"] == "unit test"
    assert back.n_days == 2
    np.testing.assert_allclose(back.time_s, tr.time_s)
    for name, col in tr.columns.items():
        assert back.columns[name].kind == col.kind
        np.testing.assert_allclose(back.columns[name].values, col.values, rtol=1e-5)


def test_iso_timestamps_not_starting_at_midnight(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("time,a\n2023-06-01T05:00,1\n2023-06-01T06:00,2\n2023-06-01T07:00,3\n")
    tr = Trace.from_csv(p)
    assert tr.start_s == 5 * H
    assert tr.columns["a"].kind == INSTANT
    tr2 = Trace.from_csv(p, kinds={"a": PRECEDING_MEAN})
    assert tr2.columns["a"].kind == PRECEDING_MEAN


def test_bad_traces_are_rejected():
    with pytest.raises(ValueError):
        Trace(np.array([0.0, 0.0]), {"a": TraceColumn(np.zeros(2))})
    with pytest.raises(ValueError):
        Trace(np.array([0.0, 1.0]), {"a": TraceColumn(np.zeros(3))})


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def test_solar_source_scales_irradiance_to_panel_power():
    tr = _tiny_trace()
    cfg = default_config()
    cfg.harvesting.measurement_noise_std = 0.0
    dt = cfg.time.timestep_s
    win = TraceWindow(tr, 1 * DAY, start_day=0)
    src = TraceSolarEnergySource(tr, cfg.harvesting, dt, win)
    src.reset(np.random.default_rng(0), 12 * H)
    ghi = tr.interval_mean("shortwave_radiation", 12 * H, 12 * H + dt)
    assert src.true_power_w() == pytest.approx(cfg.harvesting.max_power_w * cfg.harvesting.efficiency * ghi / 1000.0)
    assert src.measured_power_w() == pytest.approx(src.true_power_w())
    assert src.harvested_energy_j() == pytest.approx(src.true_power_w() * dt)
    src.update(2 * H)
    assert src.true_power_w() == 0.0  # night
    assert 0.0 <= src.daily_clearness <= 1.0


def test_field_backend_replays_normalised_moisture_and_counts_rain():
    tr = _tiny_trace()
    cfg = default_config()
    dt = cfg.time.timestep_s
    win = TraceWindow(tr, 1 * DAY, start_day=0)
    fld = TraceFieldEnvironment(tr, cfg.agriculture, dt, win, field_capacity=0.40)
    fld.reset(np.random.default_rng(0), 0.0)
    assert fld.moisture == pytest.approx(0.30 / 0.40)
    t = 0.0
    rain_before = fld.rain_events
    while t < 31 * H:
        st = fld.step(t)
        t += dt
    assert fld.rain_events == rain_before + 1  # one wet interval in hour 29→30
    assert st.soil_moisture == pytest.approx(tr.value_at("soil_moisture_0_to_7cm", t) / 0.40)
    assert 0.0 < st.relative_humidity <= 1.0  # percent converted to fraction
    assert st.zone in (0, 1, 2)


def test_window_bounds():
    tr = _tiny_trace(4)
    win = TraceWindow(tr, 2 * DAY)
    assert win.max_start_day == 1
    with pytest.raises(ValueError):
        TraceWindow(tr, 2 * DAY, start_day=2)
    with pytest.raises(ValueError):
        TraceWindow(tr, 5 * DAY)
    days = {win.choose(np.random.default_rng(i)) for i in range(20)}
    assert days <= {0, 1}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def demo_trace() -> Trace:
    if not DEMO.exists():
        pytest.skip("demo trace not present")
    return Trace.from_csv(DEMO)


def test_trace_env_passes_gymnasium_checker(demo_trace):
    env = TraceDrivenEnv(demo_trace, default_config(), start_day=150)
    check_env(env, skip_render_check=True)


def test_trace_env_same_spaces_as_synthetic(demo_trace):
    cfg = default_config()
    a = EdgeEngineAwareEnv(cfg)
    b = TraceDrivenEnv(demo_trace, cfg)
    assert a.observation_space == b.observation_space
    assert a.action_space == b.action_space
    obs, info = b.reset(seed=3)
    assert obs.shape == a.observation_space.shape
    assert "trace_start_day" in info


def test_trace_env_is_reproducible_and_windows_vary(demo_trace):
    cfg = default_config()
    env = TraceDrivenEnv(demo_trace, cfg)
    pol = RuleBasedPolicy(RuleBasedParams(), NodeProfile.from_config(cfg))
    r1 = run_episode(env, pol, seed=11)
    r2 = run_episode(env, pol, seed=11)
    assert r1.total_reward == pytest.approx(r2.total_reward)
    starts = {env.reset(seed=s)[1]["trace_start_day"] for s in range(12)}
    assert len(starts) > 3
    _, info = env.reset(seed=0, options={"start_day": 200})
    assert info["trace_start_day"] == 200


def test_trace_env_replays_ground_truth(demo_trace):
    cfg = default_config()
    cfg.time.episode_days = 2
    env = TraceDrivenEnv(demo_trace, cfg, start_day=180)
    env.reset(seed=0)
    gt0 = env.ground_truth()
    assert gt0["soil_moisture"] == pytest.approx(min(1.0, demo_trace.value_at("soil_moisture_0_to_7cm", 180 * DAY) / 0.40))
    # harvest at noon must follow the recorded irradiance
    for _ in range(48):
        env.step(np.array([0, 0]))
    gt = env.ground_truth()
    t = 180 * DAY + gt["time_s"]
    ghi = demo_trace.interval_mean("shortwave_radiation", t, t + cfg.time.timestep_s)
    assert gt["harvest_power_true_w"] == pytest.approx(cfg.harvesting.max_power_w * cfg.harvesting.efficiency * ghi / 1000.0)
    assert gt["soil_moisture"] == pytest.approx(min(1.0, demo_trace.value_at("soil_moisture_0_to_7cm", t) / 0.40))


def test_partial_replacement(demo_trace):
    cfg = default_config()
    env = TraceDrivenEnv(demo_trace, cfg, start_day=10, field=False)
    assert isinstance(env.source, TraceSolarEnergySource)
    assert not isinstance(env.field, TraceFieldEnvironment)
    env.reset(seed=0)
    env.step(np.array([2, 1]))
    with pytest.raises(ValueError):
        TraceDrivenEnv(demo_trace, cfg, harvesting=False, field=False)
