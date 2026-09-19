"""Named evaluation scenarios.

A scenario is a function that returns a fully configured
:class:`EdgeEngineAwareConfig`. The set below is the *benchmark protocol* of the
project: policies are compared on every scenario, over a fixed set of seeds,
with the same metrics. ``default`` is the training distribution; the others
stress one aspect of the problem so that the value of an adaptive policy
becomes visible where a fixed duty cycle cannot cope.

Scenario                what is stressed                            what a good policy does
----------------------  ------------------------------------------  --------------------------------------------
default                 nothing in particular (mixed weather)       report ~hourly, more near thresholds
cloudy_week             harvesting ~40 % of default, dark days      slow down early, keep a reserve, use cheap checks
tiny_battery            storage halved, small buffer                smooth consumption, avoid bursts at night
lossy_link              +6 dB path loss, deep slow fading           choose the radio mode from the link estimate
drought                 fast drying, no rain, slow irrigation       track the approach to the thresholds closely
demanding_application   frequent elevated/urgent campaigns          follow the priority, save energy in between
"""

from __future__ import annotations

from typing import Callable

from .config import HOUR_S, EdgeEngineAwareConfig, default_config


def default() -> EdgeEngineAwareConfig:
    return default_config()


def cloudy_week() -> EdgeEngineAwareConfig:
    cfg = default_config()
    cfg.harvesting.clearness_mean = 0.35
    cfg.harvesting.clearness_std = 0.15
    cfg.harvesting.cloud_noise_std = 0.25
    cfg.storage.initial_soc = 0.4
    return cfg


def tiny_battery() -> EdgeEngineAwareConfig:
    cfg = default_config()
    cfg.storage.capacity_j = 150.0
    cfg.storage.initial_soc = 0.5
    return cfg


def lossy_link() -> EdgeEngineAwareConfig:
    """Node far from the gateway: +6 dB path loss and slower fading. The
    standard mode drops to a mean margin of -2 dB (p ~ 0.25), the robust mode
    keeps +6 dB (p ~ 0.85) at twice the energy; the fast mode only works in
    the favourable fading phases."""
    cfg = default_config()
    cfg.communication.path_loss_mean_db += 6.0
    cfg.communication.slow_fading_autocorr = 0.98
    return cfg


def drought() -> EdgeEngineAwareConfig:
    cfg = default_config()
    cfg.agriculture.et_rate_per_day = 0.12
    cfg.agriculture.rain_events_per_day = 0.02
    cfg.agriculture.irrigation_delay_mean_s = 18.0 * HOUR_S
    cfg.agriculture.initial_moisture_range = (0.40, 0.55)
    return cfg


def demanding_application() -> EdgeEngineAwareConfig:
    cfg = default_config()
    cfg.application.request_rate_per_day = 1.5
    cfg.application.request_duration_range_s = (2.0 * HOUR_S, 8.0 * HOUR_S)
    cfg.application.request_urgent_fraction = 0.5
    cfg.application.aoi_elevated_s = 4.0 * HOUR_S
    return cfg


SCENARIOS: dict[str, Callable[[], EdgeEngineAwareConfig]] = {
    "default": default,
    "cloudy_week": cloudy_week,
    "tiny_battery": tiny_battery,
    "lossy_link": lossy_link,
    "drought": drought,
    "demanding_application": demanding_application,
}


def get_scenario(name: str, *, randomize: bool = False) -> EdgeEngineAwareConfig:
    """Return a fresh configuration for ``name`` (see :data:`SCENARIOS`)."""
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; available: {sorted(SCENARIOS)}")
    cfg = SCENARIOS[name]()
    cfg.randomization.enabled = randomize
    return cfg
