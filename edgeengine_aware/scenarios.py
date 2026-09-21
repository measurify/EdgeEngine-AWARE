"""Named evaluation scenarios of the agriculture domain (and the lookup of
every domain's scenarios by qualified name).

A scenario is a function that returns a fully configured
:class:`EdgeEngineAwareConfig`. The set below is the *benchmark protocol* of the
project: policies are compared on every scenario, over a fixed set of seeds,
with the same metrics. ``default`` is the training distribution; the others
stress one aspect of the problem so that the value of an adaptive policy
becomes visible where a fixed duty cycle cannot cope. The scenarios of the
other domains live in ``domains.py`` and are addressed as ``"indoor_air:no_window"``
or ``"industrial:degrading"`` (``scenario_names("all")`` lists everything).

Scenario                what is stressed                            what a good policy does
----------------------  ------------------------------------------  --------------------------------------------
default                 nothing in particular (mixed weather)       report ~hourly, more near thresholds
cloudy_week             harvesting ~55 % of default (30 J/day),     slow down early, keep a reserve, use cheap checks
                        battery starts at 40 %
tiny_battery            storage halved (150 J), small buffer        smooth consumption, avoid bursts at night
lossy_link              +6 dB path loss, slower fading              choose the radio mode from the link estimate
drought                 fast drying, almost no rain, slow           track the approach to the thresholds closely
                        irrigation
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
    """Node far from the gateway: +6 dB path loss and slower fading (AR(1)
    coefficient 0.98 instead of 0.97, i.e. bad phases last longer). Mean
    margins become -8 / -2 / +6 dB for fast / standard / robust; averaged over
    the fading, the standard mode delivers ~40 % and the robust mode ~85 % of
    the uplinks at twice the energy; the fast mode only works in favourable
    fading phases."""
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


def split_scenario_name(name: str) -> tuple[str, str]:
    """``"indoor_air:no_window"`` -> ``("indoor_air", "no_window")``; a bare name is agricultural."""
    if ":" in name:
        domain, scen = name.split(":", 1)
        return domain, scen
    return "agriculture", name


def scenario_names(domain: str = "agriculture") -> list[str]:
    """Qualified names (``domain:scenario``) of the benchmark scenarios of ``domain``,
    or of every domain with ``domain="all"``. Agricultural names are returned bare
    for backwards compatibility."""
    from .domains import DOMAIN_NAMES, domain_scenarios  # local import (domains.py imports SCENARIOS)

    if domain == "all":
        return [n for d in DOMAIN_NAMES for n in scenario_names(d)]
    if domain == "agriculture":
        return list(SCENARIOS)
    return [f"{domain}:{s}" for s in domain_scenarios(domain)]


def get_scenario(name: str, *, randomize: bool = False) -> EdgeEngineAwareConfig:
    """Return a fresh configuration for ``name``.

    ``name`` is an agricultural scenario (see :data:`SCENARIOS`) or a qualified
    ``domain:scenario`` of another domain, e.g. ``"industrial:degrading"``
    (see ``domains.DOMAINS``).
    """
    from .domains import DOMAIN_NAMES, domain_scenarios  # local import (domains.py imports SCENARIOS)

    domain, scen = split_scenario_name(name)
    if domain not in DOMAIN_NAMES:
        raise KeyError(f"unknown domain {domain!r} in scenario {name!r}; available: {DOMAIN_NAMES}")
    table = domain_scenarios(domain)
    if scen not in table:
        raise KeyError(f"unknown scenario {name!r}; available in {domain}: {sorted(table)}")
    cfg = table[scen]()
    cfg.randomization.enabled = randomize
    return cfg
