"""Application domains of EdgeEngine AWARE.

A *domain* is a monitored process (what the sensor samples and the
application wants to know), an energy source (where the harvested energy comes
from), a hardware profile (how much each operation costs) and a set of
benchmark scenarios. Three are provided:

=============  =====================================  ==========================  =================
domain         monitored quantity (danger side)        energy source               radio
=============  =====================================  ==========================  =================
agriculture    soil moisture (low)                    solar cell, weather         LoRa-like, 3 SF
indoor_air     CO2 of a room (high)                    indoor PV, lights + window  BLE-like, 3 PHY
industrial     bearing temperature of a motor (high)   TEG on the warm casing      LoRa-like, 3 SF
=============  =====================================  ==========================  =================

Whatever the domain, the node sees the same 18-number observation and returns
the same (sensing level, radio mode) action: the monitored quantity is
normalised to [0, 1] and the thresholds' direction is a profile constant. A
policy or a controller therefore runs unchanged in every domain - which is
what makes cross-domain evaluation possible (``examples/domains.ipynb``).

``build_world(cfg, dt)`` instantiates the schedule, process and source for a
configuration; ``domain_config(name)`` returns a domain's default
configuration; ``DOMAIN_SCENARIOS`` holds the benchmark scenarios of every
domain (``scenarios.get_scenario("indoor_air:no_window")``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .agriculture import FieldEnvironment
from .config import HOUR_S, EdgeEngineAwareConfig, ble_radio, default_config
from .energy import SolarEnergySource
from .indoor import IndoorAirProcess, IndoorLightSource
from .industrial import BearingProcess, ThermoelectricSource
from .process import ActivitySchedule

# ---------------------------------------------------------------------------
# Default configurations
# ---------------------------------------------------------------------------


def agriculture_config() -> EdgeEngineAwareConfig:
    """The original smart-agriculture node (see ``config.default_config``)."""
    return default_config()


def indoor_air_config() -> EdgeEngineAwareConfig:
    """CO2 monitoring in a classroom / office with an indoor PV cell.

    Micro-power node: 60 J storage, 20 uW always-on, ~140 uW harvested under
    the ceiling lights, sub-millijoule BLE uplinks; the CO2 sensor is the
    energy hog (20 / 150 mJ per reading). Occupancy 08:00-18:00 on weekdays.
    """
    cfg = default_config()
    cfg.domain = "indoor_air"
    cfg.harvesting_source = "indoor_light"
    cfg.storage.capacity_j = 60.0
    cfg.storage.initial_soc = 0.5
    cfg.storage.reserve_soc = 0.05
    cfg.mcu.baseline_power_w = 20e-6
    cfg.sensing.energy_j = (0.0, 0.02, 0.15)
    cfg.sensing.noise_std = (0.0, 0.05, 0.015)  # 80 ppm / 24 ppm on a 1600 ppm span
    cfg.communication = ble_radio()
    cfg.observation.harvest_ref_power_w = 300e-6
    cfg.application.aoi_elevated_s = 12.0 * HOUR_S
    cfg.application.aoi_urgent_s = 36.0 * HOUR_S
    cfg.schedule.active_days = (0, 1, 2, 3, 4)
    cfg.schedule.start_hour, cfg.schedule.end_hour = 8.0, 18.0
    cfg.schedule.dip_hours, cfg.schedule.dip_level = (12.5, 13.5), 0.3
    cfg.schedule.base_level = 0.8
    return cfg


def industrial_config() -> EdgeEngineAwareConfig:
    """Bearing-temperature monitoring of a motor on two shifts with a TEG.

    Same always-on load and radio as the agricultural node, a 120 J storage;
    sensing is a cheap temperature reading (50 mJ) or a vibration burst with
    on-board FFT (0.5 J). The machine runs 06:00-22:00 on weekdays, sometimes on
    Saturday: energy is plentiful while it runs and absent over the weekend.
    """
    cfg = default_config()
    cfg.domain = "industrial"
    cfg.harvesting_source = "thermoelectric"
    cfg.storage.capacity_j = 120.0  # enough for a weekend of hourly reports, not more
    cfg.sensing.energy_j = (0.0, 0.05, 0.50)
    cfg.sensing.noise_std = (0.0, 0.03, 0.01)  # 3 degC / 1 degC on a 100 degC span
    cfg.schedule.active_days = (0, 1, 2, 3, 4)
    cfg.schedule.start_hour, cfg.schedule.end_hour = 6.0, 22.0
    cfg.schedule.dip_hours = None
    cfg.schedule.base_level = 0.9
    cfg.schedule.day_factor_std = 0.10
    cfg.schedule.p_extra_day = 0.15
    return cfg


# ---------------------------------------------------------------------------
# Scenarios per domain
# ---------------------------------------------------------------------------


def _indoor_no_window() -> EdgeEngineAwareConfig:
    cfg = indoor_air_config()
    cfg.indoor_light.daylight_lux = 0.0  # energy only while the lights are on
    return cfg


def _indoor_weak_ventilation() -> EdgeEngineAwareConfig:
    cfg = indoor_air_config()
    cfg.indoor_air.ach_hvac = 1.5  # CO2 climbs well past the critical threshold every day
    cfg.indoor_air.ach_base = 0.4
    return cfg


def _indoor_long_hours() -> EdgeEngineAwareConfig:
    cfg = indoor_air_config()
    cfg.schedule.active_days = (0, 1, 2, 3, 4, 5)  # Saturday too
    cfg.schedule.end_hour = 21.0
    cfg.schedule.base_level = 0.9
    return cfg


def _indoor_dim_lights() -> EdgeEngineAwareConfig:
    cfg = indoor_air_config()
    cfg.indoor_light.artificial_lux = 250.0  # LED office at half the illuminance: half the energy
    cfg.storage.initial_soc = 0.4
    return cfg


def _industrial_single_shift() -> EdgeEngineAwareConfig:
    cfg = industrial_config()
    cfg.schedule.start_hour, cfg.schedule.end_hour = 8.0, 17.0  # half the running hours, half the energy
    cfg.schedule.p_extra_day = 0.0
    return cfg


def _industrial_degrading() -> EdgeEngineAwareConfig:
    cfg = industrial_config()
    cfg.industrial.fault_onsets_per_day = 0.6  # faults every couple of days
    cfg.industrial.initial_health_range = (0.3, 0.7)
    cfg.industrial.maintenance_delay_mean_s = 24.0 * HOUR_S
    return cfg


def _industrial_continuous() -> EdgeEngineAwareConfig:
    cfg = industrial_config()
    cfg.schedule.active_days = (0, 1, 2, 3, 4, 5, 6)  # 24/7 plant: energy-rich, no weekend
    cfg.schedule.start_hour, cfg.schedule.end_hour = 0.0, 24.0
    cfg.schedule.ramp_h = 0.01
    cfg.schedule.p_day_off = 0.02
    return cfg


def _industrial_weak_link() -> EdgeEngineAwareConfig:
    cfg = industrial_config()
    cfg.communication.path_loss_mean_db += 6.0  # metal hall, gateway far away
    cfg.communication.slow_fading_autocorr = 0.98
    return cfg


INDOOR_AIR_SCENARIOS: dict[str, Callable[[], EdgeEngineAwareConfig]] = {
    "default": indoor_air_config,
    "no_window": _indoor_no_window,
    "weak_ventilation": _indoor_weak_ventilation,
    "long_hours": _indoor_long_hours,
    "dim_lights": _indoor_dim_lights,
}

INDUSTRIAL_SCENARIOS: dict[str, Callable[[], EdgeEngineAwareConfig]] = {
    "default": industrial_config,
    "single_shift": _industrial_single_shift,
    "degrading": _industrial_degrading,
    "continuous": _industrial_continuous,
    "weak_link": _industrial_weak_link,
}


@dataclass(frozen=True)
class DomainInfo:
    name: str
    title: str
    quantity: str
    source: str
    radio: str
    description: str
    config: Callable[[], EdgeEngineAwareConfig]
    scenarios: dict[str, Callable[[], EdgeEngineAwareConfig]]


def _agriculture_scenarios() -> dict[str, Callable[[], EdgeEngineAwareConfig]]:
    from .scenarios import SCENARIOS  # local import: scenarios.py imports this module

    return SCENARIOS


DOMAINS: dict[str, DomainInfo] = {
    "agriculture": DomainInfo(
        "agriculture", "Smart agriculture", "soil moisture (danger: low)", "solar cell, weather-driven", "LoRa-like, 3 spreading factors",
        "A soil-moisture node in a field: energy from the sun, information relevance from a slow soil process with rain and irrigation events.",
        agriculture_config, {},
    ),
    "indoor_air": DomainInfo(
        "indoor_air", "Indoor air quality", "CO2 concentration (danger: high)", "indoor PV cell, lights and window", "BLE-like, 3 PHY modes",
        "A CO2 node in a classroom: people raise the CO2 and switch the lights on - energy and relevance coincide; nights and weekends bring neither.",
        indoor_air_config, INDOOR_AIR_SCENARIOS,
    ),
    "industrial": DomainInfo(
        "industrial", "Industrial condition monitoring", "bearing temperature (danger: high)", "thermoelectric generator on the casing", "LoRa-like, 3 spreading factors",
        "A node on a motor bearing: the machine's heat is the energy source and the monitored variable; faults make it run hotter until maintenance.",
        industrial_config, INDUSTRIAL_SCENARIOS,
    ),
}
DOMAIN_NAMES = tuple(DOMAINS)


def domain_config(name: str) -> EdgeEngineAwareConfig:
    if name not in DOMAINS:
        raise KeyError(f"unknown domain {name!r}; available: {DOMAIN_NAMES}")
    return DOMAINS[name].config()


def domain_scenarios(name: str) -> dict[str, Callable[[], EdgeEngineAwareConfig]]:
    if name == "agriculture":
        return _agriculture_scenarios()
    return DOMAINS[name].scenarios


# ---------------------------------------------------------------------------
# World factory used by the environment
# ---------------------------------------------------------------------------


def build_world(cfg: EdgeEngineAwareConfig, timestep_s: float) -> tuple[ActivitySchedule | None, Any, Any]:
    """Instantiate ``(schedule, process, source)`` for ``cfg``.

    The schedule is shared by the process and the source when the domain has
    one; ``None`` for the agriculture domain, whose weather and soil have
    their own stochastic models.
    """
    schedule: ActivitySchedule | None = None
    if cfg.uses_schedule:
        schedule = ActivitySchedule(cfg.schedule, start_weekday=cfg.time.start_weekday, horizon_days=int(cfg.time.episode_days) + 2)

    if cfg.domain == "agriculture":
        process: Any = FieldEnvironment(cfg.agriculture, timestep_s)
    elif cfg.domain == "indoor_air":
        assert schedule is not None
        process = IndoorAirProcess(cfg.indoor_air, timestep_s, schedule)
    elif cfg.domain == "industrial":
        assert schedule is not None
        process = BearingProcess(cfg.industrial, timestep_s, schedule)
    else:
        raise ValueError(f"unknown domain {cfg.domain!r}")

    if cfg.harvesting_source == "solar":
        source: Any = SolarEnergySource(cfg.harvesting, timestep_s)
    elif cfg.harvesting_source == "indoor_light":
        assert schedule is not None
        source = IndoorLightSource(cfg.indoor_light, timestep_s, schedule)
    elif cfg.harvesting_source == "thermoelectric":
        if not isinstance(process, BearingProcess):
            raise ValueError("the thermoelectric source needs the industrial process")
        source = ThermoelectricSource(cfg.thermoelectric, timestep_s, process)
    else:
        raise ValueError(f"unknown harvesting source {cfg.harvesting_source!r}")
    return schedule, process, source


__all__ = [
    "DOMAINS",
    "DOMAIN_NAMES",
    "DomainInfo",
    "INDOOR_AIR_SCENARIOS",
    "INDUSTRIAL_SCENARIOS",
    "agriculture_config",
    "indoor_air_config",
    "industrial_config",
    "domain_config",
    "domain_scenarios",
    "build_world",
]
