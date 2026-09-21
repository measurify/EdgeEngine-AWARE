"""Domain *indoor air quality*: CO2 in a classroom or office, powered by an
indoor photovoltaic cell.

The hidden world is driven by one weekly **activity schedule** (people in the
room). It raises the CO2 concentration through breathing and, at the same
time, switches the lights on - which is where the node's energy comes from.
Energy and information relevance therefore coincide: nights and weekends
bring neither. This is the mirror image of the agricultural node (energy from
the sun, information relevance from a slow soil process) and a good test of
whether a policy has learned *energy-information trade-offs* rather than the
solar day.

Numbers are for a small sensor node with a low-power NDIR CO2 sensor: a
cheap reading is a short single-shot measurement (~20 mJ), an accurate one a
longer averaged acquisition (~150 mJ); the radio is BLE-like (sub-millijoule
uplinks); the always-on consumption is 20 uW; the storage is a 60 J
supercapacitor / small cell; the cell delivers ~140 uW under 500 lux.
"""

from __future__ import annotations

import math

import numpy as np

from .config import IndoorAirConfig, IndoorLightConfig, QuantityConfig
from .process import DAY_S, HOUR_S, ActivitySchedule, ProcessState


class IndoorAirProcess:
    """CO2 mass balance of a room driven by the occupancy schedule."""

    def __init__(self, cfg: IndoorAirConfig, timestep_s: float, schedule: ActivitySchedule):
        self.cfg = cfg
        self.q: QuantityConfig = cfg.quantity()
        self.dt = timestep_s
        self.schedule = schedule
        self._rng = np.random.default_rng()
        self.ppm = cfg.outdoor_ppm
        self.occupancy = 0.0
        self.ach = cfg.ach_base
        self.window_until_s: float | None = None
        self.last_change = 0.0
        self.event_occurred = False
        self.window_events = 0

    # -- lifecycle ----------------------------------------------------------------
    def reset(self, rng: np.random.Generator, start_time_s: float) -> None:
        c = self.cfg
        self._rng = rng
        self.ppm = float(rng.uniform(*c.initial_ppm_range))
        self.occupancy = self.schedule.level(start_time_s)
        self.ach = c.ach_hvac if self.occupancy > 0 else c.ach_base
        self.window_until_s = None
        self.last_change = 0.0
        self.event_occurred = False
        self.window_events = 0

    # -- accessors ----------------------------------------------------------------
    @property
    def value(self) -> float:
        return float(np.clip(self.q.to_normalised(self.ppm), 0.0, 1.0))

    def zone(self, value: float | None = None) -> int:
        return self.q.zone(self.value if value is None else value)

    # -- dynamics -----------------------------------------------------------------
    def step(self, time_s: float) -> ProcessState:
        """Advance over [time_s, time_s + dt] (occupancy read at the start of the step)."""
        c, rng, dt = self.cfg, self._rng, self.dt
        prev_value = self.value
        prev_zone = self.q.zone(prev_value)
        occ = self.schedule.level(time_s)
        self.occupancy = occ

        # ventilation: HVAC follows the activity window; window openings are random
        ach = c.ach_hvac if occ > 0.0 else c.ach_base
        if self.window_until_s is not None and time_s >= self.window_until_s:
            self.window_until_s = None
        if occ > 0.0 and self.window_until_s is None and rng.random() < c.window_events_per_day * dt / (10.0 * HOUR_S):
            self.window_until_s = time_s + c.window_duration_s
            self.window_events += 1
        if self.window_until_s is not None:
            ach = max(ach, c.ach_window)
        self.ach = ach

        # exact solution of dC/dt = S - lambda (C - C_out) over the step
        lam = ach / HOUR_S  # 1/s
        source_ppm_s = c.co2_per_person_l_h * c.max_occupants * occ / (c.room_volume_m3 * 1000.0) * 1e6 / HOUR_S
        steady = c.outdoor_ppm + source_ppm_s / lam
        self.ppm = steady + (self.ppm - steady) * math.exp(-lam * dt)
        self.ppm += rng.normal(0.0, c.process_noise_ppm)
        self.ppm = max(c.outdoor_ppm * 0.95, self.ppm)

        self.last_change = self.value - prev_value
        crossed = self.q.zone(self.value) != prev_zone
        self.event_occurred = abs(self.last_change) >= self.q.event_change_threshold or crossed
        return self.state()

    def state(self) -> ProcessState:
        return ProcessState(
            value=self.value,
            last_step_change=self.last_change,
            event_occurred=self.event_occurred,
            zone=self.zone(),
            aux={
                "co2_ppm": self.ppm,
                "occupancy": self.occupancy,
                "air_changes_per_hour": self.ach,
                "window_open": self.window_until_s is not None,
                "window_events": self.window_events,
            },
        )


class IndoorLightSource:
    """Indoor PV harvesting: artificial light when the room is active plus
    daylight through a window (0 for a windowless room).

    Same interface as ``energy.SolarEnergySource``: ``reset``, ``update(t)``,
    ``measured_power_w`` (noisy, node side), ``true_power_w``,
    ``harvested_energy_j`` and ``daily_clearness`` (here: today's daylight factor).
    """

    def __init__(self, cfg: IndoorLightConfig, timestep_s: float, schedule: ActivitySchedule):
        self.cfg = cfg
        self.dt = timestep_s
        self.schedule = schedule
        self._rng = np.random.default_rng()
        self._weather = 1.0
        self._power_w = 0.0
        self._measured_w = 0.0
        self._lux = 0.0
        self._day = -1
        self._day_factor = 1.0

    def reset(self, rng: np.random.Generator, start_time_s: float) -> None:
        self._rng = rng
        self._weather = 0.0
        self._day = -1
        self._day_factor = float(np.clip(rng.uniform(0.4, 1.0), 0.0, 1.0))
        self.update(start_time_s)

    def _daylight_shape(self, time_s: float) -> float:
        c = self.cfg
        h = (time_s % DAY_S) / HOUR_S
        if h <= c.sunrise_hour or h >= c.sunset_hour:
            return 0.0
        return math.sin(math.pi * (h - c.sunrise_hour) / (c.sunset_hour - c.sunrise_hour))

    def update(self, time_s: float) -> None:
        c = self.cfg
        day = int(time_s // DAY_S)
        if day != self._day:  # a new day: draw its overall brightness
            self._day_factor = float(np.clip(self._rng.uniform(0.4, 1.0), 0.0, 1.0))
            self._day = day
        rho = c.daylight_autocorr
        self._weather = rho * self._weather + math.sqrt(max(0.0, 1 - rho**2)) * self._rng.normal(0.0, c.daylight_noise_std)
        daylight = c.daylight_lux * self._daylight_shape(time_s) * float(np.clip(self._day_factor + self._weather, 0.0, 1.0))
        lights = c.artificial_lux if self.schedule.level(time_s) > c.lights_threshold else 0.0
        self._lux = lights + daylight
        self._power_w = c.cell_power_w_at_ref * self._lux / c.reference_lux * c.efficiency
        noise = 1.0 + self._rng.normal(0.0, c.measurement_noise_std)
        self._measured_w = max(0.0, self._power_w * noise)

    # -- EnergySource protocol ------------------------------------------------------
    def measured_power_w(self) -> float:
        return self._measured_w

    # -- simulator-only ---------------------------------------------------------------
    def true_power_w(self) -> float:
        return self._power_w

    def harvested_energy_j(self) -> float:
        return self._power_w * self.dt

    @property
    def illuminance_lux(self) -> float:
        return self._lux

    @property
    def daily_clearness(self) -> float:
        return self._day_factor


__all__ = ["IndoorAirProcess", "IndoorLightSource"]
