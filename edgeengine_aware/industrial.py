"""Domain *industrial condition monitoring*: the bearing temperature of a
motor, powered by a thermoelectric generator (TEG) on the warm casing.

The hidden world is driven by the **shift schedule** (machine load). Running
heats the bearing - which is what the TEG harvests - and wears it; a random
*fault onset* accelerates the wear, the temperature drifts towards the alarm
thresholds, and maintenance eventually restores the bearing. Energy and
information are coupled through the same physical variable: the node harvests
most exactly when the machine runs hot, and a fault makes it run hotter still,
but with the thermal inertia of the casing (tens of minutes) and nothing at
all over a cold weekend.

Numbers: a 45 K steady-state rise at full load, a 45-minute thermal time
constant, a 30x30 mm TEG module giving ~1.2 mW electrical at 30 K before the
converter (~1.6 mW after it at full-load temperature), a 120 J storage, a
LoRa-like radio; sensing is a cheap temperature reading (50 mJ) or a vibration burst
with on-board FFT (0.5 J).
"""

from __future__ import annotations

import math

import numpy as np

from .config import IndustrialConfig, QuantityConfig, ThermoelectricConfig
from .process import DAY_S, HOUR_S, ActivitySchedule, ProcessState


class BearingProcess:
    """Thermal model of a motor bearing with slow wear, fault onsets and maintenance."""

    def __init__(self, cfg: IndustrialConfig, timestep_s: float, schedule: ActivitySchedule):
        self.cfg = cfg
        self.q: QuantityConfig = cfg.quantity()
        self.dt = timestep_s
        self.schedule = schedule
        self._rng = np.random.default_rng()
        self.temperature_c = cfg.ambient_c
        self.ambient_c = cfg.ambient_c
        self.load = 0.0
        self.health = 1.0
        self.fault_active = False
        self.maintenance_due_s: float | None = None
        self.last_change = 0.0
        self.event_occurred = False
        self.fault_onsets = 0
        self.maintenance_events = 0

    # -- lifecycle ----------------------------------------------------------------
    def reset(self, rng: np.random.Generator, start_time_s: float) -> None:
        c = self.cfg
        self._rng = rng
        self.health = float(rng.uniform(*c.initial_health_range))
        self.fault_active = False
        self.maintenance_due_s = None
        self.load = self.schedule.level(start_time_s)
        self.ambient_c = self._ambient(start_time_s)
        # start at the steady state of the current load (a node installed on a running or idle machine)
        self.temperature_c = self.ambient_c + self._rise_c(self.load)
        self.last_change = 0.0
        self.event_occurred = False
        self.fault_onsets = 0
        self.maintenance_events = 0

    # -- helpers ------------------------------------------------------------------
    def _ambient(self, time_s: float) -> float:
        c = self.cfg
        h = (time_s % DAY_S) / HOUR_S
        return c.ambient_c + c.ambient_amplitude_c * math.cos(2 * math.pi * (h - 15.0) / 24.0)

    def _rise_c(self, load: float) -> float:
        c = self.cfg
        return c.temp_rise_full_load_c * load * (1.0 + c.fault_heat_gain * (1.0 - self.health))

    @property
    def value(self) -> float:
        return float(np.clip(self.q.to_normalised(self.temperature_c), 0.0, 1.0))

    def zone(self, value: float | None = None) -> int:
        return self.q.zone(self.value if value is None else value)

    # -- dynamics -----------------------------------------------------------------
    def step(self, time_s: float) -> ProcessState:
        c, rng, dt = self.cfg, self._rng, self.dt
        prev_value = self.value
        prev_zone = self.q.zone(prev_value)
        prev_load = self.load
        load = self.schedule.level(time_s)
        self.load = load
        t_next = time_s + dt

        # wear and faults happen only while running
        if load > 0.0:
            hours = dt / HOUR_S * load
            wear = (c.fault_wear_per_hour if self.fault_active else c.wear_per_hour) * hours
            self.health = max(0.0, self.health - wear)
            if not self.fault_active and rng.random() < c.fault_onsets_per_day * dt / (16.0 * HOUR_S):
                self.fault_active = True
                self.fault_onsets += 1

        # maintenance: scheduled some time after the temperature exceeds the critical threshold
        if self.maintenance_due_s is None and self.q.beyond_critical(prev_value):
            self.maintenance_due_s = time_s + float(rng.exponential(c.maintenance_delay_mean_s))
        maintenance = False
        if self.maintenance_due_s is not None and t_next >= self.maintenance_due_s:
            self.health = 1.0
            self.fault_active = False
            self.maintenance_due_s = None
            self.maintenance_events += 1
            maintenance = True

        # first-order thermal response towards the steady state of the current load
        self.ambient_c = self._ambient(t_next)
        target = self.ambient_c + self._rise_c(load)
        alpha = 1.0 - math.exp(-dt / c.thermal_time_constant_s)
        self.temperature_c += alpha * (target - self.temperature_c) + rng.normal(0.0, c.process_noise_c)

        self.last_change = self.value - prev_value
        crossed = self.q.zone(self.value) != prev_zone
        started_or_stopped = (prev_load == 0.0) != (load == 0.0)
        self.event_occurred = abs(self.last_change) >= self.q.event_change_threshold or crossed or started_or_stopped or maintenance
        return self.state()

    def state(self) -> ProcessState:
        return ProcessState(
            value=self.value,
            last_step_change=self.last_change,
            event_occurred=self.event_occurred,
            zone=self.zone(),
            aux={
                "temperature_c": self.temperature_c,
                "ambient_c": self.ambient_c,
                "load": self.load,
                "health": self.health,
                "fault_active": self.fault_active,
                "fault_onsets": self.fault_onsets,
                "maintenance_events": self.maintenance_events,
            },
        )


class ThermoelectricSource:
    """TEG harvesting from the casing-to-ambient temperature difference of a
    :class:`BearingProcess`. Same interface as ``energy.SolarEnergySource``."""

    def __init__(self, cfg: ThermoelectricConfig, timestep_s: float, process: BearingProcess):
        self.cfg = cfg
        self.dt = timestep_s
        self.process = process
        self._rng = np.random.default_rng()
        self._power_w = 0.0
        self._measured_w = 0.0
        self._dt_c = 0.0

    def reset(self, rng: np.random.Generator, start_time_s: float) -> None:
        self._rng = rng
        self.update(start_time_s)

    def update(self, time_s: float) -> None:
        c = self.cfg
        dt_c = max(0.0, self.process.temperature_c - self.process.ambient_c)
        self._dt_c = dt_c
        if dt_c < c.min_dt_c:
            self._power_w = 0.0
        else:
            self._power_w = c.power_w_at_ref_dt * (dt_c / c.reference_dt_c) ** 2 * c.efficiency
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
    def temperature_difference_c(self) -> float:
        return self._dt_c

    @property
    def daily_clearness(self) -> float:
        """Stand-in for the solar clearness: the machine load of the current step."""
        return float(self.process.load)


__all__ = ["BearingProcess", "ThermoelectricSource"]
