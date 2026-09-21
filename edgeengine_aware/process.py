"""Domain-independent pieces of the hidden world: the state every monitored
process reports and the weekly activity schedule shared by the indoor and
industrial domains.

A *monitored process* is anything the node's sensor samples and the
application wants to track: soil moisture (``agriculture.FieldEnvironment``),
the CO2 concentration of a room (``indoor.IndoorAirProcess``), the temperature
of a motor bearing (``industrial.BearingProcess``). Each of them implements

    reset(rng, start_time_s)      -> None
    step(time_s)                  -> ProcessState   (advance over [t, t + dt])
    state()                       -> ProcessState
    value                         -> float           (normalised, in [0, 1]; what the sensor samples)

and reports its physical extras in ``ProcessState.aux``. Everything is
simulator-only: the node sees the process through the noisy sensor only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import ScheduleConfig

DAY_S = 86400.0
HOUR_S = 3600.0


@dataclass
class ProcessState:
    """Snapshot of a monitored process (privileged information)."""

    value: float
    """Normalised value of the monitored quantity in [0, 1]."""

    last_step_change: float
    """Change of the value during the last step."""

    event_occurred: bool
    """A sudden change or a zone crossing happened in the last step."""

    zone: int
    """0 = normal, 1 = warning, 2 = critical (direction given by QuantityConfig)."""

    aux: dict[str, Any] = field(default_factory=dict)
    """Domain-specific extras (temperatures, occupancy, health, counters, ...)."""

    # backwards-compatible alias used by the agriculture code paths
    @property
    def soil_moisture(self) -> float:
        return self.value


class ActivitySchedule:
    """Hidden weekly activity level in [0, 1] (people in a room, a machine on
    shift). Drives both the monitored process and the harvesting source of a
    domain, so that energy and information relevance are coupled the way they
    are in reality.

    ``level(t)`` is deterministic given the per-day draws made at ``reset``
    (day factors, days off, extra days) plus a within-day AR(1) perturbation
    advanced by ``step()`` once per environment step.
    """

    def __init__(self, cfg: ScheduleConfig, start_weekday: int = 0, horizon_days: int = 10):
        self.cfg = cfg
        self.start_weekday = int(start_weekday)
        self.horizon_days = int(horizon_days)
        self._rng = np.random.default_rng()
        self._day_factor = np.ones(self.horizon_days + 2)
        self._day_active = np.ones(self.horizon_days + 2, dtype=bool)
        self._noise = 0.0
        self.reset(self._rng, 0.0)

    def reset(self, rng: np.random.Generator, start_time_s: float) -> None:
        c = self.cfg
        self._rng = rng
        n = self.horizon_days + 2
        first_day = int(start_time_s // DAY_S)
        self._first_day = first_day
        self._day_factor = np.clip(1.0 + rng.normal(0.0, c.day_factor_std, size=n), 0.3, 1.5)
        active = np.array([self.weekday(first_day + k) in c.active_days for k in range(n)])
        flips = rng.random(n)
        self._day_active = np.where(active, flips >= c.p_day_off, flips < c.p_extra_day)
        self._noise = 0.0

    # -- calendar -----------------------------------------------------------------
    def weekday(self, day_index: int) -> int:
        return (self.start_weekday + day_index) % 7

    def is_active_day(self, time_s: float) -> bool:
        k = int(time_s // DAY_S) - self._first_day
        if not 0 <= k < len(self._day_active):
            return self.weekday(int(time_s // DAY_S)) in self.cfg.active_days
        return bool(self._day_active[k])

    def _day_factor_at(self, time_s: float) -> float:
        k = int(time_s // DAY_S) - self._first_day
        if not 0 <= k < len(self._day_factor):
            return 1.0
        return float(self._day_factor[k])

    # -- level --------------------------------------------------------------------
    def nominal_level(self, time_s: float) -> float:
        """Deterministic window shape (ramps and dip) for the day type of ``time_s``."""
        c = self.cfg
        if not self.is_active_day(time_s):
            return 0.0
        h = (time_s % DAY_S) / HOUR_S
        if h <= c.start_hour - c.ramp_h or h >= c.end_hour + c.ramp_h:
            return 0.0
        ramp_in = min(1.0, max(0.0, (h - (c.start_hour - c.ramp_h)) / max(c.ramp_h, 1e-6)))
        ramp_out = min(1.0, max(0.0, ((c.end_hour + c.ramp_h) - h) / max(c.ramp_h, 1e-6)))
        shape = min(ramp_in, ramp_out)
        if c.dip_hours is not None and c.dip_hours[0] <= h < c.dip_hours[1]:
            shape *= c.dip_level / max(c.base_level, 1e-6)
        return c.base_level * shape * self._day_factor_at(time_s)

    def step(self) -> None:
        """Advance the within-day perturbation by one environment step."""
        c = self.cfg
        rho = c.noise_autocorr
        self._noise = rho * self._noise + math.sqrt(max(0.0, 1 - rho**2)) * self._rng.normal(0.0, c.noise_std)

    def level(self, time_s: float) -> float:
        nominal = self.nominal_level(time_s)
        if nominal <= 0.0:
            return 0.0
        return float(np.clip(nominal * (1.0 + self._noise), 0.0, 1.0))


__all__ = ["ProcessState", "ActivitySchedule"]
