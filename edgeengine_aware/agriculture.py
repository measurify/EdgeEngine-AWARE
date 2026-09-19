"""Hidden ground-truth model of the agricultural field.

Everything in this module is *simulator-only*. The node never reads these
variables directly; it can only sample them through a ``Sensor`` (which adds
noise) and the reward/renderer use them for evaluation.

Soil moisture ``theta`` is dimensionless (1 = field capacity). Its dynamics
per step of length ``dt`` are

    theta(t+1) = clip(theta(t) - ET(t) * dt + rain(t) + irrigation(t) + noise, 0, max)

    ET(t) = et_rate_per_day / 86400 * (1 + et_temp_coeff * (T(t) - T_mean))
            * (1 - a + a * pi/2 * daylight_factor(t))     with a = et_diurnal_amplitude

(the factor pi/2 makes the 24-hour mean of the modulation equal to 1, so
``et_rate_per_day`` is the actual mean daily ET at the reference temperature)

Rain events follow a Poisson process; irrigation is applied by an external
actor (the farmer's system) some random time after the *true* moisture drops
below ``irrigation_trigger``. Temperature follows a daily cosine with AR(1)
noise, and humidity is linearly anti-correlated with the temperature anomaly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import AgricultureConfig

DAY_S = 86400.0


@dataclass
class FieldState:
    """Snapshot of the ground truth (privileged information)."""

    soil_moisture: float
    air_temperature_c: float
    relative_humidity: float
    last_step_change: float
    """Change of soil moisture during the last step (positive after rain)."""
    event_occurred: bool
    """True if a rain/irrigation event or threshold crossing happened in the last step."""
    zone: int
    """0 = normal, 1 = warning (below warning_threshold), 2 = critical."""
    rain_events: int = 0
    irrigation_events: int = 0


class FieldEnvironment:
    """Stochastic soil / atmosphere simulator."""

    def __init__(self, cfg: AgricultureConfig, timestep_s: float):
        self.cfg = cfg
        self.dt = timestep_s
        self._rng = np.random.default_rng()
        self.reset(self._rng, start_time_s=0.0)

    # -- lifecycle ----------------------------------------------------------
    def reset(self, rng: np.random.Generator, start_time_s: float) -> None:
        c = self.cfg
        self._rng = rng
        if c.initial_moisture_range is not None:
            self.moisture = float(rng.uniform(*c.initial_moisture_range))
        else:
            self.moisture = c.initial_moisture
        self._temp_dev = 0.0
        self.temperature = self._temperature_base(start_time_s)
        self.humidity = c.humidity_mean
        self.last_change = 0.0
        self.event_occurred = False
        self.irrigation_pending_s: float | None = None
        self.rain_events = 0
        self.irrigation_events = 0
        self.total_rain = 0.0
        self.total_irrigation = 0.0

    # -- helpers ------------------------------------------------------------
    def _temperature_base(self, time_s: float) -> float:
        c = self.cfg
        h = (time_s % DAY_S) / 3600.0
        return c.temp_mean_c + c.temp_amplitude_c * math.cos(2 * math.pi * (h - c.temp_peak_hour) / 24.0)

    def _daylight_factor(self, time_s: float) -> float:
        """Half-sine daylight shape in [0, 1] (0 at night)."""
        c = self.cfg
        h = (time_s % DAY_S) / 3600.0
        if not c.sunrise_hour < h < c.sunset_hour:
            return 0.0
        return math.sin(math.pi * (h - c.sunrise_hour) / (c.sunset_hour - c.sunrise_hour))

    def zone(self, moisture: float | None = None) -> int:
        m = self.moisture if moisture is None else moisture
        if m < self.cfg.critical_threshold:
            return 2
        if m < self.cfg.warning_threshold:
            return 1
        return 0

    # -- dynamics -----------------------------------------------------------
    def step(self, time_s: float) -> FieldState:
        """Advance the field by one timestep starting at ``time_s``."""
        c, rng, dt = self.cfg, self._rng, self.dt
        prev = self.moisture
        prev_zone = self.zone(prev)

        # atmosphere
        self._temp_dev = c.temp_autocorr * self._temp_dev + math.sqrt(max(0.0, 1 - c.temp_autocorr**2)) * rng.normal(
            0.0, c.temp_noise_std
        )
        self.temperature = self._temperature_base(time_s) + self._temp_dev
        self.humidity = float(
            np.clip(
                c.humidity_mean + c.humidity_temp_coeff * (self.temperature - c.temp_mean_c) + rng.normal(0.0, c.humidity_noise_std),
                0.05,
                1.0,
            )
        )

        # evapotranspiration
        et_per_s = c.et_rate_per_day / DAY_S * max(0.0, 1.0 + c.et_temp_coeff * (self.temperature - c.temp_mean_c))
        # 24 h mean of daylight_factor = (daylight_h / 24) * (2 / pi); the prefactor
        # below normalises the modulation to a mean of exactly 1 over the day.
        daylight_h = c.sunset_hour - c.sunrise_hour
        modulation = (math.pi / 2.0) * (24.0 / daylight_h) * self._daylight_factor(time_s)
        et_per_s *= (1.0 - c.et_diurnal_amplitude) + c.et_diurnal_amplitude * modulation
        delta = -et_per_s * dt

        # rain (Poisson process)
        n_rain = rng.poisson(c.rain_events_per_day * dt / DAY_S)
        if n_rain > 0:
            amount = float(sum(rng.uniform(*c.rain_amount_range) for _ in range(n_rain)))
            delta += amount
            self.rain_events += int(n_rain)
            self.total_rain += amount

        # external irrigation reacting to the true moisture with a random delay
        if c.irrigation_enabled:
            if self.irrigation_pending_s is None and prev < c.irrigation_trigger:
                self.irrigation_pending_s = float(rng.exponential(c.irrigation_delay_mean_s))
            if self.irrigation_pending_s is not None:
                self.irrigation_pending_s -= dt
                if self.irrigation_pending_s <= 0.0:
                    delta += c.irrigation_amount
                    self.irrigation_events += 1
                    self.total_irrigation += c.irrigation_amount
                    self.irrigation_pending_s = None

        delta += rng.normal(0.0, c.process_noise_std)
        self.moisture = float(np.clip(prev + delta, 0.0, c.max_moisture))
        self.last_change = self.moisture - prev

        crossed = self.zone(self.moisture) != prev_zone
        self.event_occurred = abs(self.last_change) >= c.event_change_threshold or crossed
        return self.state()

    def state(self) -> FieldState:
        return FieldState(
            soil_moisture=self.moisture,
            air_temperature_c=self.temperature,
            relative_humidity=self.humidity,
            last_step_change=self.last_change,
            event_occurred=self.event_occurred,
            zone=self.zone(),
            rain_events=self.rain_events,
            irrigation_events=self.irrigation_events,
        )
