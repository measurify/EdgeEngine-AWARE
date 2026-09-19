"""Simulated clock, energy storage and solar harvesting.

The storage update implemented by the environment is

    E(t+1) = clip(E(t) + harvested(t) - baseline(t) - sensing(t) - comm(t), 0, E_max)

with all terms in joules. This module provides the building blocks; the
ordering of the terms and the feasibility rule are handled in ``env.py``.
"""

from __future__ import annotations

import math

import numpy as np

from .config import EnergyStorageConfig, HarvestingConfig, TimeConfig

DAY_S = 86400.0


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------
class SimulatedClock:
    """Discrete clock advancing by a fixed timestep."""

    def __init__(self, cfg: TimeConfig):
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._t = self.cfg.start_hour * 3600.0

    def advance(self) -> None:
        self._t += self.cfg.timestep_s

    def now_s(self) -> float:
        return self._t

    def time_of_day_s(self) -> float:
        return self._t % DAY_S

    def hour_of_day(self) -> float:
        return self.time_of_day_s() / 3600.0

    def day_index(self) -> int:
        return int(self._t // DAY_S)


# ---------------------------------------------------------------------------
# Energy storage
# ---------------------------------------------------------------------------
class SimulatedEnergyStorage:
    """Ideal (lossless) finite energy buffer with bookkeeping.

    Battery ageing, temperature effects and charge/discharge efficiency are
    intentionally omitted; ``charge_efficiency`` is the single knob left to
    approximate a real cell (1.0 = ideal). Real hardware replaces this class by
    a fuel-gauge reading (``energy_j``) - the bookkeeping methods are only used
    by the simulator.
    """

    def __init__(self, cfg: EnergyStorageConfig, charge_efficiency: float = 1.0):
        self.cfg = cfg
        self.charge_efficiency = charge_efficiency
        self._energy = cfg.capacity_j * cfg.initial_soc
        self.wasted_j = 0.0  # harvested energy that could not be stored (full buffer)

    def reset(self, rng: np.random.Generator) -> None:
        if self.cfg.initial_soc_range is not None:
            soc = rng.uniform(*self.cfg.initial_soc_range)
        else:
            soc = self.cfg.initial_soc
        self._energy = float(np.clip(soc, 0.0, 1.0)) * self.cfg.capacity_j
        self.wasted_j = 0.0

    # -- EnergyStorage protocol -------------------------------------------
    def capacity_j(self) -> float:
        return self.cfg.capacity_j

    def energy_j(self) -> float:
        return self._energy

    def soc(self) -> float:
        return self._energy / self.cfg.capacity_j

    def reserve_j(self) -> float:
        return self.cfg.reserve_soc * self.cfg.capacity_j

    # -- simulator-only bookkeeping -----------------------------------------
    def charge(self, energy_j: float) -> float:
        """Add harvested energy; returns the amount actually stored."""
        usable = max(0.0, energy_j) * self.charge_efficiency
        room = self.cfg.capacity_j - self._energy
        stored = min(usable, room)
        self.wasted_j += usable - stored
        self._energy += stored
        return stored

    def discharge(self, energy_j: float) -> float:
        """Remove energy; returns the amount actually delivered (the buffer
        cannot go negative, so a depleted buffer delivers less than asked)."""
        delivered = min(max(0.0, energy_j), self._energy)
        self._energy -= delivered
        return delivered

    def can_afford(self, energy_j: float, keep_reserve: bool = True) -> bool:
        floor = self.reserve_j() if keep_reserve else 0.0
        return self._energy - energy_j >= floor

    def is_depleted(self) -> bool:
        return self._energy <= 1e-12


# ---------------------------------------------------------------------------
# Solar harvesting
# ---------------------------------------------------------------------------
class SolarEnergySource:
    """Stochastic daily solar cycle with cloud attenuation.

    ground-truth power:  P(t) = P_max * profile(t) * cloud(t) * efficiency
    profile(t)   = sin(pi * (h - sunrise) / (sunset - sunrise)) during daylight, else 0
    cloud(t)     = clip(k_day + c(t), 0, 1)   where
    k_day        ~ AR(1) across days around ``clearness_mean``
    c(t)         ~ AR(1) within the day (fast cloud passages)

    The node observes a *noisy* measurement of P(t) (``measured_power_w``),
    never the future profile.
    """

    def __init__(self, cfg: HarvestingConfig, timestep_s: float):
        self.cfg = cfg
        self.dt = timestep_s
        self._rng = np.random.default_rng()
        self.reset(self._rng, start_time_s=0.0)

    def reset(self, rng: np.random.Generator, start_time_s: float) -> None:
        self._rng = rng
        self._day = int(start_time_s // DAY_S)
        self._k_day = self._draw_clearness(prev=None)
        self._cloud_dev = 0.0
        self._power_w = 0.0
        self._measured_w = 0.0
        self.update(start_time_s)

    def _draw_clearness(self, prev: float | None) -> float:
        c = self.cfg
        if prev is None:
            k = self._rng.normal(c.clearness_mean, c.clearness_std)
        else:
            k = c.clearness_mean + c.clearness_autocorr * (prev - c.clearness_mean) + math.sqrt(
                max(0.0, 1 - c.clearness_autocorr**2)
            ) * self._rng.normal(0.0, c.clearness_std)
        return float(np.clip(k, 0.05, 1.0))

    def solar_profile(self, time_s: float) -> float:
        """Deterministic clear-sky shape in [0, 1] for a given wall-clock time."""
        h = (time_s % DAY_S) / 3600.0
        c = self.cfg
        if h <= c.sunrise_hour or h >= c.sunset_hour:
            return 0.0
        return math.sin(math.pi * (h - c.sunrise_hour) / (c.sunset_hour - c.sunrise_hour))

    def update(self, time_s: float) -> None:
        """Advance the stochastic processes and compute the power for the
        interval starting at ``time_s``."""
        c = self.cfg
        day = int(time_s // DAY_S)
        if day != self._day:
            self._k_day = self._draw_clearness(prev=self._k_day)
            self._day = day
        self._cloud_dev = c.cloud_autocorr * self._cloud_dev + math.sqrt(
            max(0.0, 1 - c.cloud_autocorr**2)
        ) * self._rng.normal(0.0, c.cloud_noise_std)
        cloud = float(np.clip(self._k_day + self._cloud_dev, 0.0, 1.0))
        self._power_w = c.max_power_w * self.solar_profile(time_s) * cloud * c.efficiency
        noise = 1.0 + self._rng.normal(0.0, c.measurement_noise_std)
        self._measured_w = max(0.0, self._power_w * noise)

    # -- EnergySource protocol (measurable) --------------------------------
    def measured_power_w(self) -> float:
        return self._measured_w

    # -- simulator-only ground truth ----------------------------------------
    def true_power_w(self) -> float:
        return self._power_w

    def harvested_energy_j(self) -> float:
        """Energy delivered to the storage during the current interval."""
        return self._power_w * self.dt

    @property
    def daily_clearness(self) -> float:
        return self._k_day
