"""Simulated scalar sensor (soil moisture, CO2, temperature - any normalised quantity).

The sensor reads the *true* field moisture and returns a noisy measurement
whose noise level depends on the requested sensing level. A real driver would
implement the same ``Sensor`` protocol on top of an ADC / I2C transaction.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .config import SensingConfig
from .interfaces import Measurement


class SimulatedSoilMoistureSensor:
    """``Sensor`` implementation backed by a ground-truth callable."""

    def __init__(self, cfg: SensingConfig, true_value: Callable[[], float]):
        self.cfg = cfg
        self._true_value = true_value
        self._rng = np.random.default_rng()

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng

    # -- Sensor protocol ----------------------------------------------------
    def energy_cost_j(self, level: int) -> float:
        return self.cfg.energy_j[level]

    def noise_std(self, level: int) -> float:
        return self.cfg.noise_std[level]

    def read(self, level: int, now_s: float) -> Measurement:
        if level < 1 or level >= self.cfg.n_levels:
            raise ValueError(f"sensing level must be in [1, {self.cfg.n_levels}), got {level}")
        truth = self._true_value()
        noise = self._rng.normal(0.0, self.cfg.noise_std[level]) + self.cfg.bias[level]
        value = float(np.clip(truth + noise, 0.0, 1.0))
        return Measurement(value=value, timestamp_s=now_s, level=level, noise_std=self.cfg.noise_std[level])


SimulatedScalarSensor = SimulatedSoilMoistureSensor  # domain-neutral name
