"""Abstract low-power long-range link (LoRa-like).

Modelled at the level of *one uplink attempt*: it costs ``tx_energy_j`` and
succeeds with a probability that follows a slowly varying channel-quality
process. No PHY/MAC details (spreading factor, duty cycle, collisions) are
modelled yet; the class is the natural place to add them later.
"""

from __future__ import annotations

import math

import numpy as np

from .config import CommunicationConfig
from .interfaces import Packet


class SimulatedLoRaRadio:
    """``Radio`` implementation with a stochastic channel.

    success_prob(t) = clip(base_success_prob + q(t), min_success_prob, 1)
    q(t) ~ AR(1) zero-mean channel-quality perturbation
    """

    def __init__(self, cfg: CommunicationConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng()
        self._q = 0.0
        self.last_success: bool | None = None
        self.last_packet: Packet | None = None
        self.attempts = 0
        self.successes = 0

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._q = 0.0
        self.last_success = None
        self.last_packet = None
        self.attempts = 0
        self.successes = 0

    def update_channel(self) -> None:
        """Advance the channel process by one timestep (called by the env)."""
        c = self.cfg
        self._q = c.channel_autocorr * self._q + math.sqrt(max(0.0, 1 - c.channel_autocorr**2)) * self._rng.normal(
            0.0, c.channel_noise_std
        )

    # -- simulator-only ground truth ----------------------------------------
    def success_probability(self) -> float:
        return float(np.clip(self.cfg.base_success_prob + self._q, self.cfg.min_success_prob, 1.0))

    # -- Radio protocol -----------------------------------------------------
    def tx_energy_j(self) -> float:
        return self.cfg.tx_energy_j

    def transmit(self, packet: Packet) -> bool:
        self.attempts += 1
        ok = bool(self._rng.random() < self.success_probability())
        self.last_success = ok
        self.last_packet = packet
        if ok:
            self.successes += 1
        return ok
