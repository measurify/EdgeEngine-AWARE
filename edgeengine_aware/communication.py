"""Abstract low-power long-range link (LoRa-like) with selectable modes.

Modelled at the level of *one uplink attempt* in a given *mode* (a spreading
factor / power setting): it costs ``modes[k].energy_j`` and is delivered with a
probability given by the link budget

    margin_k(t) = tx_power_k - path_loss(t) - sensitivity_k          [dB]
    p_k(t)      = 1 / (1 + exp(-margin_k(t) / margin_scale_db))
    path_loss(t) = path_loss_mean_db + slow_fading(t) + fast_fading

where ``slow_fading`` is an AR(1) process (shadowing by vegetation, humidity,
gateway load) and ``fast_fading`` is redrawn at every attempt. On a delivered
packet the node measures the margin of the used mode from the ACK (with noise)
and can convert it into a *path-loss estimate* that is valid for every mode:
this is how it learns which mode is currently affordable.

Duty-cycle limits, collisions and multi-gateway reception are not modelled;
this class is where they belong.
"""

from __future__ import annotations

import math

import numpy as np

from .config import CommunicationConfig
from .interfaces import Packet, TxResult


class SimulatedLoRaRadio:
    """``Radio`` implementation with a link-budget channel."""

    def __init__(self, cfg: CommunicationConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng()
        self.reset(self._rng)

    def reset(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self._slow_db = 0.0
        self.last_result: TxResult | None = None
        self.last_mode: int | None = None
        self.last_packet: Packet | None = None
        self.attempts = 0
        self.successes = 0
        self.attempts_per_mode = [0] * self.cfg.n_modes
        self.successes_per_mode = [0] * self.cfg.n_modes

    def update_channel(self) -> None:
        """Advance the slow-fading process by one timestep (called by the env)."""
        c = self.cfg
        rho = c.slow_fading_autocorr
        self._slow_db = rho * self._slow_db + math.sqrt(max(0.0, 1 - rho**2)) * self._rng.normal(0.0, c.slow_fading_std_db)

    # -- simulator-only ground truth ----------------------------------------
    def path_loss_db(self) -> float:
        """Current path loss without the per-attempt fast fading [dB]."""
        return self.cfg.path_loss_mean_db + self._slow_db

    def margin_db(self, mode: int) -> float:
        m = self.cfg.modes[mode]
        return m.tx_power_dbm - self.path_loss_db() - m.sensitivity_dbm

    def success_probability(self, mode: int | None = None) -> float:
        """Delivery probability of ``mode`` (default: the reference mode) given
        the current slow fading, averaged over the fast fading."""
        if mode is None:
            mode = self.cfg.reference_mode
        margin = self.margin_db(mode)
        # logistic in margin, fast fading adds variance: average over a few points
        z = margin + self.cfg.fast_fading_std_db * np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
        w = np.array([0.1, 0.25, 0.3, 0.25, 0.1])
        return float(np.sum(w / (1.0 + np.exp(-z / self.cfg.margin_scale_db))))

    # -- Radio protocol -----------------------------------------------------
    def n_modes(self) -> int:
        return self.cfg.n_modes

    def tx_energy_j(self, mode: int) -> float:
        return self.cfg.modes[mode].energy_j

    def transmit(self, packet: Packet, mode: int) -> TxResult:
        if not 0 <= mode < self.cfg.n_modes:
            raise ValueError(f"radio mode must be in [0, {self.cfg.n_modes}), got {mode}")
        c = self.cfg
        self.attempts += 1
        self.attempts_per_mode[mode] += 1
        margin = self.margin_db(mode) + self._rng.normal(0.0, c.fast_fading_std_db)
        z = float(np.clip(-margin / c.margin_scale_db, -60.0, 60.0))
        p = 1.0 / (1.0 + math.exp(z))
        ok = bool(self._rng.random() < p)
        measured = (margin + self._rng.normal(0.0, c.ack_margin_noise_db)) if ok else None
        result = TxResult(acked=ok, margin_db=measured)
        self.last_result, self.last_mode, self.last_packet = result, mode, packet
        if ok:
            self.successes += 1
            self.successes_per_mode[mode] += 1
        return result
