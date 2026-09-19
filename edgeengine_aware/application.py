"""Remote monitoring application: information utility and priority.

The application is the *consumer* of the data. It has two faces:

1. **What it knows** - only the packets that were delivered. From them (and
   from the age of the last one) it derives the *priority* it sends back to
   the node. This is the realistic part: a real back-end would do exactly the
   same and push the priority through a downlink.

2. **How valuable the information is** - the *utility*. Utility is an
   evaluation quantity, so the simulator is allowed to compare what the
   application believes with the hidden ground truth. This is used for the
   reward only and never enters the policy observation.

Utility has two parts.

**Tracking utility (every step)** - the value of the application holding an
accurate picture of the field *right now*::

    u_track(t) = tracking_weight * criticality(true_t) * exp(-|v_app - true_t| / error_scale)

where ``v_app`` is the last delivered value (0 utility before the first
packet). Stale information drifts away from the truth, noisy low-cost
readings sit further from it, and a rain event makes the old value suddenly
wrong: all three reduce ``u_track`` until a fresh accurate report arrives. A
report identical to the previous one changes nothing, so redundant
transmissions earn nothing and only cost energy.

**Packet bonus (on delivery)** - a shaping term that credits the packet that
fixes the picture, so the agent gets an immediate signal::

    u_packet = criticality(true) * freshness * ( w_gain * tanh(gain / gain_scale)
                                                 + w_event * event * accuracy )
    gain      = |v_app_before - true| - |v_reported - true|   (signed! a worse report is penalised)
    freshness = exp(-measurement_age_at_tx / tau_freshness_s)
    event     = 1 if an environmental event happened since the last delivered packet
    accuracy  = exp(-|v_reported - true| / error_scale)

Both parts share::

    criticality = 1 + criticality_gain * exp(-dist(true, nearest threshold) / criticality_scale)
                  (saturates at 1 + criticality_gain below the critical threshold)

so information is worth more when the crop is close to water stress.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .agriculture import FieldState
from .config import AgricultureConfig, ApplicationConfig
from .interfaces import Packet
from .observation import PRIORITY_ELEVATED, PRIORITY_ROUTINE, PRIORITY_URGENT


@dataclass
class UtilityBreakdown:
    """Details of the packet bonus (for ``info`` and debugging)."""

    total: float
    accuracy: float
    freshness: float
    gain: float
    event: float
    criticality: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass
class TrackingStatus:
    """Per-step evaluation of the application's picture of the field."""

    utility: float
    error: float | None
    """|believed - true| (None before the first delivered packet)."""
    accuracy: float
    criticality: float

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


class RemoteMonitoringApplication:
    """Crop water-stress monitoring back-end."""

    def __init__(self, cfg: ApplicationConfig, agri: AgricultureConfig, timestep_s: float):
        self.cfg = cfg
        self.agri = agri
        self.dt = timestep_s
        self._rng = np.random.default_rng()
        self.reset(self._rng, start_time_s=0.0)

    # -- lifecycle ----------------------------------------------------------
    def reset(self, rng: np.random.Generator, start_time_s: float) -> None:
        self._rng = rng
        self._now = start_time_s
        self._episode_start = start_time_s
        self.last_packet: Packet | None = None
        self.last_received_at_s: float | None = None
        self._request_until_s: float | None = None
        self._request_level = PRIORITY_ROUTINE
        self._unreported_event_time_s: float | None = None
        self._priority = PRIORITY_ROUTINE
        self.packets_received = 0

    # -- knowledge of the application -------------------------------------
    @property
    def believed_value(self) -> float | None:
        return None if self.last_packet is None else self.last_packet.measurement.value

    def age_of_information_s(self, now_s: float | None = None) -> float:
        """Age of the freshest information available at the application.

        Before the first packet, AoI counts from the start of the episode
        (the application has *no* information yet)."""
        now = self._now if now_s is None else now_s
        if self.last_packet is None or self.last_received_at_s is None:
            return now - self._episode_start
        return (now - self.last_received_at_s) + self.last_packet.measurement_age_s

    def priority(self) -> int:
        return self._priority

    def _compute_priority(self, now_s: float) -> int:
        c, a = self.cfg, self.agri
        prio = PRIORITY_ROUTINE
        aoi = self.age_of_information_s(now_s)
        v = self.believed_value
        if v is not None:
            if v < a.critical_threshold:
                prio = PRIORITY_URGENT
            elif v < a.warning_threshold:
                prio = PRIORITY_ELEVATED
        if aoi > c.aoi_urgent_s:
            prio = PRIORITY_URGENT
        elif aoi > c.aoi_elevated_s:
            prio = max(prio, PRIORITY_ELEVATED)
        if self._request_until_s is not None and now_s < self._request_until_s:
            prio = max(prio, self._request_level)
        return prio

    # -- dynamics -----------------------------------------------------------
    def step(self, now_s: float, field_state: FieldState) -> None:
        """Advance the application clock; register external requests and
        ground-truth events (the latter only for utility evaluation)."""
        c = self.cfg
        self._now = now_s
        # external monitoring campaigns requested e.g. by an agronomist
        if self._request_until_s is not None and now_s >= self._request_until_s:
            self._request_until_s = None
        if self._request_until_s is None and self._rng.random() < c.request_rate_per_day * self.dt / 86400.0:
            dur = self._rng.uniform(*c.request_duration_range_s)
            self._request_until_s = now_s + dur
            self._request_level = PRIORITY_URGENT if self._rng.random() < c.request_urgent_fraction else PRIORITY_ELEVATED
        # environmental events worth reporting (privileged bookkeeping)
        if field_state.event_occurred:
            self._unreported_event_time_s = now_s
        if self._unreported_event_time_s is not None and now_s - self._unreported_event_time_s > c.event_memory_s:
            self._unreported_event_time_s = None
        self._priority = self._compute_priority(now_s)

    @property
    def has_unreported_event(self) -> bool:
        return self._unreported_event_time_s is not None

    @property
    def external_request_active(self) -> bool:
        return self._request_until_s is not None

    # -- utility (privileged) -----------------------------------------------
    def criticality(self, true_moisture: float) -> float:
        a, c = self.agri, self.cfg
        if true_moisture <= a.critical_threshold:
            return 1.0 + c.criticality_gain
        dist = min(abs(true_moisture - a.warning_threshold), abs(true_moisture - a.critical_threshold))
        return 1.0 + c.criticality_gain * math.exp(-dist / c.criticality_scale)

    def tracking(self, field_state: FieldState) -> TrackingStatus:
        """Per-step tracking utility of the application's current belief."""
        crit = self.criticality(field_state.soil_moisture)
        v = self.believed_value
        if v is None:
            return TrackingStatus(0.0, None, 0.0, crit)
        err = abs(v - field_state.soil_moisture)
        acc = math.exp(-err / self.cfg.error_scale)
        return TrackingStatus(self.cfg.tracking_weight * crit * acc, err, acc, crit)

    def receive(self, packet: Packet, now_s: float, field_state: FieldState) -> UtilityBreakdown:
        """Register a delivered packet and return its bonus utility."""
        c = self.cfg
        m = packet.measurement
        truth = field_state.soil_moisture

        err_after = abs(m.value - truth)
        err_before = abs(self.believed_value - truth) if self.believed_value is not None else c.error_scale * 3.0
        gain = err_before - err_after
        accuracy = math.exp(-err_after / c.error_scale)
        freshness = math.exp(-max(0.0, packet.measurement_age_s) / c.tau_freshness_s)
        event = 1.0 if self._unreported_event_time_s is not None else 0.0
        crit = self.criticality(truth)
        total = crit * freshness * (c.w_gain * math.tanh(gain / c.gain_scale) + c.w_event * event * accuracy)

        # update knowledge
        self.last_packet = packet
        self.last_received_at_s = now_s
        self.packets_received += 1
        self._unreported_event_time_s = None
        self._priority = self._compute_priority(now_s)
        return UtilityBreakdown(total, accuracy, freshness, gain, event, crit)
