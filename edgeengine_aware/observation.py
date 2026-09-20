"""Node-side state tracking and policy-observation construction.

This module is the heart of the simulation-to-real design. It contains:

* :class:`NodeProfile`  - the constants a firmware would keep in flash
  (hardware energy profile, application thresholds, normalisation constants);
* :class:`NodeState`    - the *hardware-measurable state*: every field can be
  obtained on a microcontroller from a fuel gauge, an ADC, a timer, the sensor
  driver, the radio ACKs and the last downlink;
* :class:`NodeStateTracker` - bookkeeping that turns raw events (a new
  measurement, an ACK, a downlink) into a :class:`NodeState`;
* :class:`ObservationBuilder` - the *only* place where a :class:`NodeState` is
  turned into the float32 policy observation vector.

The Gymnasium environment (``env.py``) and the deployment runtime
(``deployment.py``) both use exactly these classes, so the policy sees the
same quantities, normalised the same way, in simulation and on hardware.
Nothing in this module has access to simulator ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .config import EdgeEngineAwareConfig, ObservationConfig
from .interfaces import Measurement, Packet

PRIORITY_ROUTINE, PRIORITY_ELEVATED, PRIORITY_URGENT = 0, 1, 2
N_PRIORITY_LEVELS = 3


# ---------------------------------------------------------------------------
# Constants stored on the node
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NodeProfile:
    """Hardware/application constants known to the node (flash constants)."""

    sensing_energy_j: tuple[float, ...]
    """Energy per sensing level [J]; index 0 is 'no sensing' (0 J)."""

    sensing_noise_std: tuple[float, ...]
    """Nominal noise per sensing level (quality tags)."""

    tx_energy_j: tuple[float, ...]
    """Energy per radio mode [J] (index = mode)."""

    tx_power_dbm: tuple[float, ...]
    sensitivity_dbm: tuple[float, ...]
    """Link-budget constants per radio mode, used to turn a measured ACK
    margin into a mode-independent path-loss estimate."""

    reference_mode: int
    """Mode whose energy is reported in the observation."""

    warning_threshold: float
    critical_threshold: float
    timestep_s: float = 900.0
    """Decision period [s] (wake-up interval of the firmware)."""

    baseline_power_w: float = 200e-6
    """Always-on consumption used by the feasibility rule."""

    reserve_soc: float = 0.02
    """Brown-out reserve used by the feasibility rule."""

    ack_available: bool = True
    """Whether uplinks are confirmed (see CommunicationConfig.ack_available)."""

    observation: ObservationConfig = field(default_factory=ObservationConfig)

    @classmethod
    def from_config(cls, cfg: EdgeEngineAwareConfig) -> "NodeProfile":
        return cls(
            sensing_energy_j=tuple(cfg.sensing.energy_j),
            sensing_noise_std=tuple(cfg.sensing.noise_std),
            tx_energy_j=tuple(m.energy_j for m in cfg.communication.modes),
            tx_power_dbm=tuple(m.tx_power_dbm for m in cfg.communication.modes),
            sensitivity_dbm=tuple(m.sensitivity_dbm for m in cfg.communication.modes),
            reference_mode=cfg.communication.reference_mode,
            warning_threshold=cfg.agriculture.warning_threshold,
            critical_threshold=cfg.agriculture.critical_threshold,
            timestep_s=cfg.time.timestep_s,
            baseline_power_w=cfg.mcu.baseline_power_w,
            reserve_soc=cfg.storage.reserve_soc,
            ack_available=cfg.communication.ack_available,
            observation=cfg.observation,
        )

    @property
    def baseline_energy_j(self) -> float:
        """Baseline energy of one decision interval [J]."""
        return self.baseline_power_w * self.timestep_s

    @property
    def n_modes(self) -> int:
        return len(self.tx_energy_j)

    def path_loss_from_margin(self, mode: int, margin_db: float) -> float:
        """Path loss implied by a margin measured with ``mode`` [dB]."""
        return self.tx_power_dbm[mode] - self.sensitivity_dbm[mode] - margin_db

    def margin_for_mode(self, mode: int, path_loss_db: float) -> float:
        """Expected margin of ``mode`` for a given path-loss estimate [dB]."""
        return self.tx_power_dbm[mode] - path_loss_db - self.sensitivity_dbm[mode]


# ---------------------------------------------------------------------------
# Hardware-measurable state
# ---------------------------------------------------------------------------
@dataclass
class NodeState:
    """Everything the node knows about itself at decision time.

    All quantities are in physical units; normalisation happens in
    :class:`ObservationBuilder`. The docstring of each field names the
    hardware source that would provide it on a real device.
    """

    time_of_day_s: float
    """Seconds since midnight (RTC)."""

    stored_energy_j: float
    """Stored energy (fuel gauge / voltage-based estimate)."""

    capacity_j: float
    """Usable capacity (flash constant, or fuel-gauge full-charge value)."""

    harvest_power_w: float
    """Average harvesting power over the interval that just elapsed, as
    integrated by the harvester monitor since the previous wake-up."""

    harvest_power_recent_w: float
    """EWMA of the measured harvesting power (computed in firmware)."""

    has_measurement: bool
    """Whether a measurement is stored in RAM."""

    measurement_value: float
    """Latest stored measurement (0 when none)."""

    measurement_noise_std: float
    """Quality tag of the latest stored measurement (from the profile)."""

    measurement_age_s: float
    """now - timestamp of the latest measurement (timer)."""

    has_reported: bool
    """Whether at least one uplink has been acknowledged."""

    time_since_tx_success_s: float
    """now - time of the last acknowledged uplink (timer)."""

    app_info_age_s: float
    """Node-side estimate of the age of information at the application:
    time since last ACK + age of the measurement that was in that packet.
    Exact when ACKs are available; with unconfirmed uplinks the node assumes
    delivery, so the estimate is optimistic (see
    ``CommunicationConfig.ack_available``)."""

    reported_value: float
    """Value contained in the last acknowledged uplink (0 when none)."""

    app_priority: int
    """Latest priority received from the application (downlink), 0/1/2."""

    link_quality: float
    """EWMA of ACK outcomes in [0, 1] (1 = every recent uplink delivered)."""

    has_link_estimate: bool
    """Whether at least one ACK carried a usable margin measurement."""

    path_loss_est_db: float
    """Path loss implied by the margin of the last ACK, converted with the
    profile's link-budget constants (mode independent)."""

    sensing_energy_low_j: float
    sensing_energy_high_j: float
    tx_energy_j: float
    """Hardware energy profile (flash constants or on-line measurements);
    ``tx_energy_j`` is the energy of the reference radio mode."""

    def soc(self) -> float:
        return 0.0 if self.capacity_j <= 0 else self.stored_energy_j / self.capacity_j


# ---------------------------------------------------------------------------
# Tracker: raw events -> NodeState
# ---------------------------------------------------------------------------
class NodeStateTracker:
    """Firmware-style bookkeeping shared by simulation and deployment.

    Usage per decision step::

        tracker.begin_step(now_s, time_of_day_s, energy_j, capacity_j,
                           harvest_power_w, priority)
        state = tracker.state()            # -> policy observation
        ... execute the action ...
        tracker.on_measurement(measurement)   # if sensing happened
        tracker.on_transmission(packet, acked, now_s, mode, margin_db)  # if a tx happened
        # (pass the mode used and the ACK margin, or the path-loss estimate cannot be updated correctly)
    """

    def __init__(self, profile: NodeProfile):
        self.profile = profile
        self.reset()

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self._now_s = 0.0
        self._time_of_day_s = 0.0
        self._energy_j = 0.0
        self._capacity_j = 1.0
        self._harvest_w = 0.0
        self._harvest_recent_w = 0.0
        self._harvest_initialised = False
        self._measurement: Measurement | None = None
        self._last_ack_time_s: float | None = None
        self._last_ack_packet: Packet | None = None
        self._priority = PRIORITY_ROUTINE
        self._link_quality = 1.0
        self._path_loss_est_db: float | None = None

    def begin_step(
        self,
        now_s: float,
        time_of_day_s: float,
        energy_j: float,
        capacity_j: float,
        harvest_power_w: float,
        priority: int | None,
    ) -> None:
        """Update the periodically sampled quantities.

        ``priority=None`` means 'no new downlink received' (keep the old one).
        """
        self._now_s = now_s
        self._time_of_day_s = time_of_day_s
        self._energy_j = energy_j
        self._capacity_j = capacity_j
        self._harvest_w = max(0.0, harvest_power_w)
        alpha = self.profile.observation.harvest_ewma_alpha
        if not self._harvest_initialised:
            self._harvest_recent_w = self._harvest_w
            self._harvest_initialised = True
        else:
            self._harvest_recent_w = (1 - alpha) * self._harvest_recent_w + alpha * self._harvest_w
        if priority is not None:
            self._priority = int(priority)

    def on_measurement(self, measurement: Measurement) -> None:
        self._measurement = measurement

    def on_transmission(self, packet: Packet, acked: bool | None, now_s: float, mode: int = 0, margin_db: float | None = None) -> None:
        """Register an uplink attempt made with radio ``mode``.

        ``acked`` is the ACK outcome, or ``None`` when the link gives no
        confirmation (``profile.ack_available`` is False): the node then
        assumes delivery for its age bookkeeping and leaves the link-quality
        indicator untouched. ``margin_db`` is the link margin measured from the
        ACK (if any); it is converted into a mode-independent path-loss estimate.
        """
        if acked is None or not self.profile.ack_available:
            self._last_ack_time_s = now_s
            self._last_ack_packet = packet
            return
        alpha = self.profile.observation.link_ewma_alpha
        self._link_quality = (1 - alpha) * self._link_quality + alpha * (1.0 if acked else 0.0)
        if acked:
            self._last_ack_time_s = now_s
            self._last_ack_packet = packet
            if margin_db is not None:
                self._path_loss_est_db = self.profile.path_loss_from_margin(mode, margin_db)
        else:
            # A lost uplink in ``mode`` says the margin of that mode was about
            # zero or negative, i.e. the path loss is at least the mode's link
            # budget: raise the estimate accordingly (a firmware-friendly,
            # conservative update that makes the node escalate after failures).
            floor_db = self.profile.path_loss_from_margin(mode, 0.0)
            self._path_loss_est_db = floor_db if self._path_loss_est_db is None else max(self._path_loss_est_db, floor_db)

    def set_priority(self, priority: int) -> None:
        """Explicit downlink handling (used when priority arrives with an ACK)."""
        self._priority = int(priority)

    # -- accessors ----------------------------------------------------------
    @property
    def measurement(self) -> Measurement | None:
        return self._measurement

    @property
    def last_acked_packet(self) -> Packet | None:
        return self._last_ack_packet

    @property
    def priority(self) -> int:
        return self._priority

    @property
    def path_loss_est_db(self) -> float | None:
        return self._path_loss_est_db

    def state(self) -> NodeState:
        p = self.profile
        age_cap = p.observation.age_scale_s
        if self._measurement is None:
            has_meas, m_value, m_noise, m_age = False, 0.0, 0.0, age_cap
        else:
            has_meas = True
            m_value = self._measurement.value
            m_noise = self._measurement.noise_std
            m_age = self._now_s - self._measurement.timestamp_s
        if self._last_ack_time_s is None or self._last_ack_packet is None:
            has_rep, t_since, app_age, rep_value = False, age_cap, age_cap, 0.0
        else:
            has_rep = True
            t_since = self._now_s - self._last_ack_time_s
            app_age = t_since + self._last_ack_packet.measurement_age_s
            rep_value = self._last_ack_packet.measurement.value
        return NodeState(
            time_of_day_s=self._time_of_day_s,
            stored_energy_j=self._energy_j,
            capacity_j=self._capacity_j,
            harvest_power_w=self._harvest_w,
            harvest_power_recent_w=self._harvest_recent_w,
            has_measurement=has_meas,
            measurement_value=m_value,
            measurement_noise_std=m_noise,
            measurement_age_s=m_age,
            has_reported=has_rep,
            time_since_tx_success_s=t_since,
            app_info_age_s=app_age,
            reported_value=rep_value,
            app_priority=self._priority,
            link_quality=self._link_quality,
            has_link_estimate=self._path_loss_est_db is not None,
            path_loss_est_db=self._path_loss_est_db if self._path_loss_est_db is not None else p.observation.path_loss_max_db,
            sensing_energy_low_j=p.sensing_energy_j[1],
            sensing_energy_high_j=p.sensing_energy_j[2],
            tx_energy_j=p.tx_energy_j[p.reference_mode],
        )


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ObservationField:
    name: str
    description: str
    normalisation: str
    hardware_source: str
    low: float = 0.0
    high: float = 1.0


OBSERVATION_FIELDS: tuple[ObservationField, ...] = (
    ObservationField("battery_soc", "State of charge of the energy storage", "E / E_max", "fuel gauge / ADC"),
    ObservationField("harvest_power", "Harvesting power measured in the last interval", "P / harvest_ref_power_w, clipped to 1", "harvester current monitor"),
    ObservationField("harvest_recent", "EWMA of recent harvesting power", "P_ewma / harvest_ref_power_w, clipped to 1", "computed in firmware"),
    ObservationField("time_of_day_sin", "Time of day, sine component", "(sin(2*pi*t/86400) + 1) / 2", "RTC"),
    ObservationField("time_of_day_cos", "Time of day, cosine component", "(cos(2*pi*t/86400) + 1) / 2", "RTC"),
    ObservationField("measurement", "Latest locally stored soil-moisture measurement (0 if none)", "moisture units (already in [0, 1])", "sensor driver + RAM"),
    ObservationField("measurement_quality", "Quality tag of the latest measurement", "1 - 0.8*noise/noise_low (0 none, 0.2 low, 0.8 high)", "hardware profile"),
    ObservationField("measurement_age", "Time since the latest measurement", "age / age_scale_s, clipped to 1 (1 if none)", "timer"),
    ObservationField("time_since_tx_success", "Time since the last acknowledged uplink", "t / age_scale_s, clipped to 1 (1 if none)", "timer + radio ACK"),
    ObservationField("app_info_age", "Node-side estimate of the information age at the application", "age / age_scale_s, clipped to 1 (1 if none)", "timer + radio ACK"),
    ObservationField("reported_value", "Value in the last acknowledged uplink (0 if none)", "moisture units", "RAM"),
    ObservationField("app_priority", "Priority requested by the application", "priority / 2", "downlink message"),
    ObservationField("importance", "Node-side importance of the stored measurement (proximity to thresholds)", "exp(-dist/importance_scale), 1 below critical, 0 if none", "computed in firmware from flash thresholds"),
    ObservationField("link_quality", "EWMA of recent ACK outcomes", "already in [0, 1]", "radio ACK"),
    ObservationField("path_loss_est", "Path-loss estimate from the margin of the last ACK (mode independent)", "(PL - path_loss_min_db) / (path_loss_max_db - path_loss_min_db), clipped; 1 if none", "radio ACK SNR/RSSI + flash link-budget table"),
    ObservationField("sense_low_cost", "Energy of a low-cost sensing operation", "E / E_max, clipped to 1", "hardware profile"),
    ObservationField("sense_high_cost", "Energy of a high-quality sensing operation", "E / E_max, clipped to 1", "hardware profile"),
    ObservationField("tx_cost", "Energy of one transmission attempt in the reference radio mode", "E / E_max, clipped to 1", "hardware profile"),
)


class ObservationBuilder:
    """Turns a :class:`NodeState` into the float32 policy observation.

    The builder is stateless; all normalisation constants come from the
    :class:`NodeProfile`, so a firmware port only needs the same constants.
    """

    fields: tuple[ObservationField, ...] = OBSERVATION_FIELDS

    def __init__(self, profile: NodeProfile):
        self.profile = profile

    # -- metadata -----------------------------------------------------------
    @property
    def dim(self) -> int:
        return len(self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    @property
    def low(self) -> np.ndarray:
        return np.array([f.low for f in self.fields], dtype=np.float32)

    @property
    def high(self) -> np.ndarray:
        return np.array([f.high for f in self.fields], dtype=np.float32)

    def index(self, name: str) -> int:
        return self.names.index(name)

    def describe(self) -> str:
        """Human-readable table of the observation vector."""
        lines = [f"{'idx':>3}  {'name':<22} {'normalisation':<45} source", "-" * 100]
        for i, f in enumerate(self.fields):
            lines.append(f"{i:>3}  {f.name:<22} {f.normalisation:<45} {f.hardware_source}")
        return "\n".join(lines)

    def to_dict(self, observation: np.ndarray) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self.names, observation)}

    # -- construction -------------------------------------------------------
    def importance(self, value: float, has_measurement: bool) -> float:
        """Node-side importance of the stored measurement in [0, 1]."""
        if not has_measurement:
            return 0.0
        p = self.profile
        if value <= p.critical_threshold:
            return 1.0
        dist = min(abs(value - p.warning_threshold), abs(value - p.critical_threshold))
        return math.exp(-dist / p.observation.importance_scale)

    def quality_tag(self, noise_std: float, has_measurement: bool) -> float:
        """Map the nominal noise of a measurement to a quality in [0, 1].

        The low-cost level maps to a small positive value and the best level
        to a value close to 1, so the policy can tell the modes apart. A node
        with no stored measurement reports 0.
        """
        if not has_measurement:
            return 0.0
        ref = self.profile.sensing_noise_std[1]  # low-cost noise as reference
        if ref <= 0:
            return 1.0
        return float(np.clip(1.0 - 0.8 * noise_std / ref, 0.0, 1.0))

    def build(self, s: NodeState) -> np.ndarray:
        o = self.profile.observation
        cap = max(s.capacity_j, 1e-9)
        age = o.age_scale_s
        href = o.harvest_ref_power_w
        phase = 2.0 * math.pi * s.time_of_day_s / 86400.0
        obs = np.array(
            [
                s.stored_energy_j / cap,
                s.harvest_power_w / href,
                s.harvest_power_recent_w / href,
                (math.sin(phase) + 1.0) / 2.0,
                (math.cos(phase) + 1.0) / 2.0,
                s.measurement_value if s.has_measurement else 0.0,
                self.quality_tag(s.measurement_noise_std, s.has_measurement),
                s.measurement_age_s / age if s.has_measurement else 1.0,
                s.time_since_tx_success_s / age if s.has_reported else 1.0,
                s.app_info_age_s / age if s.has_reported else 1.0,
                s.reported_value if s.has_reported else 0.0,
                s.app_priority / (N_PRIORITY_LEVELS - 1),
                self.importance(s.measurement_value, s.has_measurement),
                s.link_quality,
                (s.path_loss_est_db - o.path_loss_min_db) / (o.path_loss_max_db - o.path_loss_min_db) if s.has_link_estimate else 1.0,
                s.sensing_energy_low_j / cap,
                s.sensing_energy_high_j / cap,
                s.tx_energy_j / cap,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.low, self.high).astype(np.float32)


def observation_names() -> Iterable[str]:
    return (f.name for f in OBSERVATION_FIELDS)
