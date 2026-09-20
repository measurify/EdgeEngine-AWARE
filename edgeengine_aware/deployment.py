"""Simulation-to-real scaffolding.

Three things live here:

* :class:`NodeController` - the *firmware main loop* written against the
  hardware protocols of ``interfaces.py``. It reuses ``NodeStateTracker``,
  ``ObservationBuilder`` and ``plan_execution`` from the simulator, so the
  policy receives byte-for-byte the same observation vector it saw in
  training and its actions are executed by the same feasibility rule.

* :class:`MockHardwareBackend` - stand-in drivers (a fuel gauge with a minimal
  power-path emulation, a harvester monitor fed by a callable, a sensor driver
  reading a callable with the profile's noise, a radio whose ACKs follow a
  link budget at a fixed path loss). It has **no ground truth**: it only knows
  what a real board would know. It exists to prove that the controller and the
  policy run unchanged outside the simulator, and it is what a real port
  replaces with ADC / I2C / radio-stack calls.

* :class:`PolicyBundle` - a JSON-serialisable description of everything that
  must travel with a trained policy to the microcontroller: observation
  ordering and normalisation constants, action encoding, model parameters and
  metadata. Freezing this contract is the single most important step for a
  credible sim-to-real transfer (see ``docs/deployment.md``).
"""

from __future__ import annotations

import dataclasses
import json
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import __version__
from .actions import DEFAULT_N_MODES, SENSE_NONE, action_nvec, n_flat_actions, plan_execution
from .interfaces import Clock, EnergySource, EnergyStorage, Measurement, Packet, Policy, Radio, RemoteApplication, Sensor, TxResult
from .observation import OBSERVATION_FIELDS, NodeProfile, NodeStateTracker, ObservationBuilder


# ---------------------------------------------------------------------------
# Firmware-style controller
# ---------------------------------------------------------------------------
@dataclass
class HardwareBackend:
    """Bundle of drivers implementing the protocols of ``interfaces.py``."""

    clock: Clock
    storage: EnergyStorage
    source: EnergySource
    sensor: Sensor
    radio: Radio
    application: RemoteApplication


@dataclass
class CycleReport:
    """What happened during one decision cycle (for logging / evaluation)."""

    time_s: float
    observation: np.ndarray
    requested_action: tuple[int, int]
    executed_sensing_level: int
    executed_transmit: bool
    tx_mode: int
    """Radio mode used (-1 when no transmission)."""
    tx_success: bool | None
    rejected: tuple[str, ...]
    measurement: Measurement | None
    energy_spent_j: float = 0.0
    """Nominal sensing + transmission energy of the executed operations [J]."""


class NodeController:
    """Main loop of the deployed node.

    ::

        controller = NodeController(backend, profile, policy)
        while True:
            report = controller.run_cycle()
            sleep_until_next_wakeup(profile.timestep_s)

    Nothing in this class knows whether ``backend`` is simulated or real. All
    constants (energy profile, timestep, baseline, reserve, ACK availability,
    normalisation) come from the :class:`NodeProfile`, i.e. from the exported
    policy bundle, so simulator and firmware cannot drift apart silently.
    ``priority_update_mode`` mirrors ``CommunicationConfig.priority_update_mode``:
    'immediate' reads the downlink priority at every wake-up, 'on_uplink' only
    after an acknowledged uplink.
    """

    def __init__(self, backend: HardwareBackend, profile: NodeProfile, policy: Policy, priority_update_mode: str = "immediate"):
        if priority_update_mode not in ("immediate", "on_uplink"):
            raise ValueError("priority_update_mode must be 'immediate' or 'on_uplink'")
        self.hw = backend
        self.profile = profile
        self.policy = policy
        self.priority_update_mode = priority_update_mode
        self.tracker = NodeStateTracker(profile)
        self.obs_builder = ObservationBuilder(profile)
        self.policy.reset()

    def build_observation(self) -> np.ndarray:
        """Read the measurable quantities and produce the policy input."""
        hw = self.hw
        priority = hw.application.priority() if self.priority_update_mode == "immediate" else None
        self.tracker.begin_step(
            now_s=hw.clock.now_s(),
            time_of_day_s=hw.clock.time_of_day_s(),
            energy_j=hw.storage.energy_j(),
            capacity_j=hw.storage.capacity_j(),
            harvest_power_w=hw.source.measured_power_w(),
            priority=priority,
        )
        return self.obs_builder.build(self.tracker.state())

    def execute(self, action) -> CycleReport:
        """Apply the feasibility rule and drive the drivers."""
        hw, p = self.hw, self.profile
        now = hw.clock.now_s()
        plan = plan_execution(
            action,
            stored_energy_j=hw.storage.energy_j(),
            baseline_energy_j=p.baseline_energy_j,
            reserve_energy_j=p.reserve_soc * hw.storage.capacity_j(),
            sensing_energy_j=p.sensing_energy_j,
            tx_energy_j=p.tx_energy_j,
            has_measurement=self.tracker.measurement is not None,
        )
        measurement = None
        if plan.sensing_level != SENSE_NONE:
            measurement = hw.sensor.read(plan.sensing_level, now)
            self.tracker.on_measurement(measurement)
        tx_success: bool | None = None
        if plan.transmit and self.tracker.measurement is not None:
            packet = Packet(measurement=self.tracker.measurement, sent_at_s=now)
            result = hw.radio.transmit(packet, plan.mode)
            tx_success = result.acked if p.ack_available else None
            self.tracker.on_transmission(packet, tx_success, now, mode=plan.mode, margin_db=result.margin_db)
            if self.priority_update_mode == "on_uplink" and result.acked:
                self.tracker.set_priority(hw.application.priority())  # downlink piggybacked on the ACK
        a = np.asarray(action).reshape(-1)
        report = CycleReport(
            time_s=now,
            observation=np.empty(0),
            requested_action=(int(a[0]), int(a[1])),
            executed_sensing_level=plan.sensing_level,
            executed_transmit=plan.transmit,
            tx_mode=plan.mode,
            tx_success=tx_success,
            rejected=plan.rejected,
            measurement=measurement,
            energy_spent_j=plan.sensing_energy_j + plan.tx_energy_j,
        )
        return report

    def run_cycle(self) -> CycleReport:
        obs = self.build_observation()
        action = self.policy.act(obs)
        report = self.execute(action)
        report.observation = obs
        return report


# ---------------------------------------------------------------------------
# Mock hardware (no ground truth!)
# ---------------------------------------------------------------------------
class MockClock:
    def __init__(self, start_s: float = 0.0, timestep_s: float = 900.0):
        self._t = start_s
        self.dt = timestep_s

    def now_s(self) -> float:
        return self._t

    def time_of_day_s(self) -> float:
        return self._t % 86400.0

    def tick(self) -> None:
        self._t += self.dt


class MockFuelGauge:
    """Battery monitor over a scripted energy value."""

    def __init__(self, capacity_j: float, energy_j: float):
        self._cap = capacity_j
        self._e = float(np.clip(energy_j, 0.0, capacity_j))

    def capacity_j(self) -> float:
        return self._cap

    def energy_j(self) -> float:
        return self._e

    def apply_delta_j(self, delta_j: float) -> None:
        """Power-path emulation hook (charge > 0, discharge < 0)."""
        self._e = float(np.clip(self._e + delta_j, 0.0, self._cap))


class MockHarvesterMonitor:
    """Reports the average power of the *elapsed* interval, like a coulomb
    counter read at wake-up (``power_fn`` is a scripted or recorded profile)."""

    def __init__(self, power_fn: Callable[[float], float], clock: MockClock, samples: int = 8):
        self._fn = power_fn
        self._clock = clock
        self._n = samples

    def measured_power_w(self) -> float:
        end = self._clock.now_s()
        ts = np.linspace(end - self._clock.dt, end, self._n, endpoint=False)
        return max(0.0, float(np.mean([self._fn(t) for t in ts])))


class MockSensorDriver:
    """Returns values from a trace (e.g. a CSV recorded in the field)."""

    def __init__(self, trace: Callable[[float], float], energy_j: Sequence[float], noise_std: Sequence[float], rng: np.random.Generator | None = None):
        self._trace = trace
        self._energy = tuple(energy_j)
        self._noise = tuple(noise_std)
        self._rng = rng or np.random.default_rng(0)

    def energy_cost_j(self, level: int) -> float:
        return self._energy[level]

    def noise_std(self, level: int) -> float:
        return self._noise[level]

    def read(self, level: int, now_s: float) -> Measurement:
        raw = self._trace(now_s) + self._rng.normal(0.0, self._noise[level])
        return Measurement(float(np.clip(raw, 0.0, 1.0)), now_s, level, self._noise[level])


class MockRadio:
    """Radio stub: fixed path loss, ACK probability from the link budget of
    the requested mode, margin reported back like a LinkCheck answer."""

    def __init__(self, profile: NodeProfile, path_loss_db: float = 139.0, rng: np.random.Generator | None = None):
        self._profile = profile
        self._pl = path_loss_db
        self._rng = rng or np.random.default_rng(1)
        self.sent: list[tuple[Packet, int]] = []

    def n_modes(self) -> int:
        return self._profile.n_modes

    def tx_energy_j(self, mode: int) -> float:
        return self._profile.tx_energy_j[mode]

    def transmit(self, packet: Packet, mode: int) -> TxResult:
        self.sent.append((packet, mode))
        margin = self._profile.margin_for_mode(mode, self._pl) + self._rng.normal(0.0, 2.0)
        ok = bool(self._rng.random() < 1.0 / (1.0 + np.exp(-margin / 1.5)))
        return TxResult(acked=ok, margin_db=float(margin) if ok else None)


class MockDownlink:
    """Priority as received from the back-end (defaults to routine)."""

    def __init__(self, priority: int = 0):
        self._p = priority

    def priority(self) -> int:
        return self._p

    def set_priority(self, p: int) -> None:
        self._p = int(p)


@dataclass
class MockHardwareBackend(HardwareBackend):
    """A fake board: drivers plus a minimal power-path emulation.

    Call :meth:`end_of_cycle` after every controller cycle to advance the clock
    and to debit/credit the fuel gauge (baseline + executed operations - harvest).
    """

    profile: NodeProfile = None  # type: ignore[assignment]

    def end_of_cycle(self, report: CycleReport) -> None:
        # same order as the simulator: the interval's load is drawn first, the
        # energy harvested during the interval is credited afterwards
        self.storage.apply_delta_j(-(self.profile.baseline_energy_j + report.energy_spent_j))  # type: ignore[attr-defined]
        self.clock.tick()  # type: ignore[attr-defined]
        harvested_j = self.source.measured_power_w() * self.profile.timestep_s  # the interval that just elapsed
        self.storage.apply_delta_j(+harvested_j)  # type: ignore[attr-defined]


def make_mock_backend(profile: NodeProfile, *, capacity_j: float = 300.0, energy_j: float = 150.0, peak_harvest_w: float = 0.003, seed: int = 0) -> MockHardwareBackend:
    """A complete fake board with a plausible diurnal harvest and a slowly
    drying soil trace. Useful for tests and as a template for a real port."""
    rng = np.random.default_rng(seed)
    clock = MockClock(0.0, profile.timestep_s)

    def harvest(t: float) -> float:
        h = (t % 86400.0) / 3600.0
        return peak_harvest_w * float(np.sin(np.pi * (h - 6.0) / 12.0)) if 6.0 < h < 18.0 else 0.0

    def soil_trace(t: float) -> float:
        return 0.6 - 0.05 * (t / 86400.0)

    return MockHardwareBackend(
        clock=clock,
        storage=MockFuelGauge(capacity_j, energy_j),
        source=MockHarvesterMonitor(harvest, clock),
        sensor=MockSensorDriver(soil_trace, profile.sensing_energy_j, profile.sensing_noise_std, rng),
        radio=MockRadio(profile, rng=rng),
        application=MockDownlink(0),
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Policy export
# ---------------------------------------------------------------------------
@dataclass
class PolicyBundle:
    """Everything that must be preserved when moving a policy to a device."""

    policy_type: str
    """'rule_based', 'mlp', 'q_table', ... (free-form identifier)."""

    observation_names: list[str]
    """Ordering of the observation vector (index = position)."""

    observation_normalisation: list[str]
    """Human-readable formula per component (documentation of the contract)."""

    profile: dict[str, Any]
    """NodeProfile constants (energy costs, thresholds, normalisation scales)."""

    action_encoding: dict[str, Any]
    """MultiDiscrete nvec, flat-index formula and semantics of each value."""

    model: dict[str, Any] = field(default_factory=dict)
    """Parameters: thresholds for a rule-based policy, weights/biases/activations
    for an MLP, table for a tabular policy, ... (lists, not ndarrays)."""

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def observation_dim(self) -> int:
        return len(self.observation_names)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(dataclasses.asdict(self), indent=indent, default=_json_default)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(self.to_json())
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PolicyBundle":
        return cls(**json.loads(Path(path).read_text()))


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def action_encoding_spec(n_modes: int = DEFAULT_N_MODES) -> dict[str, Any]:
    transmit = {"0": "no transmission"}
    transmit.update({str(k + 1): f"transmit latest stored measurement with radio mode {k}" for k in range(n_modes)})
    return {
        "type": "MultiDiscrete",
        "nvec": list(action_nvec(n_modes)),
        "n_flat": n_flat_actions(n_modes),
        "flat_index": f"sensing_level * {1 + n_modes} + transmit",
        "sensing_level": {"0": "no sensing", "1": "low-cost sensing", "2": "high-quality sensing"},
        "transmit": transmit,
    }


def export_policy(policy: Any, profile: NodeProfile, *, policy_type: str, model: dict[str, Any] | None = None, notes: str = "") -> PolicyBundle:
    """Create a :class:`PolicyBundle` for ``policy``.

    For the built-in rule-based policy the parameters are exported
    automatically; for other policies pass ``model`` explicitly (e.g. the
    weights of a small MLP as nested lists).
    """
    if model is None:
        params = getattr(policy, "p", None)
        model = dataclasses.asdict(params) if dataclasses.is_dataclass(params) else {}
    return PolicyBundle(
        policy_type=policy_type,
        observation_names=[f.name for f in OBSERVATION_FIELDS],
        observation_normalisation=[f.normalisation for f in OBSERVATION_FIELDS],
        profile=dataclasses.asdict(profile),
        action_encoding=action_encoding_spec(profile.n_modes),
        model=model,
        metadata={
            "edgeengine_aware_version": __version__,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "policy_class": type(policy).__name__,
            "notes": notes,
        },
    )
