"""Hardware abstraction layer of EdgeEngine AWARE.

The policy never talks to these objects directly; the *node controller*
(simulated by ``env.py``, or a real firmware loop as sketched in
``deployment.py``) does. Each protocol is deliberately tiny so that it can be
implemented by

* a stochastic model (this package: ``energy.py``, ``sensing.py``, ...),
* a driver on a microcontroller (ADC, fuel gauge, sensor driver, radio stack),
* a replay of recorded traces (future: real sensor / harvesting traces).

Only *measurable* quantities cross these interfaces. Hidden ground truth
(true soil moisture, future irradiance, ...) stays inside the simulated
implementations and is only exposed through explicit ``ground_truth()`` style
accessors used for reward, evaluation and rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Data records exchanged between subsystems
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Measurement:
    """A value acquired by the node from a sensor."""

    value: float
    """Measured quantity (moisture units, normalised to field capacity)."""

    timestamp_s: float
    """Node clock time at which the measurement was taken [s]."""

    level: int
    """Sensing level that produced it (1 = low-cost, 2 = high-quality)."""

    noise_std: float
    """Nominal standard deviation of the measurement error (from the sensor
    datasheet / hardware profile). Used as a *quality tag*, it is not the
    realised error."""


@dataclass(frozen=True)
class Packet:
    """Payload of one uplink: the latest measurement plus its age."""

    measurement: Measurement
    sent_at_s: float

    @property
    def measurement_age_s(self) -> float:
        return self.sent_at_s - self.measurement.timestamp_s


# ---------------------------------------------------------------------------
# Hardware-facing protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class Clock(Protocol):
    """Time source of the node (RTC or monotonic timer)."""

    def now_s(self) -> float: ...

    def time_of_day_s(self) -> float:
        """Seconds since local midnight, in [0, 86400)."""
        ...


@runtime_checkable
class EnergyStorage(Protocol):
    """Battery / supercapacitor with a fuel gauge."""

    def capacity_j(self) -> float: ...

    def energy_j(self) -> float:
        """Currently stored energy [J] as reported by the gauge."""
        ...


@runtime_checkable
class EnergySource(Protocol):
    """Harvesting subsystem with a power/current monitor."""

    def measured_power_w(self) -> float:
        """Average harvesting power over the interval that just elapsed [W],
        i.e. the energy integrated by the harvester monitor since the
        previous wake-up divided by the interval. Never a forecast."""
        ...


@runtime_checkable
class Sensor(Protocol):
    """Environmental sensor with selectable acquisition modes."""

    def read(self, level: int, now_s: float) -> Measurement:
        """Acquire a measurement at the requested level (>= 1)."""
        ...

    def energy_cost_j(self, level: int) -> float:
        """Nominal energy of an acquisition at ``level`` [J]."""
        ...

    def noise_std(self, level: int) -> float:
        """Nominal measurement noise at ``level`` (quality tag)."""
        ...


@runtime_checkable
class Radio(Protocol):
    """Low-power long-range uplink."""

    def transmit(self, packet: Packet) -> bool:
        """Send a packet. Returns True if an acknowledgement was received
        (delivery confirmed). Energy is consumed regardless of the outcome.
        On a link without confirmations the return value is meaningless and
        the controller ignores it (``NodeProfile.ack_available``)."""
        ...

    def tx_energy_j(self) -> float:
        """Nominal energy of one transmission attempt [J]."""
        ...


@runtime_checkable
class RemoteApplication(Protocol):
    """Application-side endpoint as seen from the node.

    On a real system the node only sees the *priority* the application sends
    back (downlink). Delivery of data happens through the ``Radio``."""

    def priority(self) -> int:
        """Current information priority requested by the application:
        0 = routine, 1 = elevated, 2 = urgent."""
        ...


# ---------------------------------------------------------------------------
# Policy protocol (independent from Gymnasium)
# ---------------------------------------------------------------------------
@runtime_checkable
class Policy(Protocol):
    """Anything that maps a policy observation to an action.

    ``observation`` is the float32 vector produced by
    ``observation.ObservationBuilder``; the returned action is the
    ``(sensing_level, transmit)`` pair encoded as in ``actions.py``.
    """

    def act(self, observation: Any) -> Any: ...

    def reset(self) -> None:
        """Clear any internal state at the start of an episode."""
        ...
