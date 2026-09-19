"""Action encoding shared by the simulator and the deployment runtime.

The action is a pair ``(sensing_level, transmit)``:

============  =====  =========================================================
component     value  meaning on the hardware
============  =====  =========================================================
sensing_level   0    keep the previous measurement (sensor stays powered off)
                1    low-cost acquisition (single sample, short warm-up)
                2    high-quality acquisition (averaging, longer warm-up)
transmit        0    radio stays off
                1    transmit the latest *stored* measurement
============  =====  =========================================================

Gymnasium represents it as ``MultiDiscrete([3, 2])``; on a microcontroller it
can be two ``uint8`` values or a single flat index in ``[0, 6)`` (see
``flatten_action`` / ``unflatten_action``). The flat index is what a
tabular or DQN-style policy would output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_SENSING_LEVELS = 3
N_TRANSMIT_OPTIONS = 2
N_FLAT_ACTIONS = N_SENSING_LEVELS * N_TRANSMIT_OPTIONS
ACTION_NVEC = (N_SENSING_LEVELS, N_TRANSMIT_OPTIONS)

SENSE_NONE, SENSE_LOW, SENSE_HIGH = 0, 1, 2
TX_NO, TX_YES = 0, 1


@dataclass(frozen=True)
class Action:
    """Decoded, validated action."""

    sensing_level: int
    transmit: int

    def __post_init__(self) -> None:
        if not 0 <= self.sensing_level < N_SENSING_LEVELS:
            raise ValueError(f"sensing_level must be in [0, {N_SENSING_LEVELS}), got {self.sensing_level}")
        if self.transmit not in (0, 1):
            raise ValueError(f"transmit must be 0 or 1, got {self.transmit}")

    def to_array(self) -> np.ndarray:
        return np.array([self.sensing_level, self.transmit], dtype=np.int64)


def decode_action(action) -> Action:
    """Convert a MultiDiscrete array-like (or an ``Action``) into an ``Action``."""
    if isinstance(action, Action):
        return action
    arr = np.asarray(action).reshape(-1)
    if arr.shape[0] != 2:
        raise ValueError(f"expected an action with 2 components, got shape {arr.shape}")
    return Action(int(arr[0]), int(arr[1]))


def encode_action(sensing_level: int, transmit: int) -> np.ndarray:
    """Build the MultiDiscrete array from its two components."""
    return Action(sensing_level, transmit).to_array()


def flatten_action(action) -> int:
    """Map ``(sensing_level, transmit)`` to a single index in ``[0, 6)``.
    ``index = sensing_level * 2 + transmit``."""
    a = decode_action(action)
    return a.sensing_level * N_TRANSMIT_OPTIONS + a.transmit


def unflatten_action(index: int) -> np.ndarray:
    """Inverse of :func:`flatten_action`."""
    if not 0 <= index < N_FLAT_ACTIONS:
        raise ValueError(f"flat action index must be in [0, {N_FLAT_ACTIONS}), got {index}")
    return encode_action(index // N_TRANSMIT_OPTIONS, index % N_TRANSMIT_OPTIONS)


def describe_action(action) -> str:
    a = decode_action(action)
    sense = ("no sensing", "low-cost sensing", "high-quality sensing")[a.sensing_level]
    tx = "transmit" if a.transmit else "no transmission"
    return f"{sense} / {tx}"


# ---------------------------------------------------------------------------
# Energy-feasibility rule (shared by simulator and firmware loop)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionPlan:
    """Which parts of a requested action the node will actually execute."""

    sensing_level: int
    """Executed sensing level (0 if rejected or not requested)."""

    transmit: bool
    """Whether a transmission will be executed."""

    sensing_energy_j: float
    tx_energy_j: float
    rejected: tuple[str, ...]
    """Reasons of rejected sub-actions, e.g. 'sensing:insufficient_energy'."""


def plan_execution(
    action,
    *,
    stored_energy_j: float,
    baseline_energy_j: float,
    reserve_energy_j: float,
    sensing_energy_j: tuple[float, ...],
    tx_energy_j: float,
    has_measurement: bool,
) -> ExecutionPlan:
    """Apply the 'execute the feasible subset' rule.

    The node first sets aside the baseline consumption of the coming interval
    and the brown-out reserve; sensing is served first, then transmission.
    A transmission with nothing to send (no stored measurement and no sensing
    in this step) is rejected without spending energy.
    """
    a = decode_action(action)
    available = stored_energy_j - baseline_energy_j - reserve_energy_j
    rejected: list[str] = []

    level = a.sensing_level
    sense_j = 0.0
    if level != SENSE_NONE:
        cost = sensing_energy_j[level]
        if cost <= available:
            sense_j = cost
            available -= cost
        else:
            rejected.append("sensing:insufficient_energy")
            level = SENSE_NONE

    transmit = False
    tx_j = 0.0
    if a.transmit == TX_YES:
        if not has_measurement and level == SENSE_NONE:
            rejected.append("transmit:no_measurement")
        elif tx_energy_j <= available:
            transmit = True
            tx_j = tx_energy_j
        else:
            rejected.append("transmit:insufficient_energy")

    return ExecutionPlan(level, transmit, sense_j, tx_j, tuple(rejected))
