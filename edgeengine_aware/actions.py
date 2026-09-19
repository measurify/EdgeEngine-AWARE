"""Action encoding shared by the simulator and the deployment runtime.

The action is a pair ``(sensing_level, transmit)``:

============  =====  =========================================================
component     value  meaning on the hardware
============  =====  =========================================================
sensing_level   0    keep the previous measurement (sensor stays powered off)
                1    low-cost acquisition (single sample, short warm-up)
                2    high-quality acquisition (averaging, longer warm-up)
transmit        0    radio stays off
                k    transmit the latest *stored* measurement with radio mode
                     ``k - 1`` (k = 1..n_modes; default modes: 1 fast, 2 standard,
                     3 robust)
============  =====  =========================================================

Gymnasium represents it as ``MultiDiscrete([3, 1 + n_modes])``; on a
microcontroller it is two ``uint8`` values or a single flat index
``sensing_level * (1 + n_modes) + transmit`` (see ``flatten_action``). With a
single radio mode the encoding reduces to the binary ``MultiDiscrete([3, 2])``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_SENSING_LEVELS = 3
DEFAULT_N_MODES = 3
N_TRANSMIT_OPTIONS = 1 + DEFAULT_N_MODES
N_FLAT_ACTIONS = N_SENSING_LEVELS * N_TRANSMIT_OPTIONS
ACTION_NVEC = (N_SENSING_LEVELS, N_TRANSMIT_OPTIONS)

SENSE_NONE, SENSE_LOW, SENSE_HIGH = 0, 1, 2
TX_NO = 0
TX_YES = 2
"""Default transmit value used by the simple policies: the *standard* mode
(mode index 1) of the default three-mode radio."""


def action_nvec(n_modes: int = DEFAULT_N_MODES) -> tuple[int, int]:
    """``MultiDiscrete`` sizes for a radio with ``n_modes`` transmission modes."""
    return (N_SENSING_LEVELS, 1 + int(n_modes))


def n_flat_actions(n_modes: int = DEFAULT_N_MODES) -> int:
    return N_SENSING_LEVELS * (1 + int(n_modes))


@dataclass(frozen=True)
class Action:
    """Decoded, validated action."""

    sensing_level: int
    transmit: int
    """0 = no transmission, k >= 1 = transmit with radio mode k - 1."""

    def __post_init__(self) -> None:
        if not 0 <= self.sensing_level < N_SENSING_LEVELS:
            raise ValueError(f"sensing_level must be in [0, {N_SENSING_LEVELS}), got {self.sensing_level}")
        if self.transmit < 0:
            raise ValueError(f"transmit must be >= 0, got {self.transmit}")

    @property
    def transmits(self) -> bool:
        return self.transmit > 0

    @property
    def mode(self) -> int:
        """Radio mode index (valid only when ``transmits``)."""
        return self.transmit - 1

    def to_array(self) -> np.ndarray:
        return np.array([self.sensing_level, self.transmit], dtype=np.int64)


def decode_action(action, n_modes: int | None = None) -> Action:
    """Convert a MultiDiscrete array-like (or an ``Action``) into an ``Action``.
    With ``n_modes`` the transmit component is range-checked."""
    a = action if isinstance(action, Action) else None
    if a is None:
        arr = np.asarray(action).reshape(-1)
        if arr.shape[0] != 2:
            raise ValueError(f"expected an action with 2 components, got shape {arr.shape}")
        a = Action(int(arr[0]), int(arr[1]))
    if n_modes is not None and a.transmit > n_modes:
        raise ValueError(f"transmit must be in [0, {n_modes}], got {a.transmit}")
    return a


def encode_action(sensing_level: int, transmit: int) -> np.ndarray:
    """Build the MultiDiscrete array from its two components."""
    return Action(sensing_level, transmit).to_array()


def flatten_action(action, n_modes: int = DEFAULT_N_MODES) -> int:
    """Map ``(sensing_level, transmit)`` to a single index:
    ``index = sensing_level * (1 + n_modes) + transmit``."""
    a = decode_action(action, n_modes)
    return a.sensing_level * (1 + n_modes) + a.transmit


def unflatten_action(index: int, n_modes: int = DEFAULT_N_MODES) -> np.ndarray:
    """Inverse of :func:`flatten_action`."""
    n_tx = 1 + n_modes
    if not 0 <= index < N_SENSING_LEVELS * n_tx:
        raise ValueError(f"flat action index must be in [0, {N_SENSING_LEVELS * n_tx}), got {index}")
    return encode_action(index // n_tx, index % n_tx)


def describe_action(action, mode_names: tuple[str, ...] | None = None) -> str:
    a = decode_action(action)
    sense = ("no sensing", "low-cost sensing", "high-quality sensing")[a.sensing_level]
    if not a.transmits:
        tx = "no transmission"
    elif mode_names is not None and a.mode < len(mode_names):
        tx = f"transmit ({mode_names[a.mode]})"
    else:
        tx = f"transmit (mode {a.mode})"
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

    mode: int
    """Radio mode of the transmission (-1 when none)."""

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
    tx_energy_j: tuple[float, ...] | float,
    has_measurement: bool,
) -> ExecutionPlan:
    """Apply the 'execute the feasible subset' rule.

    The node first sets aside the baseline consumption of the coming interval
    and the brown-out reserve; sensing is served first, then transmission.
    A transmission with nothing to send (no stored measurement and no sensing
    in this step) is rejected without spending energy. ``tx_energy_j`` is the
    energy per radio mode (a scalar is treated as a single-mode radio).
    """
    tx_energies = (float(tx_energy_j),) if np.isscalar(tx_energy_j) else tuple(tx_energy_j)
    a = decode_action(action, n_modes=len(tx_energies))
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
    mode = -1
    if a.transmits:
        cost = tx_energies[a.mode]
        if not has_measurement and level == SENSE_NONE:
            rejected.append("transmit:no_measurement")
        elif cost <= available:
            transmit = True
            tx_j = cost
            mode = a.mode
        else:
            rejected.append("transmit:insufficient_energy")

    return ExecutionPlan(level, transmit, mode, sense_j, tx_j, tuple(rejected))
