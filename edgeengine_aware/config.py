"""Configuration objects for EdgeEngine AWARE.

Every tunable quantity of the simulator lives here, grouped by subsystem, so
that experiments can be set up without touching the environment code.

Units (used consistently across the whole package)
--------------------------------------------------
* time      : seconds [s]
* energy    : joules [J]
* power     : watts [W]  (1 W = 1 J/s)
* moisture  : dimensionless volumetric soil-water content normalised to the
              field capacity, i.e. 1.0 = field capacity, 0.0 = completely dry.
* temperature: degrees Celsius; humidity: relative humidity in [0, 1].

The default values describe a *small* energy-harvesting node: a few-cm^2
solar cell that peaks at a few milliwatts, a ~300 J storage element
(roughly a 25 mAh Li-Po cell or a supercapacitor bank) and a LoRa-class radio
with three selectable modes (fast / standard / robust, 0.3 / 0.6 / 1.2 J per uplink).
On an average day the node harvests ~55 J, its baseline load takes ~17 J, one
high-quality sample + uplink costs 1.2 J: hourly reporting is sustainable on
sunny days but not on cloudy ones, which is exactly the regime in which an
energy-aware policy pays off.
They are illustrative, physically plausible values, not a digital twin of a
specific product; each of them is meant to be replaced by a measurement taken
on the real device (see ``docs/sim_to_real.md``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

HOUR_S = 3600.0
DAY_S = 24.0 * HOUR_S


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------
@dataclass
class TimeConfig:
    """Discrete-time settings of the simulator."""

    timestep_s: float = 15.0 * 60.0
    """Duration of one decision step [s]. Default: 15 minutes."""

    episode_days: float = 7.0
    """Episode length in days. The episode is *truncated* after
    ``episode_days * 86400 / timestep_s`` steps."""

    start_hour: float = 0.0
    """Time of day (hours, 0-24) at which the episode starts."""

    @property
    def max_steps(self) -> int:
        return int(round(self.episode_days * DAY_S / self.timestep_s))


# ---------------------------------------------------------------------------
# Energy storage and MCU
# ---------------------------------------------------------------------------
@dataclass
class EnergyStorageConfig:
    """Finite energy storage element (battery or supercapacitor)."""

    capacity_j: float = 300.0
    """Usable capacity E_max [J] (e.g. a ~25 mAh Li-Po cell or a supercapacitor bank)."""

    initial_soc: float = 0.5
    """Initial state of charge in [0, 1] when ``initial_soc_range`` is None."""

    initial_soc_range: tuple[float, float] | None = None
    """If given, the initial SoC is drawn uniformly from this range at reset."""

    reserve_soc: float = 0.02
    """Fraction of E_max that the node never spends on optional operations
    (sensing / transmission). Below ``reserve_soc`` only the baseline load is
    served; this mimics the brown-out protection of a real power path."""

    charge_efficiency: float = 1.0
    """Fraction of the harvested energy that is actually stored (1 = ideal buffer)."""


@dataclass
class MCUConfig:
    """Always-on consumption of the microcontroller and its peripherals."""

    baseline_power_w: float = 200e-6
    """Average sleep/idle power [W] (e.g. ~60 uA at 3.3 V, including sensor
    quiescent current and RTC)."""


# ---------------------------------------------------------------------------
# Energy harvesting
# ---------------------------------------------------------------------------
@dataclass
class HarvestingConfig:
    """Stochastic solar harvesting model.

    harvested_power(t) = max_power_w * solar_profile(t) * cloud_factor(t) * efficiency
    """

    max_power_w: float = 0.005
    """Peak electrical power of the panel under clear sky at solar noon [W]
    (a ~2 cm^2 cell, or a larger cell under partial canopy shading)."""

    efficiency: float = 0.6
    """Harvesting-circuit (MPPT/charger) efficiency in (0, 1]."""

    sunrise_hour: float = 6.0
    sunset_hour: float = 18.0

    clearness_mean: float = 0.7
    """Mean daily clearness index (1 = perfectly clear day)."""

    clearness_std: float = 0.2
    """Day-to-day standard deviation of the daily clearness index."""

    clearness_autocorr: float = 0.5
    """AR(1) coefficient linking consecutive days (weather persistence)."""

    cloud_noise_std: float = 0.15
    """Std of the intra-day cloud perturbation (per step)."""

    cloud_autocorr: float = 0.85
    """AR(1) coefficient of the intra-day cloud perturbation."""

    measurement_noise_std: float = 0.05
    """Relative noise of the *measured* harvesting power exposed to the node."""


# ---------------------------------------------------------------------------
# Sensing
# ---------------------------------------------------------------------------
@dataclass
class SensingConfig:
    """Sensing modes. Index 0 = no sensing, 1 = low-cost, 2 = high-quality."""

    energy_j: tuple[float, float, float] = (0.0, 0.10, 0.60)
    """Energy per sensing operation for each level [J]."""

    noise_std: tuple[float, float, float] = (0.0, 0.040, 0.010)
    """Std of the additive Gaussian measurement noise for each level
    (moisture units). Level 0 never produces a measurement."""

    bias: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Optional systematic offset per level (moisture units)."""

    level_names: tuple[str, str, str] = ("none", "low", "high")

    @property
    def n_levels(self) -> int:
        return len(self.energy_j)


# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------
@dataclass
class RadioModeConfig:
    """One transmission mode of the radio (a spreading factor / power setting).

    The link budget of a mode is ``tx_power_dbm - path_loss_db - sensitivity_dbm``
    (its *margin*); a mode with a longer spreading factor has a lower (better)
    sensitivity, a longer air time and therefore a higher energy per uplink.
    """

    name: str
    energy_j: float
    """Energy of one uplink attempt in this mode (radio + MCU awake time) [J]."""

    tx_power_dbm: float
    """Transmit power [dBm]."""

    sensitivity_dbm: float
    """Receiver sensitivity at the gateway for this spreading factor [dBm]
    (SX127x-class, 125 kHz: SF7 ~ -123, SF9 ~ -129, SF12 ~ -137)."""


DEFAULT_RADIO_MODES: tuple[RadioModeConfig, ...] = (
    RadioModeConfig("fast", energy_j=0.30, tx_power_dbm=14.0, sensitivity_dbm=-123.0),  # SF7-like
    RadioModeConfig("standard", energy_j=0.60, tx_power_dbm=14.0, sensitivity_dbm=-129.0),  # SF9-like
    RadioModeConfig("robust", energy_j=1.20, tx_power_dbm=14.0, sensitivity_dbm=-137.0),  # SF12-like
)


@dataclass
class CommunicationConfig:
    """Abstract low-power long-range link (LoRa-like) with a link-budget channel.

    delivery probability of mode k at time t:
        margin_k(t) = tx_power_k - path_loss(t) - sensitivity_k
        p_k(t)      = 1 / (1 + exp(-margin_k(t) / margin_scale_db))
        path_loss(t) = path_loss_mean_db + slow_fading(t) + fast_fading   (dB)

    ``slow_fading`` is an AR(1) process (shadowing, vegetation, humidity);
    ``fast_fading`` is redrawn at every attempt. With the defaults the mean
    margins are +4 dB (*standard*), -2 dB (*fast*, half the energy) and +12 dB
    (*robust*, twice the energy). Averaged over the fading, the long-run
    delivery ratios are about 0.75 / 0.36 / 0.98; at zero slow fading they are
    0.93 / 0.21 / 1.0, so the fast mode pays off only in favourable phases.
    """

    modes: tuple[RadioModeConfig, ...] = DEFAULT_RADIO_MODES
    """Selectable transmission modes; the action ``transmit = k`` (k >= 1) uses
    ``modes[k - 1]``. A single-entry tuple reproduces a binary transmit action."""

    reference_mode: int = 1
    """Index of the mode whose energy is reported in the observation and used
    by the simple policies when they have no link estimate."""

    path_loss_mean_db: float = 139.0
    slow_fading_std_db: float = 5.0
    slow_fading_autocorr: float = 0.97
    """AR(1) coefficient per step (0.97 at 15 min ~ 8 h correlation time)."""

    fast_fading_std_db: float = 2.0
    margin_scale_db: float = 1.5
    """Softness of the delivery curve around zero margin."""

    ack_available: bool = True
    """Whether the node learns if an uplink was delivered (confirmed uplink /
    ACK). With ``False`` (unconfirmed uplinks) the node assumes every uplink
    was delivered: its estimate of the information age at the application
    becomes optimistic and its link-quality indicator stays at 1."""

    ack_margin_noise_db: float = 1.0
    """Std of the noise on the link margin the node measures from an ACK
    (SNR estimate), i.e. on its path-loss estimate."""

    priority_update_mode: str = "immediate"
    """How the node learns the application priority.
    'immediate' : the node always knows the current priority (e.g. a listening
                  downlink window or a class-C style device).
    'on_uplink' : the priority is refreshed only after a *successful* uplink,
                  as in a LoRaWAN class-A downlink piggybacked on the ACK."""

    @property
    def n_modes(self) -> int:
        return len(self.modes)

    @property
    def tx_energy_j(self) -> float:
        """Energy of the reference mode [J] (backwards-compatible accessor)."""
        return self.modes[self.reference_mode].energy_j

    def mean_margin_db(self, mode: int) -> float:
        m = self.modes[mode]
        return m.tx_power_dbm - self.path_loss_mean_db - m.sensitivity_dbm


def single_mode_radio(energy_j: float = 0.60, mean_margin_db: float = 4.0) -> CommunicationConfig:
    """A CommunicationConfig with one transmission mode (binary transmit action)."""
    mode = RadioModeConfig("standard", energy_j=energy_j, tx_power_dbm=14.0, sensitivity_dbm=-129.0)
    return CommunicationConfig(modes=(mode,), reference_mode=0, path_loss_mean_db=14.0 + 129.0 - mean_margin_db)


# ---------------------------------------------------------------------------
# Agriculture ground-truth environment
# ---------------------------------------------------------------------------
@dataclass
class AgricultureConfig:
    """Stochastic soil/atmosphere model (hidden ground truth)."""

    initial_moisture: float = 0.55
    initial_moisture_range: tuple[float, float] | None = (0.40, 0.70)

    warning_threshold: float = 0.35
    """Soil moisture below which the crop starts to experience water stress."""

    critical_threshold: float = 0.25
    """Soil moisture below which irrigation is urgently needed."""

    et_rate_per_day: float = 0.06
    """Mean evapotranspiration [moisture units / day] at the reference
    temperature (the day/night modulation preserves this daily mean)."""

    et_temp_coeff: float = 0.03
    """Relative increase of ET per degree above ``temp_mean``."""

    et_diurnal_amplitude: float = 0.8
    """Fraction of ET modulated by the day/night cycle (0 = flat)."""

    sunrise_hour: float = 6.0
    sunset_hour: float = 18.0
    """Daylight window used by the ET modulation (keep consistent with HarvestingConfig)."""

    rain_events_per_day: float = 0.25
    """Rate of the Poisson process generating rain events."""

    rain_amount_range: tuple[float, float] = (0.05, 0.25)
    """Moisture added by a rain event (uniform)."""

    irrigation_enabled: bool = True
    irrigation_trigger: float = 0.20
    """Ground-truth moisture below which the (external) irrigation system
    eventually reacts. This models the field being irrigated by a farmer; it is
    *not* a decision of the node."""

    irrigation_delay_mean_s: float = 6.0 * HOUR_S
    irrigation_amount: float = 0.30

    process_noise_std: float = 0.003
    """Std of the per-step random walk on moisture."""

    max_moisture: float = 1.0

    temp_mean_c: float = 22.0
    temp_amplitude_c: float = 7.0
    temp_peak_hour: float = 15.0
    temp_noise_std: float = 0.6
    temp_autocorr: float = 0.9

    humidity_mean: float = 0.60
    humidity_temp_coeff: float = -0.02
    """Change of relative humidity per degree of temperature deviation."""
    humidity_noise_std: float = 0.03

    event_change_threshold: float = 0.05
    """A change of true moisture larger than this within one step (rain,
    irrigation) is registered as an 'environmental event'."""


# ---------------------------------------------------------------------------
# Remote application (utility + priority)
# ---------------------------------------------------------------------------
@dataclass
class ApplicationConfig:
    """Information utility and interest model of the remote application.

    See ``application.py`` for the formulas.
    """

    tracking_weight: float = 0.10
    """Per-step weight of the tracking utility (max 0.1 * (1 + criticality_gain)
    per step when the application's picture is perfect)."""

    error_scale: float = 0.05
    """Moisture error at which the accuracy factor exp(-err/scale) drops to 1/e."""

    tau_freshness_s: float = 2.0 * HOUR_S
    """Time constant discounting the packet bonus with the age of the
    measurement it carries (sending old samples is worth less)."""

    gain_scale: float = 0.05
    """Reduction of the application's estimation error giving tanh(1) ~ 76%
    of the gain bonus (kept large w.r.t. sensor noise so that re-sampling noise
    adds little reward variance)."""

    w_gain: float = 0.30
    """Weight of the (signed) estimation-error gain bonus per delivered packet."""

    w_event: float = 0.50
    """Weight of the bonus for reporting an environmental event the
    application does not know about yet."""

    criticality_gain: float = 2.0
    """Extra weight of information when the true moisture is at a threshold."""

    criticality_scale: float = 0.08
    """Moisture distance to the nearest threshold over which the extra
    weight decays."""

    event_memory_s: float = 3.0 * HOUR_S
    """An environmental event remains 'unreported' (and worth reporting) for
    at most this long."""

    aoi_elevated_s: float = 8.0 * HOUR_S
    """Age of information after which the application raises priority to 1."""

    aoi_urgent_s: float = 24.0 * HOUR_S
    """Age of information after which the application raises priority to 2."""

    request_rate_per_day: float = 0.2
    """Rate of external 'high-resolution monitoring' requests (Poisson)."""

    request_duration_range_s: tuple[float, float] = (2.0 * HOUR_S, 6.0 * HOUR_S)
    request_urgent_fraction: float = 0.3
    """Fraction of external requests that are urgent (priority 2) instead of
    elevated (priority 1)."""


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------
@dataclass
class RewardConfig:
    """Weights of the reward components (all dimensionless).

    reward = utility
             - lambda_sense * E_sense / energy_ref_j
             - lambda_tx    * E_tx    / energy_ref_j
             - lambda_stale * w_priority * min(AoI / tau_stale_s, 1)
             - lambda_battery * battery_risk
             - lambda_depletion * [battery depleted]
             - lambda_reject * [action rejected for lack of energy]
             - lambda_waste * wasted_harvest / energy_ref_j
    """

    lambda_sense: float = 0.10
    lambda_tx: float = 0.10
    lambda_stale: float = 0.05
    lambda_battery: float = 0.50
    lambda_depletion: float = 2.0
    lambda_reject: float = 0.20
    lambda_waste: float = 0.02

    energy_ref_j: float = 1.0
    """Energy normalisation constant [J] (about one transmission)."""

    tau_stale_s: float = 6.0 * HOUR_S
    priority_weights: tuple[float, float, float] = (1.0, 2.0, 4.0)
    """Multiplier of the staleness penalty per application priority."""

    safe_soc: float = 0.30
    """Below this SoC a quadratic battery-risk penalty is applied."""


# ---------------------------------------------------------------------------
# Observation normalisation
# ---------------------------------------------------------------------------
@dataclass
class ObservationConfig:
    """Constants used by the ObservationBuilder (shared with deployment)."""

    harvest_ref_power_w: float = 0.003
    """Power that maps to 1.0 in the harvesting observations."""

    age_scale_s: float = 24.0 * HOUR_S
    """Time that maps to 1.0 in the age / time-since observations."""

    harvest_ewma_alpha: float = 0.2
    """Weight of the newest sample in the recent-harvest EWMA."""

    link_ewma_alpha: float = 0.2
    """Weight of the newest ACK outcome in the link-quality EWMA."""

    importance_scale: float = 0.10
    """Distance (moisture units) to the nearest threshold over which the
    node-side importance indicator decays."""

    path_loss_min_db: float = 110.0
    path_loss_max_db: float = 170.0
    """Range mapped to [0, 1] in the path-loss observation (1 = unknown / worst)."""


# ---------------------------------------------------------------------------
# Domain randomisation
# ---------------------------------------------------------------------------
@dataclass
class DomainRandomizationConfig:
    """Multiplicative ranges applied to physical parameters at every reset.

    Each entry is ``(low, high)``: the nominal value is multiplied by a factor
    drawn uniformly in that interval. ``enabled=False`` disables everything.
    """

    enabled: bool = False
    sensor_noise: tuple[float, float] = (0.7, 1.5)
    sensing_energy: tuple[float, float] = (0.8, 1.3)
    tx_energy: tuple[float, float] = (0.8, 1.3)
    path_loss_db: tuple[float, float] = (-3.0, 3.0)
    """Additive offset [dB] on the mean path loss (this one is additive, not multiplicative)."""
    solar_intensity: tuple[float, float] = (0.6, 1.2)
    cloud_variability: tuple[float, float] = (0.5, 1.5)
    battery_capacity: tuple[float, float] = (0.8, 1.2)
    baseline_power: tuple[float, float] = (0.7, 1.5)


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------
@dataclass
class EdgeEngineAwareConfig:
    """Complete configuration of an EdgeEngine AWARE environment."""

    time: TimeConfig = field(default_factory=TimeConfig)
    storage: EnergyStorageConfig = field(default_factory=EnergyStorageConfig)
    mcu: MCUConfig = field(default_factory=MCUConfig)
    harvesting: HarvestingConfig = field(default_factory=HarvestingConfig)
    sensing: SensingConfig = field(default_factory=SensingConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    agriculture: AgricultureConfig = field(default_factory=AgricultureConfig)
    application: ApplicationConfig = field(default_factory=ApplicationConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    randomization: DomainRandomizationConfig = field(default_factory=DomainRandomizationConfig)

    terminate_on_depletion: bool = False
    """If True the episode *terminates* when the storage is fully depleted.
    Default False: the node browns out, pays a penalty and keeps running once
    energy is harvested again (closer to what happens in the field)."""

    def copy(self) -> "EdgeEngineAwareConfig":
        """Deep copy (configs are small; ``dataclasses.replace`` is shallow)."""
        import copy as _copy

        return _copy.deepcopy(self)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def validate(self) -> None:
        """Raise ``ValueError`` on obviously inconsistent settings."""
        if self.time.timestep_s <= 0 or self.time.episode_days <= 0:
            raise ValueError("timestep_s and episode_days must be positive")
        if self.storage.capacity_j <= 0:
            raise ValueError("capacity_j must be positive")
        if not 0.0 <= self.storage.reserve_soc < 1.0:
            raise ValueError("reserve_soc must be in [0, 1)")
        if not (len(self.sensing.energy_j) == len(self.sensing.noise_std) == len(self.sensing.bias) == 3):
            raise ValueError("sensing energy_j, noise_std and bias must have 3 entries (none / low / high)")
        if not 0.0 <= self.time.start_hour < 24.0:
            raise ValueError("start_hour must be in [0, 24)")
        if not 0.0 < self.storage.charge_efficiency <= 1.0:
            raise ValueError("charge_efficiency must be in (0, 1]")
        if self.communication.margin_scale_db <= 0:
            raise ValueError("margin_scale_db must be positive")
        if self.sensing.energy_j[0] != 0.0:
            raise ValueError("sensing level 0 (no sensing) must have zero energy cost")
        if self.communication.n_modes < 1:
            raise ValueError("at least one radio mode is required")
        if not 0 <= self.communication.reference_mode < self.communication.n_modes:
            raise ValueError("reference_mode must index an existing radio mode")
        if any(m.energy_j <= 0 for m in self.communication.modes):
            raise ValueError("radio mode energies must be positive")
        if self.communication.priority_update_mode not in ("immediate", "on_uplink"):
            raise ValueError("priority_update_mode must be 'immediate' or 'on_uplink'")
        if not self.agriculture.critical_threshold < self.agriculture.warning_threshold:
            raise ValueError("critical_threshold must be below warning_threshold")
        if not 0.0 < self.harvesting.efficiency <= 1.0:
            raise ValueError("harvesting efficiency must be in (0, 1]")


def default_config() -> EdgeEngineAwareConfig:
    """Return a fresh configuration with all default values."""
    return EdgeEngineAwareConfig()


def randomize_config(base: EdgeEngineAwareConfig, rng) -> EdgeEngineAwareConfig:
    """Return a copy of ``base`` with physical parameters perturbed according
    to ``base.randomization`` (simple multiplicative domain randomisation).

    The returned configuration is what the *node* experiences during the
    episode: e.g. the (randomised) energy costs are the ones the node reports
    in its observation, exactly like a real device would report its own
    measured hardware profile.
    """
    cfg = base.copy()
    dr = cfg.randomization
    if not dr.enabled:
        return cfg

    def f(rng_range: tuple[float, float]) -> float:
        return float(rng.uniform(rng_range[0], rng_range[1]))

    k = f(dr.sensor_noise)
    cfg.sensing.noise_std = tuple(s * k for s in cfg.sensing.noise_std)  # type: ignore[assignment]
    k = f(dr.sensing_energy)
    cfg.sensing.energy_j = tuple(e * k for e in cfg.sensing.energy_j)  # type: ignore[assignment]
    k = f(dr.tx_energy)
    cfg.communication.modes = tuple(dataclasses.replace(m, energy_j=m.energy_j * k) for m in cfg.communication.modes)
    cfg.communication.path_loss_mean_db += f(dr.path_loss_db)
    cfg.harvesting.max_power_w *= f(dr.solar_intensity)
    k = f(dr.cloud_variability)
    cfg.harvesting.cloud_noise_std *= k
    cfg.harvesting.clearness_std *= k
    cfg.storage.capacity_j *= f(dr.battery_capacity)
    cfg.mcu.baseline_power_w *= f(dr.baseline_power)
    return cfg
