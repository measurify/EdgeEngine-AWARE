"""Episode-level metrics and a lightweight step logger.

``EpisodeMetrics`` is updated by the environment at every step and returned
in ``info["metrics"]`` (as a dict) on every step, so it is always available -
also when an episode is cut short. ``EpisodeLog`` stores per-step arrays for
plotting (used by the renderer and the notebook).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class EpisodeMetrics:
    steps: int = 0
    total_harvested_energy_j: float = 0.0
    total_consumed_energy_j: float = 0.0
    baseline_energy_j: float = 0.0
    sensing_energy_j: float = 0.0
    communication_energy_j: float = 0.0
    wasted_harvest_energy_j: float = 0.0
    n_sensing: int = 0
    n_high_quality_sensing: int = 0
    n_transmissions: int = 0
    n_successful_transmissions: int = 0
    transmissions_per_mode: dict[int, int] = field(default_factory=dict)
    deliveries_per_mode: dict[int, int] = field(default_factory=dict)
    n_rejected_actions: int = 0
    battery_depletion_events: int = 0
    steps_low_battery: int = 0
    _soc_sum: float = 0.0
    min_battery_soc: float = 1.0
    _aoi_sum_s: float = 0.0
    max_aoi_s: float = 0.0
    total_application_utility: float = 0.0
    total_reward: float = 0.0
    reward_components: dict[str, float] = field(default_factory=dict)

    # -- derived --------------------------------------------------------------
    @property
    def average_battery_soc(self) -> float:
        return self._soc_sum / self.steps if self.steps else 0.0

    @property
    def average_aoi_s(self) -> float:
        return self._aoi_sum_s / self.steps if self.steps else 0.0

    @property
    def fraction_low_battery(self) -> float:
        return self.steps_low_battery / self.steps if self.steps else 0.0

    @property
    def delivery_ratio(self) -> float:
        return self.n_successful_transmissions / self.n_transmissions if self.n_transmissions else 0.0

    def as_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d.update(
            average_battery_soc=self.average_battery_soc,
            average_aoi_s=self.average_aoi_s,
            average_aoi_h=self.average_aoi_s / 3600.0,
            max_aoi_h=self.max_aoi_s / 3600.0,
            fraction_low_battery=self.fraction_low_battery,
            delivery_ratio=self.delivery_ratio,
        )
        return d

    def summary(self) -> str:
        d = self.as_dict()
        keys = [
            ("total_reward", "{:.2f}"),
            ("total_application_utility", "{:.2f}"),
            ("total_harvested_energy_j", "{:.1f} J"),
            ("total_consumed_energy_j", "{:.1f} J"),
            ("baseline_energy_j", "{:.1f} J"),
            ("sensing_energy_j", "{:.1f} J"),
            ("communication_energy_j", "{:.1f} J"),
            ("wasted_harvest_energy_j", "{:.1f} J"),
            ("n_sensing", "{}"),
            ("n_high_quality_sensing", "{}"),
            ("n_transmissions", "{}"),
            ("n_successful_transmissions", "{}"),
            ("transmissions_per_mode", "{}"),
            ("delivery_ratio", "{:.2f}"),
            ("n_rejected_actions", "{}"),
            ("average_battery_soc", "{:.3f}"),
            ("min_battery_soc", "{:.3f}"),
            ("fraction_low_battery", "{:.3f}"),
            ("battery_depletion_events", "{}"),
            ("average_aoi_h", "{:.2f} h"),
            ("max_aoi_h", "{:.2f} h"),
        ]
        width = max(len(k) for k, _ in keys)
        return "\n".join(f"{k:<{width}} : {fmt.format(d[k])}" for k, fmt in keys)


@dataclass
class EpisodeLog:
    """Per-step history for plotting."""

    time_s: list[float] = field(default_factory=list)
    soc: list[float] = field(default_factory=list)
    harvest_power_w: list[float] = field(default_factory=list)
    harvested_energy_j: list[float] = field(default_factory=list)
    true_moisture: list[float] = field(default_factory=list)
    measured_moisture: list[float] = field(default_factory=list)  # nan when no measurement stored
    app_moisture: list[float] = field(default_factory=list)  # nan when nothing received yet
    sensing_level: list[int] = field(default_factory=list)
    tx_attempt: list[int] = field(default_factory=list)
    tx_mode: list[int] = field(default_factory=list)  # -1 when no transmission
    tx_success: list[int] = field(default_factory=list)
    path_loss_db: list[float] = field(default_factory=list)
    aoi_s: list[float] = field(default_factory=list)
    priority: list[int] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    utility: list[float] = field(default_factory=list)
    temperature_c: list[float] = field(default_factory=list)
    humidity: list[float] = field(default_factory=list)

    def append(self, **kwargs) -> None:
        for k, v in kwargs.items():
            getattr(self, k).append(v)

    def __len__(self) -> int:
        return len(self.time_s)
