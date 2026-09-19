"""Reward computation with explicit, separately reported components.

    reward = application_utility
             - sensing_cost - communication_cost
             - staleness_penalty - battery_penalty - depletion_penalty
             - rejection_penalty - waste_penalty

Scales (defaults, 7-day episode = 672 steps): the tracking utility is worth up
to 0.1 * criticality (1..3) per step, i.e. ~100-120 per episode for a policy
that keeps the application well informed; one high-quality sample + uplink
costs 0.12; the staleness penalty saturates at lambda_stale * w_priority
(0.05 / 0.10 / 0.20 per step) after tau_stale_s = 6 h - a policy that never
transmits loses ~120 per episode to it; the battery penalty grows
quadratically below ``safe_soc`` up to lambda_battery (0.5) per step at
SoC = 0, plus lambda_depletion (2.0) at every step in which the node actually
browns out. Energy penalties alone (~5 for hourly reporting) do not dominate:
what makes energy binding is the battery penalty, which an always-on policy
pays to the tune of several hundred per episode.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import RewardConfig


@dataclass
class RewardComponents:
    application_utility: float = 0.0
    sensing_cost: float = 0.0
    communication_cost: float = 0.0
    staleness_penalty: float = 0.0
    battery_penalty: float = 0.0
    depletion_penalty: float = 0.0
    rejection_penalty: float = 0.0
    waste_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.application_utility
            - self.sensing_cost
            - self.communication_cost
            - self.staleness_penalty
            - self.battery_penalty
            - self.depletion_penalty
            - self.rejection_penalty
            - self.waste_penalty
        )

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["total"] = self.total
        return d


class RewardCalculator:
    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    def compute(
        self,
        *,
        utility: float,
        sensing_energy_j: float,
        communication_energy_j: float,
        aoi_s: float,
        priority: int,
        soc_after: float,
        depleted: bool,
        n_rejected: int,
        wasted_energy_j: float,
    ) -> RewardComponents:
        c = self.cfg
        w_prio = c.priority_weights[int(priority)]
        staleness = c.lambda_stale * w_prio * min(aoi_s / c.tau_stale_s, 1.0)
        if soc_after < c.safe_soc:
            risk = ((c.safe_soc - soc_after) / c.safe_soc) ** 2
        else:
            risk = 0.0
        return RewardComponents(
            application_utility=utility,
            sensing_cost=c.lambda_sense * sensing_energy_j / c.energy_ref_j,
            communication_cost=c.lambda_tx * communication_energy_j / c.energy_ref_j,
            staleness_penalty=staleness,
            battery_penalty=c.lambda_battery * risk,
            depletion_penalty=c.lambda_depletion if depleted else 0.0,
            rejection_penalty=c.lambda_reject * n_rejected,
            waste_penalty=c.lambda_waste * wasted_energy_j / c.energy_ref_j,
        )
