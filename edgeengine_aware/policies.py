"""Policies implementing the Gymnasium-independent ``Policy`` protocol.

All policies here consume the *normalised observation vector* produced by
``ObservationBuilder`` and return a MultiDiscrete action array. Because they
only use the observation (never the environment object), the very same code
can drive the simulator or the deployment runtime in ``deployment.py``, and
the rule-based policy translates line by line into C on a microcontroller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .actions import DEFAULT_N_MODES, SENSE_HIGH, SENSE_LOW, SENSE_NONE, TX_NO, TX_YES, encode_action, n_flat_actions, unflatten_action
from .observation import OBSERVATION_FIELDS, NodeProfile

_IDX = {f.name: i for i, f in enumerate(OBSERVATION_FIELDS)}


@dataclass
class RuleBasedParams:
    """Thresholds of the interpretable baseline (observation units unless noted)."""

    soc_critical: float = 0.20
    """Below this SoC: deep economy - one report every ``deep_eco_interval_h``,
    nothing else, unless the application is urgent and starving."""

    soc_low: float = 0.50
    """Below this SoC: economy mode (report interval stretched, no checks).
    Half of the storage is the trigger because a 300 J buffer covers only a
    few cloudy days: waiting for the battery to be nearly empty is too late."""

    deep_eco_interval_h: float = 8.0
    """Report interval below ``soc_critical`` (still high-quality samples: a
    cheap noisy sample is worth little to the application)."""

    soc_high: float = 0.70
    """Above this SoC (or under strong harvesting): generous mode."""

    harvest_strong: float = 0.5
    """Normalised recent-harvest level considered 'strong sun'."""

    report_interval_h: tuple[float, float, float] = (2.0, 1.0, 0.5)
    """Scheduled report interval per priority (routine / elevated / urgent)."""

    report_interval_eco_factor: float = 2.0
    """Economy mode stretches the report interval by this factor."""

    eco_sensing_level: int = 2
    """Sensing level used for reports in economy mode (2 = keep high quality)."""

    report_interval_generous_factor: float = 0.75
    """Generous mode shrinks the routine report interval by this factor."""

    check_interval_h: float = 1.0
    """Between reports, take a cheap low-cost sample this often to detect
    sudden changes (disabled in economy mode)."""

    event_delta: float = 0.08
    """A low-cost check deviating from the reported value by more than this
    triggers an immediate high-quality report (2 sigma of the low-cost noise)."""

    importance_immediate: float = 0.7
    """Report at once when the node-side importance exceeds this and the
    stored value differs from the reported one by more than ``importance_delta``."""

    importance_delta: float = 0.03

    retry_age_h: float = 0.3
    """A high-quality sample younger than this that has not been acknowledged
    is retransmitted."""

    age_scale_h: float = 24.0
    """Must equal ObservationConfig.age_scale_s / 3600 (de-normalisation of ages)."""

    link_margin_target_db: float = 4.0
    """Radio mode choice: the cheapest mode whose expected margin (from the
    node's path-loss estimate) is at least this is used; the most robust mode
    when none qualifies; the reference mode when there is no estimate yet."""

    link_quality_escalate: float = 0.7
    """Below this ACK-EWMA the node escalates one mode (recent failures)."""

    retry_min_link_quality: float = 0.5
    """Immediate retries are suppressed below this ACK-EWMA (link down: back off
    to the scheduled reports instead of burning energy every step)."""


class RuleBasedPolicy:
    """Interpretable heuristic controller.

    The action space lets the node sense *and* transmit in the same step, and
    the transmission always carries the freshest stored sample - so a
    "scheduled report" is one action ``(SENSE_HIGH, TX_YES)``: wake up, take
    a good sample, send it. Between reports the node may take cheap
    low-cost *checks* without transmitting; if a check reveals a large change
    the node confirms it with a high-quality sample and reports immediately.

    Rules, in order:
      1. **Deep economy** - SoC below ``soc_critical``: one high-quality report
         every ``deep_eco_interval_h`` (urgent priority: every urgent interval),
         nothing else.
      2. **Retry** - a fresh high-quality sample that was not acknowledged is
         retransmitted (no new sensing), unless the link looks down.
      3. **Scheduled report** - when the estimated information age at the
         application exceeds the report interval (shorter under higher
         priority, stretched in economy mode, shrunk in generous mode):
         high-quality sample + transmit. Economy mode (SoC below ``soc_low``)
         keeps the sample quality and saves energy by reporting less often and
         by skipping the checks - a cheap noisy sample is worth little to the
         application, a missed hour is cheap.
      4. **Event report** - the stored check differs from the reported value
         by more than ``event_delta``, or the stored value is important
         (near/below a threshold) and differs by more than ``importance_delta``:
         high-quality sample + transmit.
      5. **Check** - outside economy mode, a low-cost sample every
         ``check_interval_h`` (no transmission).
      6. Otherwise sleep.

    **Radio mode** (when the profile has several): the cheapest mode whose
    expected margin, computed from the node's path-loss estimate and the
    flash link-budget table, is at least ``link_margin_target_db``; one mode
    up when recent uplinks failed; the reference mode before the first
    estimate. Without a profile the policy always uses the default mode.
    """

    def __init__(self, params: RuleBasedParams | None = None, profile: "NodeProfile | None" = None):
        self.p = params or RuleBasedParams()
        self.profile = profile
        if profile is not None:  # keep the de-normalisation constant in sync with the node profile
            self.p.age_scale_h = profile.observation.age_scale_s / 3600.0

    def _tx(self, o: np.ndarray) -> int:
        """Transmit action value: 1 + chosen radio mode."""
        prof = self.profile
        if prof is None or prof.n_modes == 1:
            return TX_YES if prof is None else 1 + prof.reference_mode
        oc = prof.observation
        pl_norm = float(o[_IDX["path_loss_est"]])
        if pl_norm >= 1.0:  # no estimate yet
            return 1 + prof.reference_mode
        pl_db = oc.path_loss_min_db + pl_norm * (oc.path_loss_max_db - oc.path_loss_min_db)
        order = sorted(range(prof.n_modes), key=lambda k: prof.tx_energy_j[k])  # cheapest first
        chosen = order[-1]
        for k in order:
            if prof.margin_for_mode(k, pl_db) >= self.p.link_margin_target_db:
                chosen = k
                break
        if float(o[_IDX["link_quality"]]) < self.p.link_quality_escalate:
            pos = order.index(chosen)
            chosen = order[min(pos + 1, len(order) - 1)]
        return 1 + chosen

    def reset(self) -> None:  # stateless
        pass

    def act(self, observation) -> np.ndarray:
        o = np.asarray(observation, dtype=np.float32)
        p = self.p
        soc = float(o[_IDX["battery_soc"]])
        harvest_recent = float(o[_IDX["harvest_recent"]])
        meas = float(o[_IDX["measurement"]])
        quality = float(o[_IDX["measurement_quality"]])
        has_measurement = quality > 0.0
        high_quality = quality > 0.5
        meas_age_h = float(o[_IDX["measurement_age"]]) * p.age_scale_h
        since_ack_h = float(o[_IDX["time_since_tx_success"]]) * p.age_scale_h
        app_age_h = float(o[_IDX["app_info_age"]]) * p.age_scale_h
        reported = float(o[_IDX["reported_value"]])
        has_reported = float(o[_IDX["time_since_tx_success"]]) < 1.0
        priority = int(round(float(o[_IDX["app_priority"]]) * 2))  # 0 / 1 / 2
        importance = float(o[_IDX["importance"]])
        urgent = priority >= 2
        eco = soc < p.soc_low and not urgent
        generous = (soc > p.soc_high or harvest_recent > p.harvest_strong) and not eco
        unreported = has_measurement and (since_ack_h > meas_age_h + 1e-6)
        delta = abs(meas - reported) if (has_measurement and has_reported) else (1.0 if has_measurement else 0.0)

        interval_h = p.report_interval_h[priority]
        if eco:
            interval_h *= p.report_interval_eco_factor
        elif generous and priority == 0:
            interval_h *= p.report_interval_generous_factor

        # 1. deep economy
        if soc < p.soc_critical:
            limit_h = p.report_interval_h[2] if urgent else p.deep_eco_interval_h
            if app_age_h >= limit_h:
                return encode_action(SENSE_HIGH, self._tx(o))
            return encode_action(SENSE_NONE, TX_NO)
        # 2. retry a fresh, unacknowledged *report* (high-quality sample); cheap
        #    low-cost checks are never retried, they are confirmed by rule 4.
        #    No retry while the link looks down (recent ACKs mostly missing):
        #    the next scheduled report will try again.
        link_ok = float(o[_IDX["link_quality"]]) >= p.retry_min_link_quality
        if unreported and high_quality and link_ok and meas_age_h < p.retry_age_h and app_age_h > interval_h:
            return encode_action(SENSE_NONE, self._tx(o))
        # 3. scheduled report
        if app_age_h >= interval_h:
            return encode_action(p.eco_sensing_level if eco else SENSE_HIGH, self._tx(o))
        # 4. event / importance report
        if unreported and (delta > p.event_delta or (importance > p.importance_immediate and delta > p.importance_delta)):
            return encode_action(SENSE_HIGH, self._tx(o))
        # 5. cheap check between reports
        if not eco and (not has_measurement or meas_age_h >= p.check_interval_h):
            return encode_action(SENSE_LOW, TX_NO)
        return encode_action(SENSE_NONE, TX_NO)


class RandomPolicy:
    """Uniform random actions (lower bound reference)."""

    def __init__(self, seed: int | None = None, n_modes: int = DEFAULT_N_MODES):
        self.rng = np.random.default_rng(seed)
        self.n_modes = n_modes

    def reset(self) -> None:
        pass

    def act(self, observation) -> np.ndarray:
        return unflatten_action(int(self.rng.integers(n_flat_actions(self.n_modes))), self.n_modes)


class PeriodicPolicy:
    """Sense (at a fixed level) and transmit every ``period_steps`` steps -
    the classic duty-cycled firmware, oblivious to energy and application."""

    def __init__(self, period_steps: int = 4, sensing_level: int = SENSE_LOW, tx: int = TX_YES):
        self.period = max(1, int(period_steps))
        self.level = sensing_level
        self.tx = tx  # 1 + radio mode
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def act(self, observation) -> np.ndarray:
        fire = self._t % self.period == 0
        self._t += 1
        return encode_action(self.level if fire else SENSE_NONE, self.tx if fire else TX_NO)


class AlwaysOnPolicy:
    """High-quality sensing and transmission at every step (upper bound on
    information, lower bound on energy prudence)."""

    def __init__(self, tx: int = TX_YES):
        self.tx = tx

    def reset(self) -> None:
        pass

    def act(self, observation) -> np.ndarray:
        return encode_action(SENSE_HIGH, self.tx)


@dataclass
class EpisodeResult:
    total_reward: float
    metrics: dict
    log: object
    infos: list = field(default_factory=list)


def run_episode(env, policy, *, seed: int | None = None, render: bool = False, render_every: int = 1, keep_infos: bool = False) -> EpisodeResult:
    """Roll out one episode of ``policy`` in ``env`` (Gymnasium loop)."""
    obs, info = env.reset(seed=seed)
    policy.reset()
    total = 0.0
    infos: list = []
    done = False
    step = 0
    while not done:
        action = policy.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if keep_infos:
            infos.append(info)
        if render and step % render_every == 0:
            env.render()
        done = terminated or truncated
        step += 1
    return EpisodeResult(total_reward=total, metrics=env.metrics.as_dict(), log=env.log, infos=infos)
