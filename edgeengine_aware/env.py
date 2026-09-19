"""Gymnasium environment of EdgeEngine AWARE.

Timeline of one ``step(action)`` call (decision taken at time t, step dt):

1. **Feasibility** - using the energy stored at t (what a fuel gauge would
   report) the node checks which requested operations it can afford after
   reserving the baseline consumption of the interval and the brown-out
   reserve. Operations it cannot afford are *rejected* (not executed, no
   energy spent, small penalty). Sensing has priority over transmission.
2. **Sensing** - the sensor samples the *true* field at t with the noise of
   the selected level; the measurement is stored on the node.
3. **Transmission** - the latest stored measurement is put in a packet and
   sent; energy is spent regardless of the outcome. On delivery the remote
   application updates its picture of the field and credits a packet bonus.
4. **Energy update** - E(t+dt) = clip(E(t) + harvested - baseline - sensing - comm, 0, E_max).
   Consumption is drawn first and the energy harvested during the interval
   is stored afterwards (a conservative choice: the interval's own harvest
   cannot pay for the interval's load). If the baseline load cannot be
   served the node *browns out* (depletion).
5. **World update** - clock, field, harvesting process, channel and
   application priority advance to t + dt.
6. **Reward** - tracking utility of the application's (possibly stale)
   picture against the new ground truth, plus the packet bonus, minus costs
   and penalties (see ``application.py`` and ``reward.py``).
7. **Observation** - the node-side tracker is refreshed with measurable
   quantities only and the ObservationBuilder produces the vector for t + dt.
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import SENSE_NONE, action_nvec, decode_action, plan_execution
from .agriculture import FieldEnvironment
from .application import RemoteMonitoringApplication
from .communication import SimulatedLoRaRadio
from .config import EdgeEngineAwareConfig, default_config, randomize_config
from .energy import SimulatedClock, SimulatedEnergyStorage, SolarEnergySource
from .interfaces import Packet
from .metrics import EpisodeLog, EpisodeMetrics
from .observation import NodeProfile, NodeStateTracker, ObservationBuilder
from .reward import RewardCalculator
from .sensing import SimulatedSoilMoistureSensor


class EdgeEngineAwareEnv(gym.Env):
    """Energy-harvesting Edge IoT node in a smart-agriculture field.

    Observation: ``Box(0, 1, (18,), float32)`` - see ``ObservationBuilder``.
    Action:      ``MultiDiscrete([3, 1 + n_modes])`` - (sensing level, transmit /
                 radio mode); ``[3, 4]`` with the default three-mode radio.
    """

    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 10}

    def __init__(self, config: EdgeEngineAwareConfig | None = None, render_mode: str | None = None):
        super().__init__()
        self.base_config = config if config is not None else default_config()
        self.base_config.validate()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode

        # Spaces are independent from the (possibly randomised) physical parameters.
        self.cfg = self.base_config.copy()
        self.profile = NodeProfile.from_config(self.cfg)
        self.obs_builder = ObservationBuilder(self.profile)
        self.observation_space = spaces.Box(self.obs_builder.low, self.obs_builder.high, dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(np.array(action_nvec(self.cfg.communication.n_modes), dtype=np.int64))

        self._build_subsystems(self.cfg)
        self._renderer = None
        self._step_count = 0
        self._last_info: dict[str, Any] = {}
        self._last_measured_harvest_w = 0.0
        self._last_harvested_j = 0.0
        self._last_step: dict[str, Any] = self._empty_last_step()
        self.metrics = EpisodeMetrics()
        self.log = EpisodeLog()

    @staticmethod
    def _empty_last_step() -> dict[str, Any]:
        return {"sensing_level": SENSE_NONE, "tx_attempted": False, "tx_mode": -1, "tx_success": None, "reward": 0.0, "utility": 0.0}

    # ------------------------------------------------------------------
    # construction helpers
    # ------------------------------------------------------------------
    def _build_subsystems(self, cfg: EdgeEngineAwareConfig) -> None:
        dt = cfg.time.timestep_s
        self.clock = SimulatedClock(cfg.time)
        self.storage = SimulatedEnergyStorage(cfg.storage, charge_efficiency=cfg.storage.charge_efficiency)
        self.source = SolarEnergySource(cfg.harvesting, dt)
        self.field = FieldEnvironment(cfg.agriculture, dt)
        self.sensor = SimulatedSoilMoistureSensor(cfg.sensing, true_value=lambda: self.field.moisture)
        self.radio = SimulatedLoRaRadio(cfg.communication)
        self.application = RemoteMonitoringApplication(cfg.application, cfg.agriculture, dt)
        self.reward_fn = RewardCalculator(cfg.reward)
        self.profile = NodeProfile.from_config(cfg)
        self.obs_builder = ObservationBuilder(self.profile)
        self.tracker = NodeStateTracker(self.profile)

    @property
    def max_steps(self) -> int:
        return self.cfg.time.max_steps

    @property
    def timestep_s(self) -> float:
        return self.cfg.time.timestep_s

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        # (Re)build the physical parameters, applying domain randomisation.
        self.cfg = randomize_config(self.base_config, self.np_random)
        self._build_subsystems(self.cfg)

        start = self.cfg.time.start_hour * 3600.0
        dt = self.cfg.time.timestep_s
        rngs = [np.random.default_rng(int(self.np_random.integers(0, 2**63 - 1))) for _ in range(5)]
        self.storage.reset(rngs[0])
        # The harvesting process is started one interval early so that the node's
        # first reading is the power of the interval that *preceded* the episode
        # (what a harvester monitor reports at boot), never the upcoming one.
        self.source.reset(rngs[1], start_time_s=start - dt)
        previous_interval_measured_w = self.source.measured_power_w()
        self.source.update(start)
        self.field.reset(rngs[2], start_time_s=start)
        self.sensor.reset(rngs[3])
        self.radio.reset(rngs[4])
        self.application.reset(np.random.default_rng(int(self.np_random.integers(0, 2**63 - 1))), start_time_s=start)
        self.clock.reset()
        self.tracker.reset()

        self._step_count = 0
        self._last_measured_harvest_w = previous_interval_measured_w
        self._last_harvested_j = 0.0
        self.metrics = EpisodeMetrics()
        self.log = EpisodeLog()
        self._last_step = self._empty_last_step()

        self.application.step(self.clock.now_s(), self.field.state())
        self.tracker.begin_step(
            now_s=self.clock.now_s(),
            time_of_day_s=self.clock.time_of_day_s(),
            energy_j=self.storage.energy_j(),
            capacity_j=self.storage.capacity_j(),
            harvest_power_w=self._last_measured_harvest_w,
            priority=self.application.priority(),  # initial downlink assumed at join
        )
        obs = self.obs_builder.build(self.tracker.state())
        info = self._make_info(reward_components=None, utility_breakdown=None, rejected=[])
        self._last_info = info
        return obs, info

    def step(self, action):
        act = decode_action(action, n_modes=self.cfg.communication.n_modes)
        cfg = self.cfg
        dt = cfg.time.timestep_s
        now = self.clock.now_s()
        truth_before = self.field.state()

        # 1. feasibility (same rule as the firmware loop, see actions.plan_execution)
        baseline_j = cfg.mcu.baseline_power_w * dt
        plan = plan_execution(
            act,
            stored_energy_j=self.storage.energy_j(),
            baseline_energy_j=baseline_j,
            reserve_energy_j=self.storage.reserve_j(),
            sensing_energy_j=tuple(self.sensor.energy_cost_j(l) for l in range(cfg.sensing.n_levels)),
            tx_energy_j=tuple(self.radio.tx_energy_j(k) for k in range(self.radio.n_modes())),
            has_measurement=self.tracker.measurement is not None,
        )
        rejected = list(plan.rejected)
        sensing_level, sensing_j = plan.sensing_level, plan.sensing_energy_j
        tx_executed, tx_j, tx_mode = plan.transmit, plan.tx_energy_j, plan.mode

        # 2. sensing ----------------------------------------------------------
        measurement = None
        if sensing_level != SENSE_NONE:
            measurement = self.sensor.read(sensing_level, now)
            self.tracker.on_measurement(measurement)

        # 3. transmission -----------------------------------------------------
        tx_success: bool | None = None
        utility = 0.0
        utility_breakdown = None
        if tx_executed:
            packet = Packet(measurement=self.tracker.measurement, sent_at_s=now)  # type: ignore[arg-type]
            result = self.radio.transmit(packet, tx_mode)
            tx_success = result.acked
            # without confirmations the node cannot know the outcome (None)
            self.tracker.on_transmission(packet, tx_success if cfg.communication.ack_available else None, now, mode=tx_mode, margin_db=result.margin_db)
            if tx_success:
                utility_breakdown = self.application.receive(packet, now, truth_before)
                utility = utility_breakdown.total
                if cfg.communication.priority_update_mode == "on_uplink":
                    self.tracker.set_priority(self.application.priority())

        # 4. energy update ----------------------------------------------------
        requested = baseline_j + sensing_j + tx_j
        delivered = self.storage.discharge(requested)
        depleted = delivered + 1e-9 < requested  # brown-out: the load could not be served
        harvested_j = self.source.harvested_energy_j()
        wasted_before = self.storage.wasted_j
        self.storage.charge(harvested_j)
        wasted_j = self.storage.wasted_j - wasted_before
        self._last_measured_harvest_w = self.source.measured_power_w()
        self._last_harvested_j = harvested_j

        # 5. world update -----------------------------------------------------
        self.clock.advance()
        t_next = self.clock.now_s()
        truth_after = self.field.step(now)  # dynamics over [now, now + dt]
        self.source.update(t_next)
        self.radio.update_channel()
        self.application.step(t_next, truth_after)
        self._step_count += 1

        # 6. reward -----------------------------------------------------------
        aoi = self.application.age_of_information_s(t_next)
        tracking = self.application.tracking(truth_after)  # privileged evaluation
        utility += tracking.utility
        comps = self.reward_fn.compute(
            utility=utility,
            sensing_energy_j=sensing_j,
            communication_energy_j=tx_j,
            aoi_s=aoi,
            priority=self.application.priority(),
            soc_after=self.storage.soc(),
            depleted=depleted,
            n_rejected=len(rejected),
            wasted_energy_j=wasted_j,
        )
        reward = float(comps.total)

        # 7. observation ------------------------------------------------------
        if cfg.communication.priority_update_mode == "immediate":
            prio_downlink: int | None = self.application.priority()
        else:
            prio_downlink = None  # only refreshed with an ACK (handled above)
        self.tracker.begin_step(
            now_s=t_next,
            time_of_day_s=self.clock.time_of_day_s(),
            energy_j=self.storage.energy_j(),
            capacity_j=self.storage.capacity_j(),
            harvest_power_w=self._last_measured_harvest_w,
            priority=prio_downlink,
        )
        obs = self.obs_builder.build(self.tracker.state())

        # 8. bookkeeping ------------------------------------------------------
        self._update_metrics(
            harvested_j=harvested_j,
            baseline_j=baseline_j,
            sensing_j=sensing_j,
            tx_j=tx_j,
            wasted_j=wasted_j,
            sensing_level=sensing_level,
            tx_executed=tx_executed,
            tx_mode=tx_mode,
            tx_success=tx_success,
            rejected=len(rejected),
            depleted=depleted,
            aoi=aoi,
            utility=utility,
            reward=reward,
            comps=comps,
        )
        self._last_step = {
            "sensing_level": sensing_level,
            "tx_attempted": tx_executed,
            "tx_mode": tx_mode,
            "tx_success": tx_success,
            "reward": reward,
            "utility": utility,
            "requested_action": (act.sensing_level, act.transmit),
        }
        self._append_log(truth_after, aoi, sensing_level, tx_executed, tx_mode, tx_success, reward, utility)

        terminated = bool(cfg.terminate_on_depletion and depleted)
        truncated = self._step_count >= self.max_steps
        info = self._make_info(comps, utility_breakdown, rejected)
        info["tracking"] = tracking.as_dict()
        self._last_info = info
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("render() called without a render_mode; nothing will be produced.")
            return None
        from .rendering import DashboardRenderer  # lazy: keeps matplotlib optional for training

        if self._renderer is None:
            self._renderer = DashboardRenderer(self)
        return self._renderer.render(self.render_mode)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # privileged accessors (evaluation / rendering only)
    # ------------------------------------------------------------------
    def ground_truth(self) -> dict[str, Any]:
        """Simulator-only state. Never feed this to a policy."""
        fs = self.field.state()
        return {
            "time_s": self.clock.now_s(),
            "soil_moisture": fs.soil_moisture,
            "air_temperature_c": fs.air_temperature_c,
            "relative_humidity": fs.relative_humidity,
            "zone": fs.zone,
            "event_occurred": fs.event_occurred,
            "harvest_power_true_w": self.source.true_power_w(),
            "daily_clearness": self.source.daily_clearness,
            "path_loss_db": self.radio.path_loss_db(),
            "tx_success_probability": self.radio.success_probability(),  # reference mode
            "tx_success_probability_per_mode": [self.radio.success_probability(k) for k in range(self.radio.n_modes())],
            "stored_energy_j": self.storage.energy_j(),
            "app_aoi_s": self.application.age_of_information_s(),
            "app_priority": self.application.priority(),
            "app_last_value": None if self.application.last_packet is None else self.application.last_packet.measurement.value,
            "unreported_event": self.application.has_unreported_event,
        }

    def node_state(self):
        """Hardware-measurable state (what a real node knows)."""
        return self.tracker.state()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _make_info(self, reward_components, utility_breakdown, rejected: list[str]) -> dict[str, Any]:
        info: dict[str, Any] = {
            "step": self._step_count,
            "time_s": self.clock.now_s(),
            "hour_of_day": self.clock.hour_of_day(),
            "day": self.clock.day_index(),
            "battery_soc": self.storage.soc(),
            "stored_energy_j": self.storage.energy_j(),
            "harvested_energy_j": self._last_harvested_j,
            "executed_action": (self._last_step["sensing_level"], self._last_step["tx_mode"] + 1 if self._last_step["tx_attempted"] else 0),
            "sensing_level": self._last_step["sensing_level"],
            "tx_attempted": self._last_step["tx_attempted"],
            "tx_mode": self._last_step["tx_mode"],
            "tx_success": self._last_step["tx_success"],
            "rejected": list(rejected),
            "app_priority": self.application.priority(),  # what the application wants now
            "node_priority": self.tracker.priority,  # what the node has been told (differs in 'on_uplink' mode)
            "app_aoi_s": self.application.age_of_information_s(),
            "reward_components": reward_components.as_dict() if reward_components is not None else {},
            "utility_breakdown": utility_breakdown.as_dict() if utility_breakdown is not None else {},
            "metrics": self.metrics.as_dict(),
            "ground_truth": self.ground_truth(),  # privileged: evaluation only
            "observation_names": self.obs_builder.names,
        }
        return info

    def _update_metrics(self, *, harvested_j, baseline_j, sensing_j, tx_j, wasted_j, sensing_level, tx_executed, tx_mode, tx_success, rejected, depleted, aoi, utility, reward, comps) -> None:
        m = self.metrics
        m.steps += 1
        m.total_harvested_energy_j += harvested_j
        m.total_consumed_energy_j += baseline_j + sensing_j + tx_j
        m.baseline_energy_j += baseline_j
        m.sensing_energy_j += sensing_j
        m.communication_energy_j += tx_j
        m.wasted_harvest_energy_j += wasted_j
        if sensing_level != SENSE_NONE:
            m.n_sensing += 1
            if sensing_level == 2:
                m.n_high_quality_sensing += 1
        if tx_executed:
            m.n_transmissions += 1
            m.transmissions_per_mode[tx_mode] = m.transmissions_per_mode.get(tx_mode, 0) + 1
            if tx_success:
                m.n_successful_transmissions += 1
                m.deliveries_per_mode[tx_mode] = m.deliveries_per_mode.get(tx_mode, 0) + 1
        m.n_rejected_actions += rejected
        if depleted:
            m.battery_depletion_events += 1
        soc = self.storage.soc()
        m._soc_sum += soc
        m.min_battery_soc = min(m.min_battery_soc, soc)
        if soc < self.cfg.reward.safe_soc:
            m.steps_low_battery += 1
        m._aoi_sum_s += aoi
        m.max_aoi_s = max(m.max_aoi_s, aoi)
        m.total_application_utility += utility
        m.total_reward += reward
        for k, v in comps.as_dict().items():
            m.reward_components[k] = m.reward_components.get(k, 0.0) + v

    def _append_log(self, truth, aoi, sensing_level, tx_executed, tx_mode, tx_success, reward, utility) -> None:
        meas = self.tracker.measurement
        app_pkt = self.application.last_packet
        self.log.append(
            time_s=self.clock.now_s(),
            soc=self.storage.soc(),
            harvest_power_w=self._last_measured_harvest_w,
            harvested_energy_j=self._last_harvested_j,
            true_moisture=truth.soil_moisture,
            measured_moisture=meas.value if meas is not None else math.nan,
            app_moisture=app_pkt.measurement.value if app_pkt is not None else math.nan,
            sensing_level=int(sensing_level),
            tx_attempt=int(tx_executed),
            tx_mode=int(tx_mode),
            tx_success=int(bool(tx_success)),
            path_loss_db=self.radio.path_loss_db(),
            aoi_s=aoi,
            priority=int(self.application.priority()),
            reward=reward,
            utility=utility,
            temperature_c=truth.air_temperature_c,
            humidity=truth.relative_humidity,
        )
