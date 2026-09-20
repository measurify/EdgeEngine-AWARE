# From simulation to reality: the architecture

This document explains how the code is organised so that a policy trained in simulation
runs unchanged on a device, what will nevertheless differ between the two, and which
mitigations exist.

## The principle

```
                    ┌─────────────────────────────────────┐
                    │              Policy                 │   act(obs) -> (sense, tx)
                    │   RuleBasedPolicy / MLP / Q-table   │   knows nothing about the backends
                    └───────────────▲────────────┬────────┘
                       observation  │            │ action
                    ┌───────────────┴────────────▼────────┐
                    │  NodeStateTracker  ObservationBuilder│   SHARED CONTRACT (observation.py, actions.py)
                    │  plan_execution                      │   the same code in every world
                    └───────────────▲────────────┬────────┘
                       measurable   │            │ driver calls
                       readings     │            │
        ┌───────────────────────────┴───┐  ┌─────▼──────────────────────────┐  ┌──────────────────────────┐
        │  Simulation backend (env.py)  │  │  Trace-driven backend           │  │  Hardware backend         │
        │  SolarEnergySource            │  │  (traces.py)                    │  │  (deployment.NodeController│
        │  SimulatedEnergyStorage       │  │  TraceSolarEnergySource         │  │   or firmware/eea_node.c)  │
        │  SimulatedSoilMoistureSensor  │  │  TraceFieldEnvironment          │  │  harvester monitor / ADC   │
        │  SimulatedLoRaRadio           │  │  + the simulated battery, sensor│  │  fuel gauge, sensor driver │
        │  RemoteMonitoringApplication  │  │    radio and application        │  │  LoRa stack + ACK, RTC,    │
        │  SimulatedClock               │  │                                 │  │  downlink priority message │
        └───────────────────────────────┘  └─────────────────────────────────┘  └──────────────────────────┘
```

Every backend produces the same kind of *measurable readings* (stored energy, harvested
power, a sensor value, an ACK, a priority message) and consumes the same kind of *driver
calls* (read the sensor at level *l*, send the stored reading in mode *k*). In Python these
are the six **protocols** of `interfaces.py` — `Clock`, `EnergyStorage`, `EnergySource`,
`Sensor`, `Radio`, `RemoteApplication` (a `typing.Protocol` is an interface: any class with
the right methods satisfies it, no inheritance needed).

Everything in the middle box is **shared code**: the *same* `NodeStateTracker` turns readings
into a `NodeState`, the *same* `ObservationBuilder` turns it into the 18-vector, the *same*
`plan_execution` applies the feasibility rule, and the *same* policy object decides. The C
runtime in `firmware/` is a line-by-line translation of that middle box, tested for
equivalence.

Two tests pin the principle down:

* `tests/test_deployment.py` replays the driver readings of a simulated episode through
  `NodeController` (which has no ground truth at all) and checks that its observations are
  identical to the simulator's, in both priority-downlink modes;
* `tests/test_firmware.py` does the same between Python and C.

## What will differ between simulation and reality

| aspect | in the simulator | on a real device | mitigation available now | possible later |
|---|---|---|---|---|
| **sensor noise** | Gaussian, one σ per sensing level, optional bias | drift, temperature dependence, quantisation, variable soil contact | randomise σ (`randomization.sensor_noise`); set `SensingConfig.bias` | replay logged sensor readings through the `Sensor` protocol |
| **harvesting** | half-sine day × AR(1) clouds, fixed sunrise/sunset | shading, panel soiling, tilt, seasons, converter behaviour | randomise `max_power_w` and cloud variability; change sunrise/sunset; **replay recorded irradiance** (`traces.TraceSolarEnergySource`) | measured panel I–V behaviour |
| **battery** | ideal buffer, exact energy reading | voltage-based estimate, temperature, ageing, self-discharge, charge losses | randomise capacity; `charge_efficiency` | ageing model, noise on the energy reading |
| **radio losses** | link-budget channel: AR(1) shadowing + per-attempt fading, logistic delivery | fading, interference, gateway congestion, duty-cycle limits, several gateways | randomise the mean path loss (`randomization.path_loss_db`); tune fading σ and correlation | congestion model, duty-cycle enforcement, replay of measured RSSI/SNR |
| **communication energy** | constant per attempt and mode | depends on payload, receive windows, retries, the radio's actual SF/power table | randomise the mode energies together (`randomization.tx_energy`) | measured energy per (SF, power, payload) |
| **clock** | exact 15-minute steps | RTC drift, wake-up jitter | none needed for the timers (they are relative); the time-of-day components (3–4) tolerate minutes of drift | jitter on `SimulatedClock` |
| **soil / weather** | one-bucket soil, Poisson rain, external irrigation, fixed climate | real soil physics, weather fronts, seasons, crop growth | tune `AgricultureConfig`; **replay recorded soil moisture and weather** (`traces.TraceFieldEnvironment`) | crop model coupling |
| **application** | rule-based priority, Poisson campaigns | human operators, other data sources | `priority_update_mode="on_uplink"` for LoRaWAN class-A realism (downlink only after an uplink) | replay of real request logs |

## Domain randomisation

`DomainRandomizationConfig` (`cfg.randomization`, off by default) perturbs the nominal
physical parameters at every `reset()` so that a trained policy does not over-fit one exact
device or climate. Each factor is drawn uniformly from a range and multiplies the nominal
value (the path-loss offset is added, in dB):

| parameter | default range | applies to |
|---|---|---|
| `sensor_noise` | ×[0.7, 1.5] | σ of both sensing levels (same factor) |
| `sensing_energy` | ×[0.8, 1.3] | energy of both sensing levels |
| `tx_energy` | ×[0.8, 1.3] | energy of all radio modes (same factor) |
| `path_loss_db` | +[−3, +3] dB | mean path loss |
| `solar_intensity` | ×[0.6, 1.2] | panel peak power |
| `cloud_variability` | ×[0.5, 1.5] | intra-day cloud noise and day-to-day clearness spread |
| `battery_capacity` | ×[0.8, 1.2] | capacity (the initial SoC is a fraction, so the initial energy scales too) |
| `baseline_power` | ×[0.7, 1.5] | always-on consumption |

Two design points:

* randomisation happens in `config.randomize_config()` and touches only the configuration
  copy of the current episode; the observation and action spaces do not change (the number of
  radio modes is fixed by the base configuration);
* the **randomised energy costs are what the node reports** in observation components
  15–17, exactly as a real device would report its own measured profile. A policy trained
  under randomisation therefore learns to *read* its cost profile rather than assume it.

```python
import edgeengine_aware as ea

cfg = ea.default_config()
cfg.randomization.enabled = True
cfg.randomization.solar_intensity = (0.5, 1.3)
env = ea.EdgeEngineAwareEnv(cfg)
```

Evaluation in the benchmark protocol uses the *nominal* (non-randomised) physics so that
results are comparable across policies.

## Rules that make the transfer credible

1. **No hidden variable may enter `NodeState`.** A new observation component must come from
   something a driver can return.
2. **Normalisation constants and the link-budget table are part of the hardware profile**
   (`NodeProfile`: energies, transmit powers and sensitivities per mode, thresholds, scales)
   and are exported with the policy (`deployment.PolicyBundle`). Changing one without
   re-training invalidates the policy.
3. **Action semantics are frozen** in `actions.py`; the firmware must map the same index to
   the same operation.
4. **The feasibility rule runs on both sides.** A firmware that silently drops a transmission
   the policy requested behaves differently from training; it must run `plan_execution` (or
   its C translation) so that rejected actions are handled identically.
5. **Evaluate with the same metrics.** `EpisodeMetrics` (age of information, delivered
   packets, energy split, brown-outs) is computed from quantities the real system also has —
   except the utility, which needs ground truth. In the field, ground truth comes from a
   reference sensor logging at high rate (`docs/deployment.md`).

Three "SoC" thresholds exist and are easy to confuse: `EnergyStorageConfig.reserve_soc`
(2 %, the feasibility rule's brown-out reserve), `RewardConfig.safe_soc` (30 %, where the
battery-risk penalty starts) and `RuleBasedParams.soc_low` / `soc_critical` (50 % / 20 %,
the rule-based controller's economy modes). Only the first is part of the hardware contract.

## Adding a backend

* **Trace-driven backends** (`edgeengine_aware.traces`) replay a `Trace` loaded from CSV:
  `TraceSolarEnergySource` (irradiance → panel power) and `TraceFieldEnvironment` (soil
  moisture, temperature, humidity, rain). `TraceDrivenEnv` subclasses `EdgeEngineAwareEnv`,
  overrides `_build_subsystems` — the subsystems are rebuilt at every `reset()`, so assigning
  `env.source` after construction is not enough — and draws the replayed window in `reset()`.
  The same pattern adds a trace-driven `Radio` (recorded RSSI/SNR) or `Sensor`. Two
  conventions matter with weather archives: irradiance and rain are *means/sums over the
  preceding interval* (piecewise constant), state variables are *instant* (interpolated);
  and timestamps must be in local standard time so that the simulator's clock and the sun
  agree. For the *deployment* loop a backend only needs the protocol (`measured_power_w()`,
  `read(level, now)`, …); for the *environment* it must also expose what `env.py` calls on its world models — for a
  source: `reset(rng, start_time_s)`, `update(t)`, `measured_power_w()`, `true_power_w()`,
  `harvested_energy_j()` and the `daily_clearness` property; for a field: `reset(rng,
  start_time_s)`, `step(t)`, `state()` and the `moisture` attribute the sensor reads (see
  `traces.py` for a complete example of both).
* **Hardware-in-the-loop** is `NodeController` with drivers that talk to a board over a
  serial link — the controller already runs one decision cycle per call and leaves sleeping
  to the caller. Or run the C runtime on the board itself (`firmware/`).
* **Several nodes / sensors / applications** are lists of the corresponding objects; the
  observation would gain a block per sensor and the `Packet` a field per value.
