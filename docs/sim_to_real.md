# Simulation-to-real architecture

## Principle

```
                    ┌─────────────────────────────────────┐
                    │              Policy                 │   act(obs) -> (sense, tx)
                    │   RuleBasedPolicy / MLP / Q-table   │   knows nothing about backends
                    └───────────────▲────────────┬────────┘
                       observation  │            │ action
                    ┌───────────────┴────────────▼────────┐
                    │  ObservationBuilder   plan_execution │   shared contract (observation.py, actions.py)
                    │  NodeStateTracker                    │   identical code in both worlds
                    └───────────────▲────────────┬────────┘
                       NodeState    │            │ driver calls
              ┌─────────────────────┴─────┐  ┌───▼──────────────────────┐
              │  Simulation backend        │  │  Hardware backend         │
              │  (env.py)                  │  │  (deployment.NodeController)
              │  SolarEnergySource         │  │  harvester monitor / ADC   │
              │  SimulatedEnergyStorage    │  │  fuel gauge                │
              │  SimulatedSoilMoistureSensor│ │  sensor driver             │
              │  SimulatedLoRaRadio        │  │  LoRa stack + ACK          │
              │  RemoteMonitoringApplication│ │  downlink priority message │
              │  SimulatedClock            │  │  RTC / timer               │
              └────────────────────────────┘  └───────────────────────────┘
```

Both backends implement the six protocols of `interfaces.py` — `Clock`, `EnergyStorage`,
`EnergySource`, `Sensor`, `Radio`, `RemoteApplication`. Everything above the dashed line is
shared code: the *same* `NodeStateTracker` turns driver readings into a `NodeState`, the
*same* `ObservationBuilder` turns it into the 18-vector, the *same* `plan_execution`
applies the energy-feasibility rule, and the *same* policy object decides.
`tests/test_deployment.py` runs `RuleBasedPolicy` against a mock board that has no ground
truth at all, and checks that the simulator's tracker and the firmware tracker produce
identical observations from identical measurable inputs.

## What will differ between simulation and reality

| aspect | in the simulator | on the device | mitigation available now | later |
|---|---|---|---|---|
| **sensor noise** | Gaussian, level-dependent σ, optional bias | drift, temperature dependence, quantisation, soil-contact variability | randomise σ (`randomization.sensor_noise`), set `SensingConfig.bias` | replay real sensor traces through the `Sensor` protocol |
| **harvesting** | half-sine × AR(1) clouds | shading, panel soiling, angle, seasonal day length, MPPT behaviour | randomise `max_power_w`, cloud variability; change sunrise/sunset per season | replay irradiance traces (`EnergySource` from CSV) |
| **battery** | ideal buffer, exact SoC | voltage-based SoC estimate, temperature, ageing, self-discharge, charge losses | randomise capacity; `charge_efficiency` | ageing model, SoC estimation noise on `EnergyStorage` |
| **radio losses** | link-budget channel: AR(1) shadowing + per-attempt fading, logistic delivery in the margin | fading, interference, gateway congestion, duty-cycle limits, multiple gateways | randomise the mean path loss (`randomization.path_loss_db`), tune fading σ / correlation | congestion model, duty-cycle enforcement, measured RSSI/SNR traces |
| **communication energy** | constant per attempt and mode (SF7 / SF9 / SF12-like) | depends on payload, RX windows, retries, actual SF/power table of the radio | randomise mode energies together (`randomization.tx_energy`) | measured energy profile per (SF, power, payload) |
| **clock** | exact 15-minute steps | RTC drift, wake-up jitter | none needed (policy uses relative timers) | jitter on `SimulatedClock` |
| **environmental dynamics** | simplified ET, Poisson rain, external irrigation | real soil physics, weather fronts, crop growth | tune `AgricultureConfig`; randomise `et_rate_per_day` | field traces, crop model coupling |
| **application** | rule-based priority, Poisson requests | human operators, other data sources | `priority_update_mode="on_uplink"` for class-A realism | replay of real request logs |

## Domain randomisation

`DomainRandomizationConfig` (off by default) perturbs, at every `reset()`, the nominal
values of sensor noise, sensing energy, transmission energy, solar intensity, cloud
variability, battery capacity and baseline consumption by multiplicative factors, and the
mean path loss by an additive offset in dB, all drawn uniformly from configurable ranges. Two design points:

* randomisation happens in `randomize_config()` and touches only the configuration copy of
  the current episode — the observation/action spaces do not change (the number of radio
  modes is fixed by the base configuration);
* the **randomised energy costs are what the node reports** in observation components
  14–16, exactly as a real device would report its own measured profile. A policy trained
  under randomisation therefore learns to *read* its cost profile rather than assume it.

```python
cfg = ea.default_config()
cfg.randomization.enabled = True
cfg.randomization.solar_intensity = (0.5, 1.3)
env = ea.EdgeEngineAwareEnv(cfg)
```

## Rules that make the transfer credible

1. **No privileged variable may enter `NodeState`.** New observation components must come
   from something a driver can return.
2. **Normalisation constants and the link-budget table are part of the hardware profile**
   (`NodeProfile`: energies, transmit powers and sensitivities per mode) and are exported
   with the policy (`deployment.PolicyBundle`). Changing one without re-training invalidates
   the policy.
3. **Action semantics are frozen** in `actions.py`; firmware must map index → operation the
   same way.
4. **The feasibility rule runs on both sides.** Firmware that silently drops a transmission the
   policy requested behaves differently from training; it must run `plan_execution` (or its C
   translation) so that rejected actions are handled identically.
5. **Evaluate with the same metrics.** `EpisodeMetrics` (AoI, delivered packets, energy split,
   depletion events) is computed from quantities the real system also has — except the
   utility, which needs ground truth. In the field, ground truth comes from a reference
   sensor logging at high rate (see `docs/deployment.md`).

## Extending the backends

* A **trace-driven backend** (real harvesting or sensor logs) is a class reading from a table
  indexed by time. For the *deployment* loop it only needs the protocol (`measured_power_w()`,
  `read(level, now)`). For the *Gymnasium environment* it must also expose what the simulator
  needs from its world models (`update(t)`, `harvested_energy_j()`, `true_power_w()` for the
  source; the true value callable for the sensor); plug it in by subclassing
  `EdgeEngineAwareEnv` and overriding `_build_subsystems` (the subsystems are rebuilt at every
  `reset()`, so assigning `env.source` after construction is not enough).
* **Hardware-in-the-loop** is `NodeController` with drivers that talk to a board over serial —
  the controller already runs one decision cycle per call and leaves sleeping to the caller.
* **Multiple nodes / sensors / applications** are lists of the corresponding objects; the
  observation would gain a block per sensor and the `Packet` a field per value.
