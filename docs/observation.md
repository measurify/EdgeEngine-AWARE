# Policy observation vector

The policy input is a `float32` vector of dimension **17**, every component in `[0, 1]`
(`gymnasium.spaces.Box(0, 1, (17,), float32)`). It is produced by
`edgeengine_aware.observation.ObservationBuilder.build(NodeState)` — the *only* place
where raw node quantities become policy inputs — from a `NodeState` that contains
exclusively **hardware-measurable** quantities.

## Three kinds of variables

| kind | examples | where it lives | who may use it |
|---|---|---|---|
| **simulator ground truth** (hidden) | true soil moisture, true irradiance and daily clearness, channel success probability, future rain | `FieldEnvironment`, `SolarEnergySource`, `SimulatedLoRaRadio` | world dynamics, reward (`application.py`), rendering, `info["ground_truth"]` |
| **measurable on the node** | stored energy, harvested power, sensor readings (noisy), timers, ACKs, downlink priority | `NodeState` (via `NodeStateTracker`) | anything — this is what a real board knows |
| **policy input** | the 17 normalised numbers below | `ObservationBuilder` output | the policy |

The policy input is a *subset and normalisation* of the measurable state. Nothing from the
first row can reach it: `tests/test_env.py::test_observation_contains_no_privileged_information`
mutates every hidden variable and checks that the observation does not change.

## Components

| idx | name | meaning | normalisation | hardware source |
|---|---|---|---|---|
| 0 | `battery_soc` | state of charge | `E / E_max` | fuel gauge or ADC on the storage element |
| 1 | `harvest_power` | harvesting power measured over the **elapsed** interval | `P / harvest_ref_power_w`, clipped to 1 | current/power monitor of the harvester |
| 2 | `harvest_recent` | EWMA of `harvest_power` (α = 0.2 → ~1 h memory at 15-min steps) | as above | computed in firmware |
| 3 | `time_of_day_sin` | time of day (cyclic encoding) | `(sin(2πt/86400)+1)/2` | RTC |
| 4 | `time_of_day_cos` | time of day (cyclic encoding) | `(cos(2πt/86400)+1)/2` | RTC |
| 5 | `measurement` | latest stored soil-moisture sample (0 if none) | moisture units already in [0, 1] | sensor driver + RAM |
| 6 | `measurement_quality` | quality tag of the stored sample | `1 − 0.8·σ/σ_low` → 0 none, 0.2 low-cost, 0.8 high-quality | hardware profile |
| 7 | `measurement_age` | time since the stored sample was taken | `age / age_scale_s` (24 h), clipped; 1 if none | timer |
| 8 | `time_since_tx_success` | time since the last acknowledged uplink | `t / age_scale_s`, clipped; 1 if none | timer + radio ACK |
| 9 | `app_info_age` | node-side estimate of the age of information at the application = (8) + age the sample had when sent | `age / age_scale_s`, clipped; 1 if none | timer + radio ACK |
| 10 | `reported_value` | value carried by the last acknowledged uplink (0 if none) | moisture units | RAM |
| 11 | `app_priority` | priority requested by the application: routine / elevated / urgent | `p / 2` → 0, 0.5, 1 | downlink message |
| 12 | `importance` | node-side importance of the stored sample: 1 below the critical threshold, `exp(−dist/0.10)` otherwise, 0 if none | already in [0, 1] | computed in firmware from flash thresholds |
| 13 | `link_quality` | EWMA of ACK outcomes (1 = every recent uplink delivered) | already in [0, 1] | radio ACK |
| 14 | `sense_low_cost` | energy of a low-cost sample | `E / E_max`, clipped | hardware profile (flash constant or measured) |
| 15 | `sense_high_cost` | energy of a high-quality sample | `E / E_max`, clipped | hardware profile |
| 16 | `tx_cost` | energy of one uplink attempt | `E / E_max`, clipped | hardware profile |

`ObservationBuilder(profile).describe()` prints this table from code, and
`ObservationBuilder.to_dict(obs)` maps a vector back to names.

## Design notes

* **Past, not future, harvesting.** Components 1–2 describe the interval that already
  elapsed (what a harvester monitor integrates), never the upcoming solar profile.
  The agent must *infer* the daily cycle from the time-of-day encoding and the recent
  history — exactly what a deployed node would have to do.
* **The application's information age is estimated, not observed.** Component 9 is
  computed by the node from its own timers and ACKs. It is exact for confirmed uplinks;
  with unconfirmed uplinks (`CommunicationConfig.ack_available = False`) the node assumes
  delivery, so the estimate is optimistic and `link_quality` stays at 1.
* **Costs are in the observation** (14–16) even though they are constant within an
  episode: under domain randomisation they change between episodes, so a policy trained
  with them can adapt to a device whose measured profile differs from the nominal one.
* **Ages are clipped at 24 h.** Beyond one day, "very old" is all the policy needs to know;
  the clipping keeps the vector bounded for function approximators and for fixed-point
  firmware.

## Markov property and partial observability

The observation is *not* a full Markov state of the simulated world:

1. **Hidden environment state.** True soil moisture, the daily clearness index, the intra-day
   cloud process and the channel-quality process are hidden. The node only has noisy,
   dated samples of the first and a noisy power measurement that mixes the other two.
2. **Hidden application state.** External monitoring requests and the application's
   "unreported event" flag are hidden; only the resulting priority is visible.

The problem is therefore a POMDP. The observation contains the standard, cheap devices that
make a *reactive* policy work well in practice:

* **sufficient statistics of the recent past** — the EWMA of harvesting (2), the EWMA of ACK
  outcomes (13), the stored sample with its age and quality (5–7);
* **explicit timers** — ages since the last sample / ACK (7–9) turn "how long ago" into a state
  variable instead of requiring memory;
* **cyclic time** (3–4) — the solar and evapotranspiration cycles are functions of time of day.

If a learned policy needs more, two extensions keep the hardware contract intact: (a) stack
`k` consecutive observations (frame stacking, `k·17` inputs, trivially done in firmware with a
ring buffer), or (b) use a recurrent policy whose hidden state is carried across wake-ups.
Both are supported by `ObservationBuilder` unchanged, since they are wrappers around it.
