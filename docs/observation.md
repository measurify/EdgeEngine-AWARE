# The policy observation

This document describes the 18 numbers a policy receives at every decision, where each of
them comes from, and why the problem is only *partially* observable.

## What the observation is

The policy input is a `float32` vector of length **18**, every component in `[0, 1]`
(Gymnasium space `Box(0, 1, (18,), float32)`). It is produced by one function,
`edgeengine_aware.observation.ObservationBuilder.build(state)`, from a `NodeState` object.
`NodeState` holds only quantities that a real microcontroller can measure or compute on
its own: battery energy from a fuel gauge, harvested power from a current monitor, the last
sensor reading and its age from RAM and a timer, radio acknowledgements, the last priority
message received from the application.

Three kinds of variables exist in the simulator; only the third one reaches the policy:

| kind | examples | where it lives in the code | who may use it |
|---|---|---|---|
| **hidden ground truth** | true soil moisture, true irradiance, daily clearness, the radio channel's fading state, delivery probabilities, future rain | `FieldEnvironment`, `SolarEnergySource`, `SimulatedLoRaRadio` | the world dynamics, the reward, the plots, `info["ground_truth"]` |
| **measurable on the node** | stored energy, harvested power, noisy readings, timers, ACK outcomes, downlink priority | `NodeState`, maintained by `NodeStateTracker` | anything on the node |
| **policy input** | the 18 normalised numbers below | output of `ObservationBuilder` | the policy |

`tests/test_env.py::test_observation_contains_no_privileged_information` changes every hidden
variable and checks that the observation does not change.

## The 18 components

Notation: `E_max` is the battery capacity, `age_scale_s` = 24 h, `harvest_ref_power_w` = 3 mW
(the harvesting power that maps to 1.0), all from `ObservationConfig` and carried in the
`NodeProfile`. "Clipped" means values above 1 are set to 1.

| idx | name | meaning | how it is normalised | where a real node gets it |
|---|---|---|---|---|
| 0 | `battery_soc` | state of charge | `E / E_max` | fuel gauge, or battery voltage through an ADC and a lookup table |
| 1 | `harvest_power` | average harvesting power over the interval that **just ended** | `P / 3 mW`, clipped | current/power monitor of the harvester (energy counted since the last wake-up, divided by the interval) |
| 2 | `harvest_recent` | smoothed harvesting power: EWMA of (1) with α = 0.2, i.e. a memory of roughly the last hour at 15-minute steps | as (1) | computed in firmware |
| 3 | `time_of_day_sin` | time of day, sine part | `(sin(2π t/86400) + 1) / 2` | real-time clock (RTC) |
| 4 | `time_of_day_cos` | time of day, cosine part | `(cos(2π t/86400) + 1) / 2` | RTC |
| 5 | `measurement` | latest stored soil-moisture reading (0 if none yet) | already in [0, 1] (fraction of field capacity) | sensor driver + RAM |
| 6 | `measurement_quality` | quality tag of the stored reading | `1 − 0.8·σ/σ_cheap`, giving 0 = none, 0.2 = cheap reading, 0.8 = accurate reading | hardware profile (the nominal noise of each sensing level) |
| 7 | `measurement_age` | time since the stored reading was taken | `age / 24 h`, clipped; 1 if none | timer |
| 8 | `time_since_tx_success` | time since the last acknowledged uplink | `t / 24 h`, clipped; 1 if none | timer + radio ACK |
| 9 | `app_info_age` | the node's own estimate of the age of the application's information = (8) + the age the reading had when it was sent | `age / 24 h`, clipped; 1 if none | timer + radio ACK |
| 10 | `reported_value` | the value carried by the last acknowledged uplink (0 if none) | already in [0, 1] | RAM |
| 11 | `app_priority` | priority requested by the application: routine / elevated / urgent | `p / 2` → 0, 0.5, 1 | downlink message |
| 12 | `importance` | how much the stored reading matters: 1 at or below the critical threshold, `exp(−d / 0.10)` otherwise where `d` is the distance to the nearest threshold, 0 if no reading | already in [0, 1] | computed in firmware from the thresholds stored in flash |
| 13 | `link_quality` | EWMA (α = 0.2) of ACK outcomes: 1 = every recent uplink was acknowledged | already in [0, 1] | radio ACK |
| 14 | `path_loss_est` | estimate of the radio path loss, derived from the last ACK (see below); 1 = unknown or worst | `(PL − 110 dB) / 60 dB`, clipped | ACK signal-to-noise ratio (SNR) or a LoRaWAN `LinkCheckAns`, converted with the link-budget table in flash |
| 15 | `sense_low_cost` | energy of a cheap reading | `E / E_max`, clipped | hardware profile |
| 16 | `sense_high_cost` | energy of an accurate reading | `E / E_max`, clipped | hardware profile |
| 17 | `tx_cost` | energy of one uplink in the *standard* (reference) radio mode | `E / E_max`, clipped | hardware profile |

`ObservationBuilder(profile).describe()` prints this table from the code;
`ObservationBuilder.to_dict(obs)` maps a vector back to names.

### How the path-loss estimate (14) is formed

When an uplink in radio mode *k* is acknowledged, the ACK carries the *link margin* the
gateway measured (how many dB above its sensitivity the signal arrived), with 1 dB of
measurement noise. The node converts it into a path loss that does not depend on the mode:

    PL = tx_power_k − sensitivity_k − margin

so that it can then predict the margin of *any* mode as `tx_power_j − PL − sensitivity_j`.
When an uplink in mode *k* is **lost**, the node learns that the margin of that mode was
about zero or negative, i.e. the path loss is at least that mode's link budget
(`tx_power_k − sensitivity_k`): the estimate is raised to that value if it was lower. Before
the first ACK the component is 1 ("unknown").

### Two configuration switches that change what the node knows

* `CommunicationConfig.ack_available = False` (unconfirmed uplinks): the node never learns
  whether a message arrived. It then *assumes* delivery for its bookkeeping, so component 9
  becomes optimistic, component 13 stays at 1 and component 14 is never updated.
* `CommunicationConfig.priority_update_mode = "on_uplink"` (LoRaWAN class A realism): the
  application's priority reaches the node only together with an ACK, so component 11 can lag
  behind the application's actual priority. `info["app_priority"]` is what the application
  wants now, `info["node_priority"]` what the node has been told.

## Design choices

* **Past, not future, harvesting.** Components 1–2 describe the interval that already
  elapsed — what a harvester monitor can integrate — never the upcoming solar profile. The
  agent must infer the daily cycle from the time-of-day encoding and the recent history,
  exactly as a deployed node would have to.
* **The application's information age is estimated, not observed.** Component 9 is
  computed by the node from its own timers and ACKs. It is exact when uplinks are confirmed.
* **The link is seen only through the node's own measurements.** Component 14 is what a
  node can compute from an ACK and a table in flash; the true fading state stays hidden.
* **Energy costs are in the observation** (15–17) even though they are constant within an
  episode: under domain randomisation they change between episodes, so a policy trained with
  them learns to read its own cost profile instead of assuming the nominal one.
* **Ages are clipped at 24 h.** Beyond one day, "very old" is all the policy needs to know;
  clipping keeps every input bounded, which matters for neural networks and for fixed-point
  arithmetic on a microcontroller.

## Partial observability

The observation is *not* a complete description of the simulated world:

1. **Hidden environment state.** The true soil moisture, the daily clearness of the sky, the
   intra-day cloud process and the slow fading of the radio channel are hidden. The node has
   only a noisy, dated sample of the first, a noisy power measurement that mixes the next two,
   and a path-loss estimate as old as its last acknowledgement.
2. **Hidden application state.** External monitoring campaigns and the application's
   "unreported event" flag are hidden; only the resulting priority is visible.

In RL terms the problem is a **POMDP** (partially observable Markov decision process): the
agent must act on observations that do not fully determine the state. The observation
contains the standard, cheap devices that make a purely *reactive* policy (one that looks at
the current observation only) work well in practice:

* **summary statistics of the recent past** — the smoothed harvest (2), the smoothed ACK
  history (13), the last link measurement (14), the stored reading with its age and quality (5–7);
* **explicit timers** — the ages (7–9) turn "how long ago" into a state variable instead of
  requiring the policy to remember;
* **cyclic time** (3–4) — the solar and evapotranspiration cycles are functions of time of day.

If a learned policy needs more, two extensions keep the hardware contract intact: stacking
the last *k* observations (`rl.FrameStacker` / `rl.StackedPolicy`, `k·18` inputs, a ring
buffer in firmware), or a recurrent policy whose hidden state is carried across wake-ups
(`RecurrentPPO` from sb3-contrib). Both were tried in `examples/train_rl.ipynb`; neither
improved on the memoryless policy, which is consistent with the hidden processes being slow
(hours) compared with the 15-minute decision interval.
