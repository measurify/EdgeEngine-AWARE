# Modelling assumptions and RL formulation

## 1. Five layers of information

EdgeEngine AWARE keeps five quantities apart on purpose; conflating them is the most common
way a simulator becomes useless for deployment.

| layer | symbol | lives in | example |
|---|---|---|---|
| **true physical state** | `x_t` | `agriculture.FieldEnvironment`, `energy.SolarEnergySource`, `communication.SimulatedLoRaRadio` | soil moisture θ = 0.41, irradiance, path loss 141 dB |
| **node-local information** | `y_t` | `observation.NodeState` | last sample 0.43 ± 0.04 taken 45 min ago, 148 J stored, 2.1 mW harvested |
| **application information** | `z_t` | `application.RemoteMonitoringApplication` | last delivered value 0.47, received 3 h ago |
| **application utility** | `u_t` | `application.RemoteMonitoringApplication.tracking()` / `.receive()` | how right the application's picture is, weighted by criticality |
| **RL reward** | `r_t` | `reward.RewardCalculator` | `u_t` minus energy costs and penalties |

The policy sees a normalised function of `y_t` only (`docs/observation.md`). The reward uses
`x_t` and `z_t`: the simulator is allowed to grade the application's knowledge against the
truth, the node is not allowed to peek at the truth.

## 2. Physical models

All quantities are in SI units (s, J, W); soil moisture is dimensionless, normalised to field
capacity. Defaults describe a small node (300 J storage, 5 mW-peak cell, 0.6 J per uplink);
every value is a `dataclass` field meant to be replaced by a measurement.

**Energy storage** (`energy.SimulatedEnergyStorage`) — an ideal buffer:

    E(t+1) = clip( E(t) + H(t) − B − S(a_t) − C(a_t), 0, E_max )

`B = P_baseline·Δt` is always drawn (the MCU sleeps but does not switch off); `S` and `C` are
drawn only if the feasibility rule allowed them. Consumption is drawn first and the harvest
of the interval is stored afterwards (the interval's own harvest cannot pay for its load — a
conservative choice that slightly overstates brown-outs at dawn). Harvest that does not fit
is counted as *wasted*. Temperature and ageing are omitted; `EnergyStorageConfig.charge_efficiency`
(default 1) is the single knob for conversion losses. A fuel-gauge reading replaces this class
on hardware.

**Harvesting** (`energy.SolarEnergySource`):

    P(t) = P_max · sin(π (h − h_rise)/(h_set − h_rise))⁺ · clip(k_day + c(t), 0, 1) · η

`k_day` is a daily clearness index following an AR(1) process across days (weather
persistence), `c(t)` an intra-day AR(1) cloud perturbation. The node measures `P(t)` with 5 %
relative noise; it never sees `k_day` or `c(t)`.

**Field** (`agriculture.FieldEnvironment`):

    θ(t+1) = clip( θ(t) − ET(t)Δt + rain(t) + irrigation(t) + ε, 0, 1 )

Evapotranspiration is a base rate modulated by temperature anomaly and by the day/night
cycle; rain is a Poisson process with uniform amounts; irrigation is applied by an *external*
actor with a random delay once the true moisture falls below a trigger (the node does not
control it). Temperature is a daily cosine with AR(1) noise, humidity is anti-correlated with
it. Two thresholds (warning 0.35, critical 0.25) define water-stress zones; a change of more
than 0.05 in one step or a zone crossing is an *environmental event*.

**Sensing** (`sensing.SimulatedSoilMoistureSensor`): `ŷ = clip(θ + N(0, σ_level) + bias, 0, 1)`.
The node stores the sample with its timestamp and a *quality tag* (nominal σ), never the
realised error.

**Communication** (`communication.SimulatedLoRaRadio`): the radio has `K` modes (default 3:
fast / standard / robust, i.e. SF7 / SF9 / SF12-like at 14 dBm, costing 0.3 / 0.6 / 1.2 J per
uplink). An uplink in mode `k` costs `E_k` whether or not it is delivered and is delivered with

    margin_k(t) = P_tx,k − PL(t) − S_k          PL(t) = PL_0 + f_slow(t) + f_fast
    p_k(t)      = 1 / (1 + exp(−margin_k(t) / 1.5 dB))

where `S_k` is the gateway sensitivity of the mode, `f_slow` an AR(1) shadowing process
(σ = 5 dB, correlation ≈ 8 h) and `f_fast` a per-attempt Gaussian (σ = 2 dB). With the defaults
the mean margins are −2 / +4 / +12 dB. An ACK tells the node about delivery and carries the
measured margin (`ack_available`; with unconfirmed uplinks the node assumes delivery and its
information-age estimate becomes optimistic); the node converts the margin into a path-loss
estimate valid for every mode, and raises that estimate to a mode's link budget whenever an
uplink in that mode is lost. Duty cycle, collisions and gateway congestion are not modelled —
they belong in this class.

**Application** (`application.RemoteMonitoringApplication`): holds the last delivered value
and its age (AoI). Derives the *priority* it sends to the node from what it knows: value
below warning → elevated, below critical → urgent; AoI beyond 8 h / 24 h → elevated / urgent;
external campaigns (Poisson, 2–6 h) → elevated or urgent. The node learns the priority either
immediately or only with an ACK (`priority_update_mode`).

## 3. Utility and reward

**Tracking utility (every step)** — the application's picture is graded against the truth:

    u_track = w_track · crit(θ) · exp(−|z − θ| / e_scale),   crit(θ) = 1 + g · exp(−dist(θ, thresholds)/s)

**Packet bonus (on delivery)** — shaping that credits the packet that fixes the picture:

    u_pkt = crit(θ) · exp(−age_sample/τ_f) · [ w_g · tanh(gain / g_scale) + w_e · event · exp(−|ŷ − θ|/e_scale) ]
    gain  = |z_before − θ| − |ŷ − θ|      (signed)

Properties worth teaching: a redundant report has `gain ≈ 0` and does not change `u_track`
→ pure cost; a stale belief drifts away from θ → `u_track` decays → a report pays; a rain event
makes `z` suddenly wrong → large loss until a report; low-cost samples sit ~0.03 from θ →
lower `u_track` than high-quality ones; near thresholds everything is worth up to 3×.

**Reward** (`reward.RewardCalculator`):

    r = u_track + u_pkt
        − λ_s S/E_ref − λ_c C/E_ref                                  (energy, λ = 0.1, E_ref = 1 J)
        − λ_stale · w_prio · min(AoI/τ_stale, 1)                     (application freshness request)
        − λ_batt · ((soc_safe − soc)/soc_safe)²⁺ − λ_depl·[brown-out]  (energy risk)
        − λ_rej · #rejected − λ_waste · wasted/E_ref                 (infeasible actions, wasted sun)

Every term is returned in `info["reward_components"]`. Order of magnitude per 7-day episode
with defaults: utility 100–120, energy costs 15–30, staleness 5–15, battery risk 0 for a
sustainable policy and hundreds for an always-on one.

## 4. MDP formulation

* **State** `s_t = (x_t, y_t, z_t)`: hidden world, node information, application information.
  The transition `s_{t+1} ~ P(·|s_t, a_t)` is Markov (all processes are AR(1) or memoryless
  given the stored variables).
* **Observation** `o_t = ObservationBuilder(y_t) ∈ [0,1]^18` — a deterministic function of the
  node-local part of the state.
* **Action** `a_t ∈ {0,1,2} × {0,1,…,K}` (sensing level; no transmission or radio mode).
* **Reward** `r_t = R(s_t, a_t, s_{t+1})` as above.
* **Episode** `T = 672` steps (7 days × 96 steps), *truncated*, never terminated unless
  `terminate_on_depletion` is set. Long horizon, no absorbing state: an infinite-horizon
  discounted objective (γ ≈ 0.99) is the natural training objective.

Because `o_t ≠ s_t`, the agent faces a **POMDP**: the truth, the weather regime, the fading
state of the channel and the application's internal flags are hidden. `docs/observation.md` §"Markov
property" lists what is done about it (EWMA statistics, explicit timers, cyclic time) and the
two standard extensions (frame stacking, recurrent policies). In practice the observation is a
reasonable approximate state: the hidden processes are slow (hours) relative to Δt (15 min)
and their effect on the reward is mediated by quantities the node does measure.

## 5. The simulator/agent/hardware distinction, once more

| | simulator state | agent observation | hardware-measurable state |
|---|---|---|---|
| true soil moisture | ✔ | ✘ | ✘ |
| noisy sample + age + quality tag | ✔ | ✔ | ✔ |
| stored energy | ✔ | ✔ | ✔ (fuel gauge) |
| true / future irradiance | ✔ | ✘ | ✘ |
| measured harvesting power (past interval) | ✔ | ✔ | ✔ (current monitor) |
| delivery probability / fading state | ✔ | ✘ | ✘ |
| path-loss estimate from the last ACK (+ loss floors) | ✔ | ✔ | ✔ (ACK SNR + flash table) |
| ACK history (EWMA) | ✔ | ✔ | ✔ |
| application AoI | ✔ | estimated (✔) | estimated from ACKs |
| application priority | ✔ | ✔ | ✔ (downlink) |
| external requests / unreported events | ✔ | ✘ | ✘ |

The middle column is a subset of the right column — by construction, since both are built
from `NodeState`. That containment is the architectural condition for sim-to-real transfer.
