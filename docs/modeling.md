# Models, utility, reward and the RL formulation

This document gives every model of the simulator with its formula and its default
constants, then the utility and reward, then the problem as a Markov decision process.
Constants live in `edgeengine_aware/config.py`; each is a dataclass field with a docstring
and can be changed without touching the simulator code.

## 1. Three layers of information

The simulator keeps three things apart, because conflating them is the most common way a
simulator becomes useless for deployment:

| layer | symbol | lives in | example |
|---|---|---|---|
| **true physical state** (hidden) | `x_t` | `agriculture.FieldEnvironment`, `energy.SolarEnergySource`, `communication.SimulatedLoRaRadio` | soil moisture θ = 0.41, irradiance, path loss 141 dB |
| **what the node knows** | `y_t` | `observation.NodeState` | last reading 0.43 ± 0.04 taken 45 min ago, 148 J stored, 2.1 mW harvested |
| **what the application knows** | `z_t` | `application.RemoteMonitoringApplication` | last delivered value 0.47, received 3 h ago |

The policy sees a normalised function of `y_t` only (`docs/observation.md`). The reward uses
`x_t` and `z_t`: the simulator is allowed to grade the application's knowledge against the
truth; the node is not allowed to peek at the truth. `docs/observation.md` has the same
distinction as a table of variables.

## 2. Physical models

Units are SI (seconds, joules, watts, °C, dB). Soil moisture is dimensionless: 1.0 means the
soil is at *field capacity* (holding as much water as it can after drainage). The default
constants describe a small node: 300 J of storage, a 5 mW-peak solar cell, 0.6 J per uplink.
One decision interval is Δt = 900 s (15 min); an episode is 7 days = 672 steps.

### 2.1 Energy storage (`energy.SimulatedEnergyStorage`)

The battery is an ideal energy buffer of capacity `E_max` = 300 J:

    E(t+Δt) = min( E(t) − load(t)  +  η · H(t) ,  E_max )      with  load(t) = B + S(a_t) + C(a_t)

* `B` = always-on power × Δt = 200 µW × 900 s = 0.18 J is drawn at every step (the
  microcontroller sleeps but does not switch off).
* `S`, `C` = energy of the executed sensing and transmission (0 if none, see the feasibility
  rule in `docs/actions.md`).
* `H` = energy harvested during the interval; `η` = `charge_efficiency` (1.0 by default, the
  only knob for conversion losses).
* The load is drawn **first** and clipped at 0: if `E(t) < load(t)` the node **browns out**
  (the always-on load could not be served; this is what `depleted` / `battery_depletion_events`
  count). The harvest is added **afterwards** and clipped at `E_max`; the part that does not
  fit is counted as *wasted*. Drawing the load before the harvest means the interval's own
  sunshine cannot pay for the interval's load — a conservative choice that slightly
  overstates brown-outs at dawn.
* The node never spends more than `E(t) − B − reserve` on optional loads, with
  `reserve = reserve_soc · E_max` = 2 % = 6 J (feasibility rule).

Temperature, ageing and self-discharge are not modelled; the node reads its stored energy
exactly. On hardware a fuel gauge replaces this class.

### 2.2 Solar harvesting (`energy.SolarEnergySource`)

    P(t) = P_max · shape(t) · clip(k_day + c(t), 0, 1) · η_conv

* `shape(t) = sin(π (h − 6) / 12)` between 06:00 and 18:00, 0 at night (`h` = hour of day):
  a half-sine day.
* `P_max` = 5 mW is the panel power at clear-sky noon; `η_conv` = 0.6 is the efficiency of
  the harvesting circuit (charger / maximum-power-point tracker).
* `k_day` is the **daily clearness** (1 = perfectly clear day). It follows an AR(1) process
  from one day to the next around a mean of 0.7 with standard deviation 0.2 and
  autocorrelation 0.5 (weather persistence), clipped to [0.05, 1].
* `c(t)` is an intra-day cloud perturbation, AR(1) per step with autocorrelation 0.85 and
  standard deviation 0.15 (cloud passages).

An AR(1) process is `x(t+1) = ρ x(t) + √(1−ρ²) ε`, with `ε` Gaussian: values decorrelate
over about `1/(1−ρ)` steps. The node measures `P(t)` with 5 % relative noise (component
`harvest_power` of the observation) and never sees `k_day` or `c(t)`. With the defaults an
average day yields about 55 J, a cloudy one 20–30 J; the always-on load alone is 17 J/day.

### 2.3 Soil and weather (`agriculture.FieldEnvironment`)

    θ(t+Δt) = clip( θ(t) − ET(t)·Δt + rain(t) + irrigation(t) + ε ,  0, 1 )

* **Evapotranspiration** `ET` (water lost to evaporation and plants) has a base rate of
  0.06 per day at the reference temperature (22 °C), increases by 3 % per °C above it, and
  follows the day/night cycle: 80 % of it is modulated by a half-sine that is zero at night and
  normalised so that the daily mean stays 0.06.
* **Rain** events follow a Poisson process (0.25 events per day, i.e. one every four days on
  average); each adds a uniform amount in [0.05, 0.25].
* **Irrigation** is applied by an *external* actor (the farmer's system), not by the node:
  when the true moisture falls below 0.20, a 0.30 irrigation arrives after a random delay
  (exponential, mean 6 h).
* `ε` is a small random walk (σ = 0.003 per step).
* **Temperature** is a daily cosine (mean 22 °C, amplitude 7 °C, peak at 15:00) with AR(1)
  noise; **relative humidity** is anti-correlated with the temperature anomaly. Neither is
  observed by the node; they are logged and plotted.
* Two thresholds define **water-stress zones**: below 0.35 = *warning*, below 0.25 =
  *critical*. A change of at least 0.05 in one step (rain, irrigation) or a zone crossing is
  an **environmental event** the application would like to hear about.

### 2.4 Sensing (`sensing.SimulatedSoilMoistureSensor`)

    reading = clip( θ + N(0, σ_level) + bias_level ,  0, 1 )

σ = 0.040 for the cheap level, 0.010 for the accurate level; bias 0 by default. The node
stores the reading with its timestamp and its **nominal** σ (a quality tag) — never the
realised error, which it cannot know.

### 2.5 Radio link (`communication.SimulatedLoRaRadio`)

The radio has `K` modes (default 3: *fast* / *standard* / *robust*, SF7 / SF9 / SF12-like,
all at 14 dBm transmit power, costing 0.3 / 0.6 / 1.2 J per uplink). An uplink in mode `k`
costs `E_k` whether or not it is delivered, and is delivered with probability

    margin_k(t) = P_tx − PL(t) − S_k                       [dB]
    PL(t)       = 139 dB + f_slow(t) + f_fast
    p_k(t)      = 1 / (1 + exp(−margin_k(t) / 1.5 dB))

* `S_k` is the gateway's receiver sensitivity for the mode: −123 / −129 / −137 dBm. A longer
  spreading factor can be decoded from a weaker signal (lower sensitivity) at the price of a
  longer transmission, hence more energy.
* `f_slow` is an AR(1) **shadowing** process (obstacles, vegetation, humidity): σ = 5 dB,
  autocorrelation 0.97 per step, i.e. a correlation time of about 8 h.
* `f_fast` is redrawn at every attempt: Gaussian, σ = 2 dB.
* With the defaults the mean margins are −2 / +4 / +12 dB, and, averaged over the fading, the
  long-run delivery ratios about 36 % / 75 % / 98 %.

If the uplink is delivered, the acknowledgement (**ACK**) tells the node so and carries the
margin the gateway measured, with 1 dB of noise; the node turns it into the path-loss
estimate of the observation. `ack_available = False` models unconfirmed uplinks: the node
learns nothing about delivery. Duty-cycle limits, collisions and gateway congestion are not
modelled; they would belong in this class.

### 2.6 The application (`application.RemoteMonitoringApplication`)

The application holds the last delivered value and knows when it arrived. Its **age of
information** (AoI) is the time since that arrival plus the age the reading already had when
it was sent (before the first delivery, the time since the episode started). From what it
knows it derives the **priority** it sends to the node:

* value below the warning threshold → *elevated*; below the critical threshold → *urgent*;
* AoI above 8 h → at least *elevated*; above 24 h → *urgent*;
* an external monitoring campaign (an agronomist asking for high-resolution data): Poisson
  arrivals at 0.2 per day, lasting 2–6 h, urgent in 30 % of the cases, elevated otherwise.

The node learns the priority either immediately (`priority_update_mode = "immediate"`) or
only when an uplink is acknowledged (`"on_uplink"`).

## 3. Utility and reward

### 3.1 Application utility (needs the truth — simulator only)

Let `θ` be the true moisture and `z` the application's last delivered value. (The tracking
utility uses the truth at the *end* of the step; the packet bonus, credited when the packet
arrives, uses the truth at the *start* of the step, when the reading was sent.) Information
near the stress thresholds matters more:

    crit(θ) = 1 + 2 · exp(−d(θ) / 0.08)        d = distance of θ to the nearest threshold
                                              (crit = 3 at or below the critical threshold, → 1 far away)

**Tracking utility, every step:**

    u_track = 0.10 · crit(θ) · exp(−|z − θ| / 0.05)

i.e. up to 0.1 per step (0.3 near a threshold) when the application's picture is exact,
decaying to 1/e of that when it is off by 0.05, and 0 before the first delivery.

**Packet bonus, when a packet with reading `ŷ` of age `a` is delivered:**

    u_pkt = crit(θ) · exp(−a / 2 h) · [ 0.30 · tanh(gain / 0.05) + 0.50 · event · exp(−|ŷ − θ| / 0.05) ]
    gain  = |z_before − θ| − |ŷ − θ|          (before the first delivery, |z_before − θ| is taken as 0.15)

`gain` is the *signed* reduction of the application's error: a packet that corrects a wrong
picture earns up to 0.3 (×3 near a threshold), a redundant one earns ≈ 0, a packet that
carries a *worse* value (a noisy cheap reading replacing an accurate one) is penalised.
`event` = 1 if a rain/irrigation event or zone crossing happened in the last 3 h and has not
been reported yet. The freshness factor makes sending old readings worth less.

Consequences worth understanding: a stale picture drifts away from `θ` → `u_track` decays →
a report pays; a rain event makes `z` suddenly wrong → large loss until a report; cheap
readings sit ~0.03 from `θ` on average → lower `u_track` than accurate ones; near the
thresholds everything is worth up to 3×.

### 3.2 Reward (`reward.RewardCalculator`)

    r = u_track + u_pkt
        − 0.10 · S/1 J − 0.10 · C/1 J                        energy costs (sensing, transmission)
        − 0.05 · w_prio · min(AoI / 6 h, 1)                  staleness, w_prio = 1 / 2 / 4 for routine / elevated / urgent
        − 0.50 · ((0.30 − SoC) / 0.30)²  if SoC < 0.30       battery risk (quadratic below 30 %)
        − 2.0 · [brown-out]                                  the always-on load could not be served
        − 0.20 · (number of rejected operations)
        − 0.02 · wasted harvest / 1 J

Every term is returned separately in `info["reward_components"]` (and accumulated in
`EpisodeMetrics.reward_components`). Orders of magnitude per 7-day episode with the
defaults: utility 100–140; energy costs 15–30 (an accurate reading plus a standard uplink
every hour costs 168 × 0.12 ≈ 20); staleness 5–15 for a reasonable policy, up to ~130 for a
node that never reports; battery risk 0 for a sustainable policy and several hundred for an
always-on one, which is what makes energy binding.

## 4. The decision problem

* **State** `s_t = (x_t, y_t, z_t)`: hidden world, node information, application
  information. Given the stored variables, the transition `s_{t+1} ~ P(· | s_t, a_t)` is
  Markov (all random processes are AR(1) or memoryless).
* **Observation** `o_t = ObservationBuilder(y_t) ∈ [0, 1]^18`, a deterministic function of the
  node's part of the state.
* **Action** `a_t ∈ {0, 1, 2} × {0, 1, …, K}`: sensing level, and no transmission or a radio
  mode.
* **Reward** `r_t = R(s_t, a_t, s_{t+1})` as above.
* **Episode** `T = 672` steps (7 days × 96), *truncated* at the horizon; it never terminates
  early unless `terminate_on_depletion` is set (default: a brown-out costs a penalty and the
  node keeps running once energy returns, as in the field). With a long horizon and no
  absorbing state, a discounted objective with γ ≈ 0.99 is the natural training objective.

Because `o_t ≠ s_t`, the agent faces a **partially observable MDP**: the truth, the weather
regime, the channel's fading state and the application's internal flags are hidden.
`docs/observation.md` explains what the observation does about it (summary statistics,
explicit timers, cyclic time) and the two standard extensions (frame stacking, recurrent
policies). In practice the observation is a reasonable approximate state: the hidden
processes are slow (hours) compared with Δt (15 min), and their effect on the reward passes
through quantities the node does measure.

## 5. Who sees what — summary

| quantity | simulator | policy | a real node |
|---|---|---|---|
| true soil moisture | ✔ | ✘ | ✘ |
| noisy reading + age + quality tag | ✔ | ✔ | ✔ |
| stored energy | ✔ | ✔ | ✔ (fuel gauge) |
| true / future irradiance | ✔ | ✘ | ✘ |
| harvested power over the past interval | ✔ | ✔ | ✔ (current monitor) |
| channel fading state / delivery probability | ✔ | ✘ | ✘ |
| path-loss estimate from the last ACK | ✔ | ✔ | ✔ (ACK SNR + table in flash) |
| ACK history (EWMA) | ✔ | ✔ | ✔ |
| application's age of information | ✔ | estimated | estimated from ACKs |
| application priority | ✔ | ✔ | ✔ (downlink) |
| external campaigns / unreported events | ✔ | ✘ | ✘ |

The policy column is contained in the real-node column by construction — both are built from
`NodeState`. That containment is the architectural condition for sim-to-real transfer.
