# EdgeEngine AWARE

**Application- and Energy-Aware Simulation Environment for Reinforcement Learning in
Energy-Harvesting Edge IoT Systems**

EdgeEngine AWARE is a Python simulator, packaged as a standard
[Gymnasium](https://gymnasium.farama.org/) environment, of a small **energy-harvesting
embedded node** that monitors a physical quantity and reports it to a remote application over
a low-power radio link. It exists to **train and compare decision policies** — hand-written
rules or reinforcement-learning (RL) agents — that decide, every 15 minutes, how to spend a
very small and uncertain energy budget. Three application domains are built in: a
solar-powered soil-moisture node in a field (the reference case used throughout this README),
a CO₂ node in a classroom powered by the ceiling lights, and a machine-monitoring node powered
by the heat of the motor it watches (§10).

The project also contains what is needed *after* training: a frozen export format for
policies, a firmware-style controller, a C runtime that reproduces the Python decision
logic on a microcontroller, and a way to replay recorded weather data through the same
simulator.

This README is written for a reader who knows Python but not necessarily RL, IoT
hardware or radio engineering. Terms in *italics* on first use are defined in the
[glossary](#glossary) at the end.

---

## Contents

1. [The problem in plain words](#1-the-problem-in-plain-words)
2. [The one design rule](#2-the-one-design-rule-the-policy-must-not-cheat)
3. [Installation](#3-installation)
4. [Quick start](#4-quick-start)
5. [What happens in one step](#5-what-happens-in-one-step)
6. [Observation, action, reward, metrics](#6-observation-action-reward-metrics)
7. [Configuration, scenarios and domain randomisation](#7-configuration-scenarios-and-domain-randomisation)
8. [Built-in policies](#8-built-in-policies)
9. [Training and comparing RL agents](#9-training-and-comparing-rl-agents)
10. [Three application domains](#10-three-application-domains)
11. [Recorded traces instead of models](#11-recorded-traces-instead-of-models)
12. [From simulation to a microcontroller](#12-from-simulation-to-a-microcontroller)
13. [Notebooks, scripts and tools](#13-notebooks-scripts-and-tools)
14. [Repository layout](#14-repository-layout)
15. [Tests](#15-tests)
16. [Documentation](#16-documentation)
17. [What is not modelled](#17-what-is-not-modelled)
18. [Glossary](#glossary)
19. [Citation](#citation)

---

## 1. The problem in plain words

Imagine a matchbox-sized device stuck in the soil of a field:

* It has a **soil-moisture sensor**. Reading it costs energy. A quick, cheap reading is
  noisy; a careful reading (longer warm-up, averaging) is accurate but costs six times more.
* It has a **radio** that can send the latest reading to a gateway several hundred metres
  away. Sending costs energy too, and the radio has three settings: *fast* (cheap, but the
  message often gets lost when the link is poor), *standard*, and *robust* (expensive, almost
  always gets through). Whether a message arrived is known from an acknowledgement (*ACK*).
* It has a **small battery** (300 J, about a 25 mAh cell) charged by a **tiny solar cell**
  that yields about 55 J on an average day and much less on a cloudy one. Just staying alive
  (sleeping, keeping the clock running) costs about 17 J per day.
* A remote **application** (think: the farmer's dashboard) wants to know the soil moisture
  — accurately, and quickly when it matters, i.e. when the soil is getting dry and
  irrigation decisions are due. It tells the node how urgent its needs are (a *priority*
  level: routine, elevated, urgent).

Every 15 minutes the node wakes up and must decide two things: **how well to look** (do not
sense / cheap reading / accurate reading) and **whether and how loudly to speak** (do not
transmit / transmit fast / standard / robust). The score it is judged on is *how right and
how fresh the application's picture of the field is*, minus the energy it spends and minus
penalties for draining the battery. This is a sequential decision problem under uncertainty
— exactly what reinforcement learning is for — but also one where a well-designed set of
rules does very well, which is why the project ships both and compares them honestly.

## 2. The one design rule: the policy must not cheat

A simulator knows everything: the true soil moisture, tomorrow's clouds, whether the next
radio message will get through. A real device knows almost nothing of this. If a policy is
trained on information that the device will not have, it will not work in the field.

EdgeEngine AWARE therefore separates three kinds of quantities and enforces the separation
in code and tests:

| kind | examples | who may use it |
|---|---|---|
| **hidden ground truth** (simulator only) | true soil moisture, true irradiance, radio channel state, future rain | the world dynamics, the reward, the plots |
| **measurable on the node** | stored energy, harvested power in the last interval, the noisy sensor reading and its age, whether the last message was acknowledged, the priority received from the application | the node, i.e. `NodeState` |
| **policy input** | 18 numbers in [0, 1], all computed from the measurable quantities | the policy |

The code that turns measurable quantities into the 18-number *observation*
(`observation.py`), and the code that decides which requested action is affordable
(`actions.plan_execution`), are **shared** between the simulator (`env.py`), the
firmware-style controller (`deployment.NodeController`) and the C runtime (`firmware/`). The
same policy object therefore runs unchanged in all three places. A test
(`tests/test_env.py::test_observation_contains_no_privileged_information`) changes every
hidden variable and checks that the observation does not move.

## 3. Installation

Requirements: Python 3.11 or newer. A C compiler (`cc`/`gcc`/`clang`) is optional and only
needed for the firmware equivalence tests.

```bash
git clone https://github.com/measurify/EdgeEngine-AWARE.git
cd EdgeEngine-AWARE
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -e ".[dev]"      # numpy, gymnasium, matplotlib + pytest, jupyter, nbformat, nbconvert
pip install -e ".[rl]"       # only for RL training: stable-baselines3, sb3-contrib, torch (CPU is enough)
pytest                       # 129 tests, 15-60 s depending on the installed extras
```

The `-e` flag installs the package in "editable" mode: edits to the source are picked up
without reinstalling.

## 4. Quick start

**Create the environment and take one step.** `gym.make` returns the environment wrapped in
Gymnasium's standard wrappers; `env.unwrapped` is the underlying `EdgeEngineAwareEnv`.

```python
import gymnasium as gym
import edgeengine_aware as ea            # importing the package registers "EdgeEngineAware-v0"

env = gym.make("EdgeEngineAware-v0")
obs, info = env.reset(seed=0)             # obs: 18 float32 values in [0, 1]
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

print(env.unwrapped.obs_builder.describe())   # what each of the 18 numbers means, from code
print(info["reward_components"])              # utility, energy costs and penalties, separately
print(info["metrics"]["average_aoi_s"])       # running episode metrics (here: mean age of information, s)
```

**Run the rule-based controller for a whole 7-day episode** and print the metrics. The
controller needs the *node profile* (the hardware constants: energy per operation, radio link
budget table) to choose the radio mode; without it, it always uses the standard mode.

```python
import edgeengine_aware as ea
from edgeengine_aware.policies import RuleBasedPolicy, run_episode

cfg = ea.default_config()
env = ea.EdgeEngineAwareEnv(cfg, render_mode="human")       # "human" opens a live Matplotlib dashboard
profile = ea.NodeProfile.from_config(cfg)
result = run_episode(env, RuleBasedPolicy(profile=profile), seed=1, render=True, render_every=96)
print(env.metrics.summary())
```

Typical output (seed 1, default configuration): total reward ≈ 106, 185 transmissions of
which 160 delivered (86 %), minimum battery level 47 %, no brown-out, mean age of
information at the application 0.86 h.

**Compare the baselines without graphics:**

```bash
python examples/compare_policies.py --seeds 10
```

**Open the guided notebook** (environment tour, one episode step by step, metrics, policy
comparison):

```bash
jupyter notebook examples/baseline_policy.ipynb
```

## 5. What happens in one step

One call to `env.step(action)` advances the world by one decision interval (15 minutes by
default) in this order:

1. **Feasibility.** Using only the energy it can read from its battery gauge, the node sets
   aside the always-on consumption of the coming interval and a small brown-out reserve
   (2 % of capacity) and checks which requested operations it can afford. Sensing is served
   first, then transmission. Unaffordable operations are *rejected*: not executed, no energy
   spent, a small penalty. A transmission with nothing to send (no stored reading) is also
   rejected. See `docs/actions.md`.
2. **Sensing.** If requested and affordable, the sensor samples the *true* soil moisture with
   the noise of the chosen level; the reading is stored on the node together with its
   timestamp and a quality tag.
3. **Transmission.** If requested and affordable, the latest stored reading is sent with the
   chosen radio mode. Energy is spent whether or not the message arrives. Delivery is random,
   with a probability that depends on the radio mode and on the current (hidden) state of the
   link. On delivery the application updates its picture of the field; the node learns the
   outcome from the acknowledgement.
4. **Energy update.** Consumption (always-on + sensing + transmission) is drawn from the
   battery first, then the energy harvested during the interval is added, capped at the
   capacity (excess is counted as *wasted*). If the always-on load itself cannot be served
   the node *browns out*.
5. **World update.** Clock, soil, weather, radio channel and application priority move to
   the next interval.
6. **Reward.** How good the application's picture now is (compared with the new truth), plus
   a bonus for the packet that improved it, minus energy costs and penalties.
7. **Observation.** The node's bookkeeping is refreshed with measurable quantities only and
   the 18-number vector for the next decision is built.

The full timeline, with the formulas, is in `docs/modeling.md`.

## 6. Observation, action, reward, metrics

**Observation** — `Box(0, 1, (18,), float32)`. Every component is normalised to [0, 1] and
comes from something a microcontroller can measure or compute:

| group | components |
|---|---|
| energy | battery *SoC*; harvested power in the last interval; smoothed recent harvest (*EWMA*) |
| time | time of day as sine and cosine |
| stored reading | value; quality tag (0 none / 0.2 cheap / 0.8 accurate); age |
| communication | time since the last acknowledged uplink; the node's own estimate of how old the application's information is; the value it last delivered; smoothed fraction of recent uplinks acknowledged (*link quality*); *path-loss* estimate from the last ACK |
| application | requested priority (0 / 0.5 / 1) |
| node-side judgement | *importance* of the stored reading (how close it is to the stress thresholds) |
| hardware profile | energy of a cheap reading, of an accurate reading, of one standard uplink, all relative to the battery capacity |

Full table with normalisation formulas and hardware sources: `docs/observation.md`.

**Action** — `MultiDiscrete([3, 4])`, i.e. a pair of integers:

| component | values |
|---|---|
| `sensing_level` | 0 = do not sense, 1 = cheap reading (0.10 J, noise σ 0.04), 2 = accurate reading (0.60 J, σ 0.01) |
| `transmit` | 0 = do not transmit, 1 = fast mode (0.30 J), 2 = standard (0.60 J), 3 = robust (1.20 J) |

For algorithms that need a single integer (DQN, tabular methods) `rl.FlatActionWrapper`
exposes the same choices as `Discrete(12)` (`12 = 3 × 4`; it grows if you configure more
radio modes). See `docs/actions.md`.

**Reward** — a sum of separately reported terms, available in `info["reward_components"]`:

| term | sign | meaning |
|---|---|---|
| tracking utility | + | every step: how close the application's last value is to the true moisture, weighted up to 3× near the stress thresholds |
| packet bonus | + | on delivery: signed improvement of the application's error, plus a bonus for reporting a recent rain/irrigation event |
| energy costs | − | 0.1 per joule spent on sensing and on transmission |
| staleness | − | grows with the age of the application's information, weighted 1/2/4 by priority, saturating after 6 h |
| battery risk | − | quadratic penalty below 30 % SoC; 2.0 per step of brown-out |
| rejected actions, wasted harvest | − | 0.2 per rejected operation; 0.02 per joule of harvest that did not fit in the battery |

Orders of magnitude for a 7-day episode with the defaults: utility 100–140, energy costs
15–30, a policy that never reports loses ~130 to staleness, an always-on policy loses several
hundred to battery penalties. Formulas and constants: `docs/modeling.md` §3.

**Metrics** — `env.metrics` (an `EpisodeMetrics`) accumulates per episode: harvested /
consumed / always-on / sensing / communication / wasted energy [J]; number of readings,
accurate readings, transmissions, deliveries (also per radio mode), rejected operations;
average and minimum SoC, fraction of steps below 30 % SoC, brown-out events; average and
maximum *age of information* (AoI); total utility and reward. `env.metrics.summary()` prints
them, `info["metrics"]` carries the same dictionary at every step, and `env.log` (an
`EpisodeLog`) keeps the per-step history used by the plots.

**Other `info` keys.** `ground_truth` (hidden state, for evaluation only — never feed it to
a policy), `rejected` (list of reasons), `tx_mode`, `tx_success`, `app_priority` (what the
application wants now) vs `node_priority` (what the node has been told), `utility_breakdown`,
`observation_names`.

## 7. Configuration, scenarios and domain randomisation

Everything is a plain Python dataclass; the environment code never needs to change.

```python
import edgeengine_aware as ea

cfg = ea.default_config()
cfg.time.timestep_s = 600                     # 10-minute decisions
cfg.time.episode_days = 14
cfg.storage.capacity_j = 800
cfg.storage.initial_soc_range = (0.3, 0.9)    # random initial charge at every reset
cfg.mcu.baseline_power_w = 120e-6
cfg.harvesting.max_power_w = 0.008            # panel peak power [W] at clear-sky noon
cfg.observation.harvest_ref_power_w = 0.005   # keep the harvest normalisation consistent with the panel
cfg.harvesting.clearness_mean = 0.5           # cloudier climate
cfg.communication = ea.config.single_mode_radio(energy_j=0.9)   # a radio with one setting -> binary transmit action
cfg.communication.priority_update_mode = "on_uplink"   # priority only arrives with an ACK (LoRaWAN class A)
env = ea.EdgeEngineAwareEnv(cfg)
```

Two things to know: `cfg.validate()` is called by the environment and rejects inconsistent
settings with a clear message; and the observation normalisation constants
(`cfg.observation`) are part of the hardware profile — if you change the panel or the
battery, check that `harvest_ref_power_w` (the power that maps to 1.0) still makes sense.

**Scenarios** (`edgeengine_aware.scenarios`) are named configurations used as the benchmark
protocol. Each stresses one aspect:

| name | what changes with respect to `default` | what it tests |
|---|---|---|
| `default` | — (mixed weather, clearness 0.7 ± 0.2) | the nominal regime |
| `cloudy_week` | clearness 0.35 ± 0.15, more cloud variability, battery starts at 40 % → ~30 J/day (55 % of default) | rationing energy over several dark days |
| `tiny_battery` | 150 J instead of 300 J | smoothing consumption with a small buffer |
| `lossy_link` | +6 dB mean path loss, slower fading → standard mode delivers ~40 %, robust ~85 % | choosing the radio mode from the link estimate |
| `drought` | evapotranspiration ×2, almost no rain, irrigation delayed 18 h, drier start | tracking the approach to the stress thresholds |
| `demanding_application` | 1.5 external campaigns/day of 2–8 h, half of them urgent, "elevated" after 4 h of silence | following the priority, saving in between |

`ea.get_scenario(name, randomize=False)` returns the configuration; `rl.make_env(name)` the
environment; `rl.make_env("mixed")` (or a list of names) an environment that draws a scenario
at every reset (`MixedScenarioEnv`) — the recommended training distribution.

**Domain randomisation** (`cfg.randomization`, off by default) perturbs the physical
parameters at every `reset()` so that a trained policy does not over-fit one exact device:
sensor noise ×[0.7, 1.5], sensing energy ×[0.8, 1.3], radio energies ×[0.8, 1.3], mean path
loss ±3 dB, solar intensity ×[0.6, 1.2], cloud variability ×[0.5, 1.5], battery capacity
×[0.8, 1.2], always-on power ×[0.7, 1.5] (all uniform). The randomised energy costs are what
the node reports in its observation, exactly as a real device would report its own measured
profile. Enable with `cfg.randomization.enabled = True` or `get_scenario(name, randomize=True)`.

## 8. Built-in policies

All policies implement the tiny `Policy` protocol — `act(observation) -> action` and
`reset()` — and only ever see the 18-number observation, so they run identically in the
simulator, in the deployment controller and (for the rule-based one) in C.

| policy | what it does |
|---|---|
| `RuleBasedPolicy(params, profile)` | the interpretable reference controller, see below |
| `PeriodicPolicy(period_steps, sensing_level, tx)` | the classic duty-cycled firmware: sense and transmit every *n* steps, regardless of energy or application |
| `RandomPolicy(seed)` | uniform random actions — a lower bound |
| `AlwaysOnPolicy()` | accurate reading and transmission at every step — an upper bound on information, a disaster for the battery |

**The rule-based controller** (`policies.RuleBasedPolicy`, parameters in `RuleBasedParams`)
applies these rules in order, all on the normalised observation:

1. *Deep economy* — battery below 20 %: one accurate report every 8 h (every 30 min if the
   application is urgent), nothing else.
2. *Retry* — a fresh accurate reading (younger than 18 min) that was not acknowledged is
   re-sent, unless the link looks down (link quality < 0.5).
3. *Scheduled report* — when the estimated age of the application's information exceeds the
   report interval: 2 h / 1 h / 30 min for routine / elevated / urgent priority. In *economy
   mode* (battery below 50 %, unless the priority is urgent) the interval is doubled; at
   routine priority it is shortened to 75 % when the battery is above 70 % or the sun is
   strong. A report is an accurate reading plus a transmission.
4. *Event report* — the stored reading differs from the last delivered one by more than 0.08,
   or is near a stress threshold and differs by more than 0.03.
5. *Check* — outside economy mode, a cheap reading every hour (no transmission).
6. Otherwise sleep.

Radio mode: the cheapest mode whose expected link margin, computed from the node's path-loss
estimate and the link-budget table in the profile, is at least 4 dB; one mode more robust
when recent uplinks failed (link quality < 0.7); the standard mode before the first estimate.

## 9. Training and comparing RL agents

The environment follows the Gymnasium API, so any library works. The project uses
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) (SB3): PPO and A2C accept the
`MultiDiscrete` action space directly; DQN needs `rl.FlatActionWrapper`. No RL algorithm is
implemented inside the package on purpose — the package is the benchmark, the notebook is
the experiment.

```python
import edgeengine_aware as ea
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from edgeengine_aware.rl import make_env_fn, SB3Policy, evaluate, summarize
from edgeengine_aware.scenarios import SCENARIOS

# train on a mixture of all scenarios with domain randomisation, 8 parallel copies
vec_env = DummyVecEnv([make_env_fn("mixed", randomize=True, seed=i) for i in range(8)])
model = PPO("MlpPolicy", vec_env, policy_kwargs=dict(net_arch=[64, 64]), device="cpu").learn(2_000_000)

# evaluate on every scenario, 20 held-out seeds, against the rule-based controller
profile = ea.NodeProfile.from_config(ea.default_config())
rows = evaluate(
    {"PPO": lambda: SB3Policy(model), "rule-based": lambda: ea.RuleBasedPolicy(profile=profile)},
    scenarios=SCENARIOS, seeds=range(1000, 1020),
)
print(summarize(rows, "reward"))     # {scenario: {policy: (mean, std)}}
```

**Evaluation protocol** (`rl.evaluate`): deterministic policies, 7-day episodes, the six
scenarios with nominal (non-randomised) physics, seeds ≥ 1000 that training never uses; one
`EvalRow` per episode with reward, utility, energy, counts, min SoC, brown-outs, AoI and the
reward components. `evaluate(..., envs={label: env})` evaluates on ready-made environments
instead (e.g. trace-driven ones, §11, or other domains, §10).

**What we found** (`examples/train_rl.ipynb`, details in the notebook): with 1–2 M training
steps PPO learns a sensible policy — including using the robust radio mode when the link is
poor and never the fast one — but the link-aware rule-based controller is as good or better
on the nominal, lossy-link and small-battery scenarios; PPO wins on the energy-limited
`cloudy_week`. With **5 M steps** (three seeds, `examples/run_long_training.sh`) PPO reaches
parity with the rule-based controller on the nominal, small-battery, lossy-link and
demanding-application scenarios and is clearly ahead on `cloudy_week` (+6) and `drought`
(+8): 82.0 ± 1.8 vs 81.1 averaged over the six scenarios. DQN is clearly below PPO at every
budget tried, with a higher variance across seeds, and wastes uplinks in the fast mode. Adding
memory (4-frame stacking or a recurrent PPO) does not help. Repeating training over several
seeds (`examples/train_seeds.py`) gives the mean ± std that a paper should quote.

## 10. Three application domains

The agricultural node is one instance of a general problem: an embedded node with a small
energy store, a harvester and a duty to report. `edgeengine_aware.domains` makes the hidden
world pluggable and ships three domains. Whatever the domain, the node sees the **same
18-number observation** and returns the **same action**: the monitored quantity is normalised
to [0, 1], and the only new profile constant is on which side the danger lies
(`critical_is_upper`). The rule-based controller, the RL tooling, the exported bundles and the
C runtime therefore work unchanged in every domain — and a policy trained in one domain can be
evaluated in another.

| domain | monitored quantity (danger) | energy source | radio | storage / always-on | what drives the week |
|---|---|---|---|---|---|
| `agriculture` | soil moisture (low) | solar cell, weather | LoRa-like, 3 spreading factors | 300 J / 200 µW | sun, clouds, rain, irrigation |
| `indoor_air` | CO₂ of a classroom (high: 1000 / 1500 ppm) | indoor PV under the ceiling lights, plus a window | BLE-like, 3 PHY modes, sub-millijoule uplinks | 60 J / 20 µW | occupancy 08–18 on weekdays: people raise the CO₂ *and* switch the lights on; nights and weekends bring neither energy nor relevance; the CO₂ sensor (20 / 150 mJ per reading) is the energy hog |
| `industrial` | bearing temperature of a motor (high: 70 / 90 °C) | thermoelectric generator on the warm casing | LoRa-like | 120 J / 200 µW | two shifts 06–22 on weekdays: the machine's heat is both the energy source and the monitored variable; faults make it run hotter until maintenance; weekends are cold and dark |

```python
import edgeengine_aware as ea
from edgeengine_aware.rl import make_env

env = ea.EdgeEngineAwareEnv(ea.domain_config("industrial"))      # a domain's default configuration
env = make_env("indoor_air:no_window")                          # a scenario of another domain ("domain:scenario")
env = make_env("all")                                           # training mixture over every scenario of every domain
print(ea.scenario_names("all"))                                 # 6 agricultural + 5 + 5 scenarios
```

Each domain has its own stress scenarios (`indoor_air`: `no_window`, `weak_ventilation`,
`long_hours`, `dim_lights`; `industrial`: `single_shift`, `degrading`, `continuous`,
`weak_link`). `examples/domains.ipynb` puts the three side by side and asks the cross-domain
question. First answer: the 5 M-step PPO trained on the agricultural scenarios is at parity
with the rule-based controller in its own domain but **loses 21 points on the industrial and
33 on the indoor domain** — it learned the solar day, not only the trade-off.
`examples/run_cross_domain.sh` trains one policy per domain and a *universal* one on the
mixture of all three; the notebook builds the train-domain × test-domain matrix from those
runs. Models and constants of the two new domains: `docs/modeling.md` §6.

## 11. Recorded traces instead of models

The weather and soil dynamics of the simulator are *stochastic models*. To check a policy
against data those models never produced, `edgeengine_aware.traces` replays recorded hourly
series through the same environment:

* `Trace` — a time-indexed table loaded from CSV. Each column is either *instant* (soil
  moisture, temperature: linearly interpolated) or a *preceding-interval mean* (irradiance
  in W/m², precipitation in mm — the convention of weather archives, held constant over the
  hour that ends at the timestamp). Time is in seconds since the first local midnight.
* `TraceSolarEnergySource` — panel power = `max_power_w × efficiency × irradiance / 1000 W/m²`.
* `TraceFieldEnvironment` — replays soil moisture (volumetric m³/m³ divided by the field
  capacity, 0.40 by default, to get the simulator's "fraction of field capacity"),
  temperature, humidity and rain as the hidden ground truth.
* `TraceDrivenEnv(trace, config, start_day=None, harvesting=True, field=True, ...)` — an
  `EdgeEngineAwareEnv` whose harvesting and/or field come from the trace. Each `reset()`
  replays a 7-day window starting at a random admissible day (or at `start_day`, or at
  `reset(options={"start_day": k})`); `info["trace_start_day"]` reports it. Spaces,
  observation and reward are unchanged: **the policy cannot tell the two worlds apart.**

```python
import edgeengine_aware as ea
from edgeengine_aware.traces import Trace, TraceDrivenEnv

trace = Trace.from_csv("data/traces/demo_liguria_2023_hourly.csv")
print(trace.describe())
summer = trace.slice_days(151, 92)                     # days 151-243 of the year
env = TraceDrivenEnv(summer, ea.default_config())      # random summer weeks at every reset
obs, info = env.reset(seed=0)
```

**Data.** `data/traces/albenga_2023_hourly.csv` is the real ERA5 / ERA5-Land reanalysis for
a coastal site in Liguria (year 2023, hourly), downloaded with `tools/fetch_open_meteo.py`
(Open-Meteo archive, licence CC BY 4.0; the script needs only the standard library and works
for any site and year). `data/traces/demo_liguria_2023_hourly.csv` is a clearly labelled
**synthetic** stand-in with a darker winter, generated by `tools/make_demo_trace.py`. Your own
logger data works with the same CSV format (`data/traces/README.md`).
`examples/trace_driven.ipynb` compares the trace statistics with the synthetic scenarios and
evaluates the baselines and the exported PPO policies season by season: on the real trace the
PPO policies trained on the synthetic mixture behave as on the synthetic scenarios — the
2 M-step policy about 3 points below the rule-based controller, the 5 M-step policy at parity
while sending fewer packets — with no brown-outs in winter; on the darker synthetic trace the
2 M policy fails in winter, the regime the training scenarios do not contain.

## 12. From simulation to a microcontroller

Three pieces make a trained policy deployable:

1. **`deployment.PolicyBundle`** — one JSON file that freezes everything the device must
   reproduce: the observation names in order and their normalisation formulas, the node
   profile (all constants), the action encoding, the model parameters (thresholds for the
   rule-based policy, weights for a neural network) and metadata. `export_policy(...)` creates
   it; `rl.export_sb3_mlp(model)` extracts the weights of an SB3 network as plain lists.
   Three bundles are shipped in `examples/bundles/`: `rule_based_default.json`,
   `ppo_default.json` (PPO, 2 M steps, network 18→64→64→7) and `ppo_long.json` (PPO, 5 M
   steps, the best of three seeds).
2. **`deployment.NodeController`** — the firmware main loop written in Python against the
   hardware protocols of `interfaces.py` (clock, energy storage, energy source, sensor,
   radio, application). It reuses the simulator's tracker, observation builder and
   feasibility rule. `deployment.make_mock_backend(profile)` provides fake drivers with no
   ground truth so the loop can be exercised on a laptop; `tests/test_deployment.py` replays
   a simulated episode's driver readings through it and checks that the observations match
   the simulator's bit for bit.
3. **`firmware/`** — a dependency-free C99 translation of the node-side logic: tracker,
   observation builder, feasibility rule, rule-based controller and the dense-network forward
   pass (about 4 kB of code). `tools/export_c.py` turns a bundle into a header with the
   constants and weights; `firmware/main_example.c` is the decision loop with stub drivers to
   replace; `tests/test_firmware.py` compiles it and checks against Python: observations
   **bit-exact** (float32), rule-based and feasibility decisions identical, network outputs
   equal to ~1e-6 and actions identical except at exact numerical ties.

```bash
python tools/export_c.py examples/bundles/ppo_default.json -o firmware/eea_policy_data.h
cc -std=c99 -O2 -I firmware firmware/eea_node.c firmware/main_example.c -lm -o node_example && ./node_example
```

Details, the field-evaluation workflow and the alternatives (TensorFlow Lite Micro, CMSIS-NN):
`docs/deployment.md`, `firmware/README.md`.

## 13. Notebooks, scripts and tools

The notebooks are **generated** by the `examples/build_*.py` scripts (edit the script, not the
notebook, then run it and execute the notebook).

| file | purpose | runtime |
|---|---|---|
| `examples/baseline_policy.ipynb` | guided tour: environment, one episode step by step, dashboard, metrics, baseline comparison over scenarios | < 1 min |
| `examples/train_rl.ipynb` | the RL experiment: protocol, baselines, PPO and DQN training, learning curves, comparison, behaviour analysis, export check, multi-seed study, memory study | `EEA_BUDGET=quick` ~3 min (smoke test), `default` ~30–40 min + ~1.5 h for the seed study, `paper` longer |
| `examples/trace_driven.ipynb` | the simulator on recorded traces: seasons, model-vs-trace statistics, policies per season, replayed weeks | ~1 min |
| `examples/domains.ipynb` | the three domains side by side: energy budgets, one week each, baselines on all 16 scenarios, cross-domain transfer of PPO policies | ~3 min |
| `examples/compare_policies.py` | headless baseline comparison (`--seeds`, `--randomize`, `--days`) | seconds |
| `examples/train_seeds.py` | train and evaluate one (algorithm, seed) pair: `--algo ppo|dqn|ppo_stack|rppo --seed N --steps ... --domain agriculture|indoor_air|industrial|all --eval-domain ... --out DIR`; writes a JSON with the evaluation rows and learning curve, the checkpoints and (PPO/DQN) the exported bundle | 20–60 min per run |
| `examples/run_long_training.sh` | chains the long runs (PPO 5 M steps × 3 seeds, DQN 2 M × 3) into `examples/rl_runs/seeds_long/`, creating `.venv` with the RL extras if needed; resumable, keeps a macOS laptop awake | 1–5 h |
| `examples/run_cross_domain.sh` | PPO 5 M × 2 seeds trained on each domain and on the mixture of all, evaluated on every domain → `examples/rl_runs/seeds_domains/` (read by `domains.ipynb` §4b) | ~2 h |
| `tools/fetch_open_meteo.py` | download an hourly ERA5/ERA5-Land trace (`--lat --lon --start --end --utc-offset --site --out`) | seconds |
| `tools/make_demo_trace.py` | regenerate the synthetic demo trace (`--year --seed --out`) | seconds |
| `tools/export_c.py` | `PolicyBundle` JSON → `eea_policy_data.h` (`-o`) | instant |

Trained models and evaluation caches go to `examples/rl_runs/` (ignored by git; set
`EEA_RETRAIN=1` to force retraining).

## 14. Repository layout

```
edgeengine_aware/           the package
  config.py                 every parameter as a dataclass field, with units, defaults and validation; scenarios build on it
  interfaces.py             the six hardware protocols (Clock, EnergyStorage, EnergySource, Sensor, Radio, RemoteApplication),
                            the Measurement / Packet / TxResult records and the Policy protocol
  observation.py            NodeProfile (constants), NodeState (measurable state), NodeStateTracker (bookkeeping),
                            ObservationBuilder (the 18-vector) - the sim-to-real contract
  actions.py                action encoding, flat index, plan_execution (feasibility rule)
  energy.py                 SimulatedClock, SimulatedEnergyStorage, SolarEnergySource (stochastic solar model)
  agriculture.py            FieldEnvironment: hidden soil / weather ground truth of the agriculture domain
  sensing.py                SimulatedSoilMoistureSensor
  communication.py          SimulatedLoRaRadio: radio modes, link-budget channel with slow and fast fading, ACK + margin
  application.py            RemoteMonitoringApplication: utility of the delivered information, priority requests
  reward.py                 RewardCalculator with separately reported components
  metrics.py                EpisodeMetrics and EpisodeLog
  env.py                    EdgeEngineAwareEnv (gymnasium.Env)
  rendering.py              Matplotlib dashboard (human / rgb_array) and text dashboard (ansi)
  policies.py               RuleBasedPolicy, PeriodicPolicy, RandomPolicy, AlwaysOnPolicy, run_episode
  scenarios.py              the six agricultural benchmark scenarios; lookup of every domain's scenarios by "domain:name"
  domains.py                the three domains: default configurations, their scenarios, the world factory (build_world)
  process.py                ProcessState (what every monitored process reports) and the weekly ActivitySchedule
  indoor.py                 indoor_air domain: CO2 mass balance of a room, indoor-light PV source
  industrial.py             industrial domain: bearing thermal model with wear/faults/maintenance, thermoelectric source
  rl.py                     FlatActionWrapper, MixedScenarioEnv, make_env, SB3Policy, evaluate / summarize,
                            export_sb3_mlp, NumpyMLPPolicy, FrameStacker / StackedPolicy
  traces.py                 Trace, TraceSolarEnergySource, TraceFieldEnvironment, TraceDrivenEnv
  deployment.py             NodeController, mock hardware backend, PolicyBundle, export_policy
firmware/                   C99 runtime (eea_node.h/.c), main_example.c, test harness, README
tools/                      export_c.py, fetch_open_meteo.py, make_demo_trace.py
data/traces/                real Albenga 2023 trace (ERA5-Land), synthetic demo trace, CSV format description
examples/                   notebooks, their generator scripts, compare_policies.py, train_seeds.py, run_long_training.sh, bundles/
tests/                      pytest suite (see below)
docs/                       detailed documentation (see below)
```

## 15. Tests

`pytest` runs 129 tests:

| file | what it checks |
|---|---|
| `test_env.py` | Gymnasium API and spaces, `check_env`, truncation, feasibility/rejection, no leakage of hidden state into the observation, rendering, single-mode radio |
| `test_energy.py` | energy bounds and conservation, units, solar model, storage bookkeeping |
| `test_reproducibility.py` | same seed → same episode; domain randomisation changes the physics but not the spaces |
| `test_deployment.py` | protocols, mock backend, controller vs simulator observation equality, bundle export |
| `test_rl.py` | scenarios, wrappers, evaluation protocol, numpy MLP runtime vs SB3, rule-based radio-mode choice |
| `test_traces.py` | trace loader semantics (interpolation, interval means, CSV round trip), trace-driven backends and environment |
| `test_firmware.py` | C runtime vs Python for the three domain profiles (compiles `firmware/` with the system C compiler; skipped if none) |
| `test_domains.py` | threshold direction, weekly schedule, CO₂ and bearing models, TEG coupling, shared spaces across domains, qualified scenario names and mixtures, agriculture defaults unchanged |

## 16. Documentation

| document | content |
|---|---|
| `docs/observation.md` | the 18 observation components: meaning, normalisation, hardware source; why the problem is partially observable and what is done about it |
| `docs/actions.md` | action encoding, radio modes and their link budget, the feasibility rule, flat encoding |
| `docs/modeling.md` | all physical models with formulas and default constants, utility and reward, the MDP/POMDP formulation |
| `docs/sim_to_real.md` | the architecture that keeps simulation and hardware on the same contract, the known gaps and their mitigations, domain randomisation, how to add backends |
| `docs/deployment.md` | PolicyBundle, the controller loop, the C runtime, TinyML alternatives, field-evaluation workflow |
| `firmware/README.md` | the C runtime: API, generated data header, equivalence tests, porting notes |
| `data/traces/README.md` | the trace CSV format and the shipped/obtainable data |

## 17. What is not modelled

Being explicit about the simplifications matters more than the list of features:

* **Battery**: an ideal energy buffer with a single charge-efficiency knob; no voltage
  curve, temperature effect, ageing or self-discharge. The node reads its energy exactly.
* **Radio**: delivery depends only on the link margin; no duty-cycle limits, collisions,
  gateway congestion or multiple gateways; energy per uplink is constant per mode.
* **Soil**: a single bucket with evapotranspiration, rain and external irrigation; no crop
  model, no spatial variability. Irrigation is *not* a decision of the node.
* **Application**: a rule-based priority (thresholds, information age, random campaigns), not
  a human operator.
* **Clock**: exact 15-minute steps, no drift or jitter.
* **Indoor air**: a single well-mixed room with one occupancy schedule; the CO₂ sensor's own
  warm-up and drift are folded into the two energy/noise levels. **Industrial**: a first-order
  thermal model with a scalar "health"; vibration is not simulated, the accurate sensing level
  stands for a vibration burst only through its energy and precision.

`docs/sim_to_real.md` lists, for each of these, what a real device will do differently and
which mitigation (domain randomisation, configuration, recorded traces) is available.

## Glossary

* **ACK (acknowledgement)** — a short reply from the gateway confirming that an uplink was
  received. It also lets the node measure how much margin the link had.
* **AoI (age of information)** — how old the freshest value known to the application is:
  time since it was received plus the age the reading already had when sent.
* **Brown-out** — the battery cannot even supply the always-on load; the node stops
  working until energy is harvested again.
* **Domain randomisation** — perturbing the simulator's physical parameters at every
  episode so that a trained policy is robust to devices and conditions that differ from the
  nominal ones.
* **Duty cycle** — a fixed schedule ("sense and transmit every hour"), the traditional way
  low-power nodes are programmed.
* **EWMA (exponentially weighted moving average)** — a running average that gives weight
  α to the newest value and (1−α) to the previous average; a cheap memory of the recent past.
* **Field capacity** — the amount of water a soil holds after excess water has drained;
  soil moisture in this project is expressed as a fraction of it (1.0 = field capacity).
* **Held-out seeds** — random seeds used only for evaluation, never during training, so that
  results are not flattered by memorised episodes.
* **Link margin / link budget** — transmit power minus path loss minus receiver sensitivity,
  in dB. Positive margin: the message is likely to be received. Each radio mode has a
  different sensitivity, hence a different margin for the same channel.
* **Path loss** — attenuation of the radio signal between node and gateway, in dB; it varies
  slowly (vegetation, humidity, obstacles: *slow fading*) and from message to message
  (*fast fading*).
* **POMDP** — a decision problem in which the agent does not observe the full state (here:
  the true soil moisture, the weather regime, the channel state are hidden). Standard remedies
  are summary statistics of the past, explicit timers and, if needed, memory in the policy.
* **Priority** — the application's declared level of interest: routine (0), elevated (1),
  urgent (2).
* **SF (spreading factor)** — a LoRa modulation setting; higher SF means longer, slower
  transmissions that can be received at weaker signal (better sensitivity) but cost more
  energy. The project's *fast / standard / robust* modes are SF7 / SF9 / SF12-like.
* **SoC (state of charge)** — stored energy divided by capacity, from 0 to 1.
* **`MultiDiscrete([3, 4])`** — Gymnasium's action space for a vector of independent
  integers, here one in {0,1,2} and one in {0,1,2,3}.

## Citation

If you use EdgeEngine AWARE in academic work, please cite the repository (Riccardo Berta,
DITEN – ELIOS Lab, University of Genoa): https://github.com/measurify/EdgeEngine-AWARE
