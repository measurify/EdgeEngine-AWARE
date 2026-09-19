# EdgeEngine AWARE

**Application- and Energy-Aware Simulation Environment for Reinforcement Learning in
Energy-Harvesting Edge IoT Systems**

EdgeEngine AWARE is a Gymnasium environment that models an autonomous, solar-powered Edge IoT
node monitoring soil moisture in an agricultural field. At every 15-minute step an agent
decides *how well to sense* (off / low-cost / high-quality) and *whether to transmit* the
latest sample to a remote application over a LoRa-like link. It must trade off

> information quality · information freshness · energy consumption · future energy
> availability · application relevance

under a finite, time-varying energy budget. The goal is not to minimise energy but to
**maximise long-term application utility under energy constraints**.

The simulator is the first stage of a pipeline —
*simulation → policy training → validation → policy export → embedded deployment → real-world
evaluation* — and is built around one constraint: **the simulator may know more than the
deployed device, but the policy must not.** Ground truth drives the world, the reward and
the plots; the policy sees only what a microcontroller could measure.

## Repository layout

```
edgeengine_aware/
  __init__.py        package exports; registers "EdgeEngineAware-v0"
  config.py          dataclass configuration tree (all parameters, units, defaults, domain randomisation)
  interfaces.py      hardware abstraction protocols: Clock, EnergyStorage, EnergySource, Sensor, Radio,
                     RemoteApplication; Measurement / Packet records; Gymnasium-independent Policy protocol
  observation.py     NodeProfile (flash constants), NodeState (hardware-measurable state),
                     NodeStateTracker (firmware bookkeeping), ObservationBuilder (the 17-vector)
  actions.py         MultiDiscrete([3, 2]) encoding, flat index, shared energy-feasibility rule
  energy.py          SimulatedClock, SimulatedEnergyStorage, SolarEnergySource (stochastic solar cycle)
  agriculture.py     FieldEnvironment: hidden ground truth (soil moisture, temperature, humidity, rain, irrigation)
  sensing.py         SimulatedSoilMoistureSensor (level-dependent noise)
  communication.py   SimulatedLoRaRadio (energy per attempt, stochastic delivery, ACK)
  application.py     RemoteMonitoringApplication: information utility, priority requests
  reward.py          RewardCalculator with separately reported components
  metrics.py         EpisodeMetrics (incl. Age of Information) and per-step EpisodeLog
  env.py             EdgeEngineAwareEnv(gym.Env)
  rendering.py       Matplotlib dashboard (human / rgb_array) and text dashboard (ansi)
  policies.py        RuleBasedPolicy, RandomPolicy, PeriodicPolicy, AlwaysOnPolicy, run_episode
  deployment.py      NodeController (firmware loop), mock hardware backend, PolicyBundle export
examples/
  baseline_policy.ipynb   lecture notebook: environment tour, rule-based episode, metrics, comparisons
  build_notebook.py       regenerates the notebook
  compare_policies.py     headless comparison of the baselines
tests/
  test_env.py             Gymnasium API, spaces, truncation, rejection, leakage, rendering
  test_energy.py          bounds, conservation, units, solar model
  test_reproducibility.py seeding, domain randomisation
  test_deployment.py      protocols, mock backend, shared observation builder, export
docs/
  observation.md   every observation component, its normalisation and hardware source; Markov discussion
  actions.md       action encoding and feasibility rule
  modeling.md      modelling assumptions, utility, reward, MDP/POMDP formulation
  sim_to_real.md   architecture, sim-to-real gap analysis, domain randomisation
  deployment.md    policy export, firmware loop, TinyML options, field-evaluation workflow
```

## Installation

Python ≥ 3.11.

```bash
git clone <this repository> && cd edgeengine_aware
python -m venv .venv && source .venv/bin/activate      # optional
pip install -e ".[dev]"                                 # numpy, gymnasium, matplotlib, pytest, jupyter
pytest                                                  # 48 tests, ~3 s
```

## Quick start

```python
import gymnasium as gym
import edgeengine_aware as ea

env = gym.make("EdgeEngineAware-v0")            # or ea.EdgeEngineAwareEnv(config, render_mode="human")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

print(env.unwrapped.obs_builder.describe())     # the observation vector, documented from code
print(info["reward_components"])                # utility, costs and penalties, separately
print(info["metrics"]["average_aoi_s"])         # episode metrics, updated every step
```

Rule-based baseline over a full episode:

```python
from edgeengine_aware.policies import RuleBasedPolicy, run_episode
env = ea.EdgeEngineAwareEnv(render_mode="human")
result = run_episode(env, RuleBasedPolicy(), seed=1, render=True, render_every=96)
print(env.metrics.summary())
```

Headless comparison and the notebook:

```bash
python examples/compare_policies.py --seeds 10
jupyter notebook examples/baseline_policy.ipynb
```

## Environment summary

| | |
|---|---|
| observation | `Box(0, 1, (17,), float32)` — battery SoC, harvest (now, recent), time of day (sin, cos), stored sample (value, quality, age), time since ACK, estimated information age at the application, reported value, application priority, node-side importance, link quality, energy costs (low, high, tx). See `docs/observation.md`. |
| action | `MultiDiscrete([3, 2])` — (sensing level, transmit). See `docs/actions.md`. |
| step / episode | 15 min / 7 days = 672 steps (configurable); truncated at the horizon, optional termination on brown-out |
| reward | tracking utility + packet bonus − sensing/communication energy − staleness (priority-weighted AoI) − battery risk − brown-out − rejected actions − wasted harvest; all components in `info["reward_components"]` |
| metrics | harvested / consumed / baseline / sensing / communication / wasted energy; sensing, high-quality sensing, transmissions, deliveries; average & min SoC, low-battery fraction, depletion events; average & max AoI; total utility and reward |
| render modes | `human` (live Matplotlib dashboard, updated in place), `rgb_array`, `ansi` |
| seeding | fully reproducible from `reset(seed=...)`; passes `gymnasium.utils.env_checker.check_env` |

Default physical scale (all replaceable by device measurements): 300 J storage, 200 µW
baseline, 5 mW-peak cell with 60 % converter efficiency (~55 J/day on an average day),
0.10 J / 0.60 J per low / high-quality sample, 0.60 J per uplink with 90 % nominal delivery.
Hourly high-quality reporting is sustainable on sunny days and not on cloudy ones — the
regime where an energy-aware policy matters.

## Configuration

Everything is a dataclass field; nothing in the environment code needs to change.

```python
cfg = ea.default_config()
cfg.time.timestep_s = 600                 # 10-minute steps
cfg.time.episode_days = 14
cfg.storage.capacity_j = 800
cfg.storage.initial_soc_range = (0.3, 0.9)
cfg.mcu.baseline_power_w = 120e-6
cfg.harvesting.max_power_w = 0.008
cfg.harvesting.clearness_mean = 0.5       # cloudier climate
cfg.sensing.energy_j = (0.0, 0.05, 0.4)
cfg.sensing.noise_std = (0.0, 0.05, 0.005)
cfg.communication.tx_energy_j = 0.9
cfg.communication.base_success_prob = 0.8
cfg.communication.priority_update_mode = "on_uplink"   # LoRaWAN class-A style downlink
cfg.agriculture.warning_threshold, cfg.agriculture.critical_threshold = 0.40, 0.30
cfg.agriculture.rain_events_per_day = 0.5
cfg.reward.lambda_battery = 1.0
cfg.randomization.enabled = True          # domain randomisation at every reset
env = ea.EdgeEngineAwareEnv(cfg)
```

## Sim-to-real in one paragraph

Six tiny protocols (`Clock`, `EnergyStorage`, `EnergySource`, `Sensor`, `Radio`,
`RemoteApplication`) separate the node's decision logic from whatever produces its readings.
`NodeStateTracker` + `ObservationBuilder` + `plan_execution` are shared between the simulator
(`env.py`) and the firmware-style loop (`deployment.NodeController`); a policy only ever sees
the 17-vector and only ever emits `(sense, tx)`. `deployment.PolicyBundle` exports the full
contract (observation order and normalisation constants, action encoding, model parameters,
metadata) so that the same preprocessing runs on the device. Domain randomisation of the
physical parameters is one switch away. Details: `docs/sim_to_real.md`, `docs/deployment.md`.

## Using it with RL libraries

The environment follows the Gymnasium API and works unchanged with Stable-Baselines3
(`PPO`/`A2C` accept `MultiDiscrete`). For DQN-like agents wrap the action space:

```python
import gymnasium as gym
from gymnasium import spaces
from edgeengine_aware.actions import unflatten_action, N_FLAT_ACTIONS

class FlatActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env); self.action_space = spaces.Discrete(N_FLAT_ACTIONS)
    def action(self, a):
        return unflatten_action(int(a))
```

No RL algorithm is implemented here on purpose: the package is the benchmark, not the agent.

## Extensibility

The architecture leaves room, without redesign, for: multiple sensors (more `Sensor`
objects and observation components), multiple application requirements (several
`RemoteApplication` priorities), adaptive radio power / spreading factor / payload size
(extra `MultiDiscrete` columns handled in `plan_execution` and `Radio`), data compression and
edge inference / TinyML / local event detection (actions with an energy cost and an effect on
the packet), multiple nodes (a vector env of `EdgeEngineAwareEnv` or a shared channel object),
other harvesting technologies (another `EnergySource`), battery ageing (inside
`SimulatedEnergyStorage`), explicit network congestion (inside `SimulatedLoRaRadio`), real
sensor / harvesting traces (trace-driven `Sensor` / `EnergySource`), hardware-in-the-loop
(`NodeController` with serial drivers) and embedded deployment (`PolicyBundle` → firmware).

## Citation

If you use EdgeEngine AWARE in academic work, please cite the repository (Riccardo Berta,
DITEN – ELIOS Lab, University of Genoa).
