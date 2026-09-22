# From a trained policy to a microcontroller

This document covers the pipeline *policy → export → firmware → field evaluation*: what
must travel with a policy, how the node's decision loop looks in Python and in C, how to
deploy the decision function, and how to evaluate the result in the field with the same
metrics as in simulation.

## 1. The contract that must be preserved

A policy is meaningful only together with the pre-processing (how readings become the 18
inputs) and the post-processing (how the output becomes an operation) it was trained with.
`edgeengine_aware.deployment.PolicyBundle` freezes all of it in one JSON file:

| field | content | why it matters |
|---|---|---|
| `observation_names` | the 18 names **in order** | the index of each input to the network or rule |
| `observation_normalisation` | the formula of each component, as text | documents how raw readings become inputs |
| `profile` | the `NodeProfile` as a dictionary: sensing energies and noise levels, per-mode radio energies / transmit powers / sensitivities, reference mode, thresholds and their danger side (`critical_is_upper`), timestep, always-on power, brown-out reserve, ACK availability, and a nested `observation` dictionary with the normalisation constants (`harvest_ref_power_w`, `age_scale_s`, the two EWMA α, `importance_scale`, path-loss range) | every constant the firmware needs to reproduce `ObservationBuilder` and `plan_execution` |
| `action_encoding` | `nvec = [3, 1 + n_modes]`, the flat-index formula, the meaning of each value | how the output becomes `sensor_read(level)` / `radio_send(mode)` |
| `model` | the decision function: thresholds (`RuleBasedParams`) for the rule-based policy; `{"type": "mlp", "layers": [{"W", "b", "activation"}], "output", "output_split", "n_modes"}` for a dense network | the policy itself |
| `metadata` | package version, export time, Python version, policy class, free-text notes | traceability |

```python
import edgeengine_aware as ea
from edgeengine_aware.deployment import export_policy, PolicyBundle

cfg = ea.default_config()
profile = ea.NodeProfile.from_config(cfg)

# rule-based: the thresholds are exported automatically
bundle = export_policy(ea.RuleBasedPolicy(profile=profile), profile, policy_type="rule_based", notes="v1")
bundle.save("rule_based_v1.json")

# a trained SB3 network: extract the weights first
# from edgeengine_aware.rl import export_sb3_mlp
# bundle = export_policy(SB3Policy(model), profile, policy_type="mlp", model=export_sb3_mlp(model))

again = PolicyBundle.load("rule_based_v1.json")
```

If **any** of the first four fields changes (a new observation component, a different
`age_scale_s`, a re-ordered action), the model must be re-trained. The firmware embeds the
bundle's metadata (`EEA_BUNDLE_*` macros in the generated header) so that a deployed image can
report which policy it runs. A run-time check that the profile matches the board's real
constants is *not* implemented: keep them consistent by generating the header from the same
bundle that was validated.

Six bundles are shipped in `examples/bundles/`: `rule_based_default.json`,
`ppo_default.json` (the PPO actor of the RL notebook, 2 M steps, network 18→64→64→7),
`ppo_long.json` (PPO, 5 M steps, agriculture, seed 1 of `examples/run_long_training.sh`) and
`ppo_indoor_air.json`, `ppo_industrial.json`, `ppo_universal.json` (PPO, 5 M steps, seed 1 of
`examples/run_cross_domain.sh`, trained on that domain or on all three).

## 2. Sizes

* Input: **18 floats** in `[0, 1]` — suitable for 8- or 16-bit fixed-point quantisation
  without rescaling.
* Output: **12 discrete actions**, or two groups of 3 and 4 logits (PPO's two heads).
* Rule-based controller: about 20 comparisons and a handful of multiplications (ages ×
  scale, interval × factors), plus the radio-mode choice from a table of ≤ 4 entries.
* An 18→64→64→7 dense network (the notebook's PPO actor): 5 831 parameters (≈ 23 kB as
  `float`), the same number of multiply-adds per decision — negligible on any Cortex-M at one
  decision per 15 minutes.

## 3. The decision loop

### In Python (`deployment.NodeController`)

`NodeController(backend, profile, policy, priority_update_mode)` is the firmware main loop
written against the hardware protocols of `interfaces.py`. One call to `run_cycle()` performs
one decision cycle (build the observation, ask the policy, apply the feasibility rule, drive
the sensor and the radio, update the tracker) and returns a `CycleReport`; sleeping between
cycles is left to the caller. `make_mock_backend(profile)` builds fake drivers
(a fuel gauge with a minimal power-path emulation, a harvester monitor, a sensor driver, a
radio whose ACKs follow a link budget) that know **no ground truth**, so the loop can be run
on a laptop; replacing them with real drivers is the port.

### In C (`firmware/eea_node.c`, loop in `firmware/main_example.c`)

```c
static eea_tracker_t tracker;                        /* keep in retained RAM across sleep cycles */
eea_tracker_init(&tracker, &EEA_PROFILE);            /* EEA_PROFILE comes from the generated header */

for (;;) {
    /* 1. sample the measurable state */
    double now = rtc_now_s();
    eea_tracker_begin_step(&tracker, now, rtc_time_of_day_s(now),
                           gauge_energy_j(), gauge_capacity_j(),
                           harvester_avg_power_w(),                  /* energy since last wake-up / dt */
                           downlink_priority_or_minus_one());         /* -1 = no new downlink */

    /* 2. observation == ObservationBuilder.build */
    eea_node_state_t state;  float obs[EEA_OBS_DIM];
    eea_tracker_state(&tracker, &state);
    eea_observation_build(&EEA_PROFILE, &state, obs);

    /* 3. decision (one of the two, depending on the bundle) */
    int sense, tx;
    eea_mlp_act(&EEA_MLP, obs, &sense, &tx);                       /* or eea_rule_based_act(&EEA_RULE_PARAMS, &EEA_PROFILE, obs, &sense, &tx) */

    /* 4. feasibility == actions.plan_execution */
    eea_plan_t plan;
    eea_plan_execution_profile(&EEA_PROFILE, sense, tx, state.stored_energy_j, state.capacity_j,
                               state.has_measurement, &plan);

    /* 5. execute */
    if (plan.sensing_level) {
        eea_measurement_t m = { sensor_read(plan.sensing_level), now, plan.sensing_level,
                                EEA_PROFILE.sensing_noise_std[plan.sensing_level] };
        eea_tracker_on_measurement(&tracker, &m);
    }
    if (plan.transmit) {
        bool has_margin; double margin_db;
        int acked = lora_send(tracker.measurement.value, plan.mode, &has_margin, &margin_db);   /* 1 / 0 / -1 */
        eea_tracker_on_transmission(&tracker, now, acked, now, plan.mode, has_margin, margin_db);
    }
    sleep_until_next_slot(EEA_PROFILE.timestep_s);
}
```

Every function above has a Python twin — `NodeStateTracker.begin_step / state /
on_measurement / on_transmission`, `ObservationBuilder.build`, `plan_execution`,
`RuleBasedPolicy.act`, `NumpyMLPPolicy.act` — and `tests/test_firmware.py` feeds the same
events to both and compares the results.

### The C runtime files (`firmware/`)

| file | content |
|---|---|
| `eea_node.h` / `eea_node.c` | the runtime: `eea_tracker_*`, `eea_observation_build`, `eea_plan_execution[_profile]`, `eea_rule_based_act` (including the radio-mode choice), `eea_mlp_forward` / `eea_mlp_act`. C99, `<math.h>` only, no dynamic allocation; about 4 kB of code at `-Os` on x86-64. |
| `eea_policy_data.h` | **generated** by `python tools/export_c.py <bundle.json> -o firmware/eea_policy_data.h`: `EEA_PROFILE` (all `NodeProfile` constants) and either `EEA_RULE_PARAMS` (rule-based bundle) or `EEA_MLP` with the weight arrays (network bundle), plus `EEA_BUNDLE_*` metadata strings. Not committed to git: regenerate it from the bundle. |
| `main_example.c` | the loop above with stub drivers; compiles and runs on a laptop. |
| `test/eea_harness.c` | a stdin-driven harness used by the equivalence tests (not firmware). |

Numerics: the tracker and the observation builder compute in `double`, like the Python
reference, and round to `float` at the very end, as numpy does — observations are therefore
**bit-exact**. The network runs in `float` like numpy's float32 matrix product; because the
summation order differs, logits agree to about 1e-6 and the chosen action can differ only when
two logits are equal to that precision (the test tolerates such ties). `EEA_REAL` can be
redefined to `float` for single-precision-only targets; the observation may then differ in
the last bit, and the equivalence test does not cover that build.

## 4. Deploying the decision function

| policy type | how | when |
|---|---|---|
| **rule-based** (`RuleBasedPolicy`) | `eea_rule_based_act` with the thresholds from the bundle | baseline, safety fallback, first field test |
| **dense network** (PPO / DQN actor) | `tools/export_c.py` + `eea_mlp_forward` (tested against `rl.NumpyMLPPolicy`); alternatively CMSIS-NN (ARM's optimised kernels) or TensorFlow Lite for Microcontrollers after converting the network (int8 quantisation is safe here: inputs are already in `[0, 1]`, outputs are arg-maxed) | production |
| **tabular** (a Q-table over discretised observations) | store the table (12 × number of bins) in flash; not implemented — a teaching exercise | tiny MCUs |
| **vendor toolchains** | ONNX → onnx2c / STM32Cube.AI / Edge Impulse | when the platform requires it |

For anything else, the boundary is `Policy.act(obs) -> action` plus the bundle.

## 5. Field-evaluation workflow

1. **Train** in simulation, with domain randomisation → `PolicyBundle`.
2. **Validate** in simulation on held-out seeds and on the stress scenarios; compare with the
   rule-based baseline on the same `EpisodeMetrics`. Then validate on **recorded traces**
   (`examples/trace_driven.ipynb`): weather and soil data the models never generated.
3. **Mock-hardware test**: run the exported decision function through `NodeController` with
   `make_mock_backend`; `tests/test_deployment.py` replays a simulated episode's driver
   readings through the controller and checks the observations match the simulator bit for
   bit, in both priority-downlink modes.
4. **C equivalence**: `pytest tests/test_firmware.py` after any change to the runtime or the
   bundle.
5. **Hardware-in-the-loop**: the controller (Python over a serial link) or the C runtime on
   the board; compare the board's per-cycle observations with those recomputed in Python from
   the logged readings.
6. **Deploy with a safety envelope**: the brown-out reserve and the feasibility rule already
   bound the damage a bad policy can do; add a watchdog that falls back to the rule-based
   policy if the SoC stays below a threshold for too long.
7. **Evaluate in the field** with the *same* metrics as in simulation:
   * node side: energy split, sensing/transmission counts, SoC statistics, brown-outs (all in
     the firmware's own counters);
   * application side: delivered packets, age-of-information statistics, priority timeline;
   * utility: requires ground truth → co-locate a mains- or large-battery-powered *reference
     node* sampling at high rate; compute `u_track` offline from the reference trace and the
     application's belief timeline, exactly as `RemoteMonitoringApplication.tracking` does.
8. **Close the loop**: feed the recorded harvesting, sensor and channel traces back into the
   simulator (`edgeengine_aware.traces`) to re-tune the models and re-train.

## 6. Comparing policies

`examples/train_rl.ipynb` implements the protocol; `edgeengine_aware.rl` and
`edgeengine_aware.scenarios` hold the reusable parts.

| method | how it plugs in | status |
|---|---|---|
| rule-based | `RuleBasedPolicy(profile=...)` | reference and safety fallback |
| random / periodic / always-on | `RandomPolicy`, `PeriodicPolicy`, `AlwaysOnPolicy` | sanity floor and duty-cycle references |
| PPO / A2C | SB3 `MlpPolicy` on the native `MultiDiscrete` space; `rl.SB3Policy` adapts the model to the `Policy` protocol | trained and evaluated in the notebook |
| DQN | `rl.FlatActionWrapper` → `Discrete(12)` | trained and evaluated in the notebook |
| frame-stacked / recurrent | SB3 `VecFrameStack` (mirrored on the node by `rl.FrameStacker` / `rl.StackedPolicy`) or `RecurrentPPO` (sb3-contrib) | notebook section 9; `examples/train_seeds.py --algo ppo_stack|rppo` |
| tabular | discretise the 18 inputs with an `ObservationWrapper`, use `flatten_action` | not implemented |
| **multi-seed protocol** | `examples/train_seeds.py --algo <ppo|dqn|ppo_stack|rppo> --seed N` → JSON with evaluation rows, learning curve and (PPO/DQN) the exported bundle; the notebook aggregates mean ± std across runs; `examples/run_long_training.sh` chains the 5 M-step runs | notebook sections 8 and 10 |
| **evaluation** | `rl.evaluate(policies, scenarios, seeds)` → one `EvalRow` per episode (reward, utility, energies, counts, min SoC, low-battery fraction, brown-outs, mean/max AoI in hours, reward components); `rl.summarize(rows, metric)` → `{scenario: {policy: (mean, std)}}`; `evaluate(policies, envs={label: env})` for ready-made environments such as trace-driven ones | six scenarios, held-out seeds ≥ 1000, deterministic policies, nominal physics |
| **export check** | `rl.export_sb3_mlp` → bundle `model`; `rl.NumpyMLPPolicy` re-executes it with numpy and must reproduce every SB3 action | notebook and `tests/test_rl.py` |
