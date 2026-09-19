# From a trained policy to a microcontroller

## The contract that must be preserved

A policy is only meaningful together with the pre- and post-processing it was trained with.
`edgeengine_aware.deployment.PolicyBundle` freezes all of it in one JSON file:

| field | content | why it matters |
|---|---|---|
| `observation_names` | the 18 names **in order** | the index of each input to the network / rule |
| `observation_normalisation` | the formula of each component | how raw readings become inputs |
| `profile` | `NodeProfile`: sensing energies, per-mode radio energies / transmit powers / sensitivities, reference mode, thresholds, timestep, baseline power, brown-out reserve, ACK availability, `harvest_ref_power_w`, `age_scale_s`, EWMA α's, `importance_scale`, path-loss range | every constant the firmware needs to reproduce `ObservationBuilder` (incl. the path-loss estimate) and `plan_execution` |
| `action_encoding` | `nvec = [3, 1 + n_modes]`, `flat = sense*(1+n_modes) + tx`, semantics of each value | how the output becomes `sensor_read(level)` / `radio_send(mode)` |
| `model` | thresholds (rule-based), weights/biases/activations (MLP), table (tabular) | the decision function itself |
| `metadata` | package version, export time, policy class, notes | traceability |

```python
from edgeengine_aware.deployment import export_policy, PolicyBundle
bundle = export_policy(policy, NodeProfile.from_config(cfg), policy_type="mlp", model=weights_as_lists)
bundle.save("policy_v1.json")
```

If **any** of the first four fields changes (a new observation component, a different
`age_scale_s`, a re-ordered action), the model must be re-trained. The firmware should
embed the bundle's `metadata` and refuse a model whose profile does not match its flash
constants.

## Dimensionality

* Input: **18 floats** in `[0, 1]` (fits fixed-point `q15`/`uint8` quantisation without
  rescaling).
* Output: **12 discrete actions** (or two heads of 3 and 4 logits).
* Rule-based baseline: ~20 comparisons and a three-entry table lookup, no multiplications
  beyond the path-loss conversion.
* A 18→64→64→7 MLP (the notebook's PPO actor): ~5.7 k parameters, ~5.7 k MACs per decision —
  negligible on any Cortex-M at one decision per 15 minutes.

## Firmware main loop (mirrors `deployment.NodeController`)

```c
void decision_cycle(void) {
    node_state_t s;
    s.time_of_day_s   = rtc_seconds_since_midnight();
    s.stored_energy_j = gauge_read_energy_j();          // or V -> SoC lookup
    s.harvest_power_w = harvester_read_avg_power_w();   // energy integrated since last wake-up / dt
    tracker_begin_step(&tracker, &s, downlink_last_priority());

    float obs[18];
    observation_build(&tracker, &profile, obs);         // == ObservationBuilder.build

    uint8_t sense, tx;
    policy_act(obs, &sense, &tx);                       // rule table or NN inference

    plan_t plan = plan_execution(sense, tx, s.stored_energy_j, BASELINE_J, RESERVE_J,
                                 profile.sense_energy_j, profile.tx_energy_j, tracker.has_measurement);
    if (plan.sense) { measurement_t m = sensor_read(plan.sense_level); tracker_on_measurement(&tracker, &m); }
    if (plan.tx)    { tx_result_t r = lora_send_confirmed(&tracker.measurement, plan.mode);   // sets SF/power
                      tracker_on_transmission(&tracker, r.acked, plan.mode, r.margin_db); }  // updates the path-loss estimate

    sleep_until_next_slot(TIMESTEP_S);
}
```

Every function on the left has a Python twin in `observation.py` / `actions.py` — and now a C
twin in `firmware/eea_node.c` (see below); `tests/test_firmware.py` feeds the same events to
both and compares the 18 outputs bit-for-bit (after float32 rounding) and every action.

## The C runtime (`firmware/`)

| file | content |
|---|---|
| `eea_node.h` / `eea_node.c` | `eea_tracker_*` (= `NodeStateTracker`), `eea_observation_build` (= `ObservationBuilder.build`, float32 output), `eea_plan_execution[_profile]` (= `plan_execution`), `eea_rule_based_act` (= `RuleBasedPolicy.act`, incl. the radio-mode choice), `eea_mlp_forward` / `eea_mlp_act` (= `NumpyMLPPolicy`). C99, `<math.h>` only, no allocation; ~4 kB of code at `-Os` on x86-64, less on Cortex-M. |
| `eea_policy_data.h` | **generated** by `tools/export_c.py <bundle.json>`: `EEA_PROFILE` (all `NodeProfile` constants), and `EEA_RULE_PARAMS` *or* `EEA_MLP` + weight arrays (`float`, ~23 kB for the 18→64→64→7 PPO actor), plus `EEA_BUNDLE_*` metadata strings the firmware can log or check. |
| `main_example.c` | the decision loop above with HAL stubs — compiles on the host and prints a few cycles. |
| `test/eea_harness.c` | stdin-driven harness used by the equivalence tests (not firmware). |

Numerics: the tracker and the observation use `double` like the Python reference and round to
`float` at the end exactly as numpy does, so observations are bit-exact; the MLP uses `float`
accumulation like numpy's float32 matmul (logits agree to ~1e-6, actions agree except at
numerical ties, which the test tolerates). `EEA_REAL` can be redefined to `float` for
FPU-only-single targets, at the price of last-bit differences in the observation.

## Deployment options for the decision function

| policy type | how to deploy | when |
|---|---|---|
| **rule-based** (`RuleBasedPolicy`) | translate `act()` to C; thresholds from `bundle.model` | baseline, safety fallback, first field test |
| **tabular** (Q-table over a discretised observation) | store the table (6 × #bins) in flash; same binning code as in Python | teaching, tiny MCUs |
| **small MLP** (PPO / DQN actor) | `tools/export_c.py` → `eea_policy_data.h` + the dense loop of `eea_mlp_forward` (tested bit-for-bit against `rl.NumpyMLPPolicy`); alternatively CMSIS-NN or **TensorFlow Lite for Microcontrollers** after converting the network (Keras/ONNX → TFLite, int8 quantisation with the observation range `[0,1]` as calibration) | production |
| **ONNX-derived** | ONNX → `onnx2c` / STM32Cube.AI / Edge Impulse | vendor toolchains |
| **quantised NN** | post-training int8 quantisation is safe here: inputs are already in `[0,1]`, outputs are argmax'ed logits | when RAM/flash are tight |

The C runtime covers the rule-based and MLP rows; the boundary for anything else is
`Policy.act(obs) -> action` plus the bundle.

## Field-evaluation workflow

1. **Train** in simulation (with domain randomisation) → `PolicyBundle`.
2. **Validate** in simulation on held-out seeds and on stress configurations (cloudy climate,
   weak panel, lossy channel); compare with the rule-based baseline on the same
   `EpisodeMetrics`.
3. **Mock-hardware test**: run the exported decision function through `NodeController` with
   `make_mock_backend` (fake drivers plus a minimal power-path emulation) or with trace-driven
   drivers — `tests/test_deployment.py` replays the driver readings of a simulated episode
   through the controller and checks that the observations match the simulator bit for bit,
   in both priority-downlink modes.
4. **Hardware-in-the-loop**: same controller, drivers talking to the board over serial;
   compare per-cycle observations from the board with those recomputed in Python.
5. **Deploy** with a safety envelope in firmware: the brown-out reserve and the feasibility rule
   already bound the damage a bad policy can do; add a watchdog fallback to the rule-based
   policy if the SoC stays below `soc_critical` for too long.
6. **Evaluate in the field** against the *same* metrics used in simulation:
   * node side: energy split, sensing/transmission counts, SoC statistics, depletion events
     (all in the firmware's own counters);
   * application side: delivered packets, AoI statistics, priority timeline;
   * utility: requires ground truth → co-locate a mains- or large-battery-powered *reference
     node* sampling at high rate; compute `u_track` offline from the reference trace and the
     application's belief timeline, exactly as `RemoteMonitoringApplication.tracking` does.
7. **Close the loop**: feed the recorded harvesting, sensor and channel traces back into the
   simulator (`edgeengine_aware.traces`, see `examples/trace_driven.ipynb`) to re-tune the
   models and re-train. Before any field data exists, public reanalysis traces
   (`tools/fetch_open_meteo.py`) already give a season-by-season check of the transfer.

## Comparing policies in EdgeEngine AWARE

`examples/train_rl.ipynb` implements the protocol; `edgeengine_aware.rl` and
`edgeengine_aware.scenarios` hold the reusable parts.

| method | how it plugs in | status |
|---|---|---|
| rule-based baseline | `RuleBasedPolicy` | implemented; interpretable reference and safety fallback |
| random / periodic | `RandomPolicy`, `PeriodicPolicy` | implemented; sanity floor and duty-cycle reference |
| PPO / A2C (actor-critic) | SB3 `MlpPolicy` on the native `MultiDiscrete` space; `rl.SB3Policy` adapts the model to the `Policy` protocol | trained and evaluated in the notebook |
| DQN family | `rl.FlatActionWrapper` → `Discrete(12)` | trained and evaluated in the notebook |
| tabular (Q-learning / SARSA) | bin the 18 inputs with a `gym.ObservationWrapper`, use `flatten_action` | not implemented; a teaching exercise |
| frame-stacked / recurrent policies | `VecFrameStack` (mirrored on the node by `rl.FrameStacker` / `rl.StackedPolicy`) or `RecurrentPPO` (sb3-contrib) | trained and evaluated in the notebook (section 9); `examples/train_seeds.py --algo ppo_stack|rppo` |
| **multi-seed protocol** | `examples/train_seeds.py --algo <ppo|dqn|ppo_stack|rppo> --seed N` → JSON with evaluation rows and learning curve; the notebook aggregates mean ± std across runs | section 8 of the notebook |
| **evaluation protocol** | `rl.evaluate(policies, scenarios, seeds)` → one `EvalRow` per episode; `rl.summarize` for mean ± std | six scenarios (`scenarios.SCENARIOS`), held-out seeds ≥ 1000, deterministic policies, nominal (non-randomised) physics |
| **export check** | `rl.export_sb3_mlp` → `PolicyBundle.model`; `rl.NumpyMLPPolicy` re-executes it with numpy and must reproduce every SB3 action | part of the notebook and of `tests/test_rl.py` |
