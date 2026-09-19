# From a trained policy to a microcontroller

## The contract that must be preserved

A policy is only meaningful together with the pre- and post-processing it was trained with.
`edgeengine_aware.deployment.PolicyBundle` freezes all of it in one JSON file:

| field | content | why it matters |
|---|---|---|
| `observation_names` | the 17 names **in order** | the index of each input to the network / rule |
| `observation_normalisation` | the formula of each component | how raw readings become inputs |
| `profile` | `NodeProfile`: sensing energies, tx energy, thresholds, timestep, baseline power, brown-out reserve, ACK availability, `harvest_ref_power_w`, `age_scale_s`, EWMA α's, `importance_scale` | every constant the firmware needs to reproduce `ObservationBuilder` and `plan_execution` |
| `action_encoding` | `nvec = [3, 2]`, `flat = sense*2 + tx`, semantics of each value | how the output becomes `sensor_read(mode)` / `radio_send()` |
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

* Input: **17 floats** in `[0, 1]` (fits fixed-point `q15`/`uint8` quantisation without
  rescaling).
* Output: **6 discrete actions** (or two heads of 3 and 2 logits).
* Rule-based baseline: ~15 comparisons, no multiplications.
* A 17→32→32→6 MLP: ~1.8 k parameters, ~1.8 k MACs per decision — negligible on any Cortex-M
  at one decision per 15 minutes.

## Firmware main loop (mirrors `deployment.NodeController`)

```c
void decision_cycle(void) {
    node_state_t s;
    s.time_of_day_s   = rtc_seconds_since_midnight();
    s.stored_energy_j = gauge_read_energy_j();          // or V -> SoC lookup
    s.harvest_power_w = harvester_read_avg_power_w();   // energy integrated since last wake-up / dt
    tracker_begin_step(&tracker, &s, downlink_last_priority());

    float obs[17];
    observation_build(&tracker, &profile, obs);         // == ObservationBuilder.build

    uint8_t sense, tx;
    policy_act(obs, &sense, &tx);                       // rule table or NN inference

    plan_t plan = plan_execution(sense, tx, s.stored_energy_j, BASELINE_J, RESERVE_J,
                                 profile.sense_energy_j, profile.tx_energy_j, tracker.has_measurement);
    if (plan.sense) { measurement_t m = sensor_read(plan.sense_level); tracker_on_measurement(&tracker, &m); }
    if (plan.tx)    { bool ack = lora_send_confirmed(&tracker.measurement);  tracker_on_transmission(&tracker, ack); }

    sleep_until_next_slot(TIMESTEP_S);
}
```

Every function on the left has a Python twin in `observation.py` / `actions.py`; porting is a
line-by-line translation plus unit tests that feed the same `NodeState` to both and compare
the 17 outputs bit-for-bit (after float32 rounding).

## Deployment options for the decision function

| policy type | how to deploy | when |
|---|---|---|
| **rule-based** (`RuleBasedPolicy`) | translate `act()` to C; thresholds from `bundle.model` | baseline, safety fallback, first field test |
| **tabular** (Q-table over a discretised observation) | store the table (6 × #bins) in flash; same binning code as in Python | teaching, tiny MCUs |
| **small MLP** (PPO / DQN actor) | export weights as lists → generate a C array; run with CMSIS-NN, a hand-written dense loop, or **TensorFlow Lite for Microcontrollers** after converting the network (Keras/ONNX → TFLite, int8 quantisation with the observation range `[0,1]` as calibration) | production |
| **ONNX-derived** | ONNX → `onnx2c` / STM32Cube.AI / Edge Impulse | vendor toolchains |
| **quantised NN** | post-training int8 quantisation is safe here: inputs are already in `[0,1]`, outputs are argmax'ed logits | when RAM/flash are tight |

The runtime is not part of this project yet; the boundary is `Policy.act(obs) -> action`
plus the bundle, and nothing else in the package needs to change when it is added.

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
   simulator (trace-driven backends) to re-tune the models and re-train.

## Comparing policies in EdgeEngine AWARE (future work, not implemented)

| method | how it plugs in | notes |
|---|---|---|
| rule-based baseline | `RuleBasedPolicy` | interpretable reference; also the safety fallback |
| random | `RandomPolicy` | sanity floor |
| tabular (Q-learning / SARSA) | discretise the 17 inputs into a few bins each (e.g. SoC × harvest × age × priority × importance) and use `flatten_action` | needs a `gym.ObservationWrapper` that bins; illustrates the curse of dimensionality |
| DQN family | `spaces.Discrete(6)` via a `gym.ActionWrapper` around `unflatten_action` | discrete actions fit naturally |
| PPO / A2C (actor-critic) | Stable-Baselines3 supports `MultiDiscrete` directly (`MlpPolicy`) | recommended first RL baseline; add frame stacking or an LSTM policy for the POMDP |
| evaluation protocol | fixed seed sets, `EpisodeMetrics` + reward, stress configs, randomisation on/off | report mean ± std over ≥ 20 seeds |

A minimal SB3 script (not included in the package to keep dependencies light):

```python
import gymnasium as gym, edgeengine_aware
from stable_baselines3 import PPO
env = gym.make("EdgeEngineAware-v0")
model = PPO("MlpPolicy", env, gamma=0.99, n_steps=2048, verbose=1).learn(2_000_000)
```
