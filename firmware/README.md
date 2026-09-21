# EdgeEngine AWARE — node-side C runtime

This folder contains a C99 translation of the part of the Python package that runs *on the
node*: the bookkeeping of what the node knows, the construction of the 18-number
observation, the energy-feasibility rule, the rule-based controller and the forward pass of
a small dense neural network. It has no dependency beyond `<math.h>`, allocates no memory,
and is tested to give the same results as the Python reference.

| C function(s) | Python reference |
|---|---|
| `eea_tracker_init / reset / begin_step / on_measurement / on_transmission / set_priority / state` | `observation.NodeStateTracker` |
| `eea_observation_build` | `observation.ObservationBuilder.build` — 18 `float` values in [0, 1] |
| `eea_plan_execution`, `eea_plan_execution_profile` | `actions.plan_execution` |
| `eea_rule_based_act`, `eea_rule_based_tx` | `policies.RuleBasedPolicy.act` and its radio-mode choice |
| `eea_mlp_forward`, `eea_mlp_act` | `rl.NumpyMLPPolicy.forward / act` |

## Files

| file | role |
|---|---|
| `eea_node.h`, `eea_node.c` | the runtime (about 4 kB of code compiled with `-Os`) |
| `eea_policy_data.h` | **generated, not committed** — the constants of one policy bundle (see below) |
| `main_example.c` | a complete decision loop with stub hardware functions; compiles and runs on a laptop |
| `test/eea_harness.c` | reads commands on standard input and prints results; used only by `tests/test_firmware.py` |

## Generating the policy data header

Everything a device must know about a policy is in a `PolicyBundle` JSON file (see
`docs/deployment.md`). `tools/export_c.py` turns it into a header:

```bash
python tools/export_c.py examples/bundles/ppo_default.json -o firmware/eea_policy_data.h
cc -std=c99 -O2 -I firmware firmware/eea_node.c firmware/main_example.c -lm -o node_example
./node_example
```

The header defines:

* `EEA_PROFILE` (`eea_profile_t`) — every `NodeProfile` constant: sensing energies and noise
  levels, radio energies / transmit powers / sensitivities per mode, the reference mode, the
  two thresholds and the side of the danger (`critical_is_upper`: low for soil moisture, high
  for CO₂ or a temperature), the decision interval, the always-on power, the brown-out
  reserve, whether ACKs are available, and the observation normalisation constants;
* either `EEA_RULE_PARAMS` (`eea_rule_params_t`, with `#define EEA_HAS_RULE_PARAMS`) for a
  rule-based bundle, or `EEA_MLP` (`eea_mlp_t`, with `#define EEA_HAS_MLP`) plus the weight
  arrays `EEA_L<i>_W` / `EEA_L<i>_b` for a network bundle;
* `EEA_BUNDLE_POLICY_TYPE`, `EEA_BUNDLE_VERSION`, `EEA_BUNDLE_EXPORTED_AT`, `EEA_BUNDLE_NOTES`,
  `EEA_BUNDLE_OBS_DIM` — string/integer macros with the bundle's metadata, so that a deployed
  image can report which policy it runs.

Numbers are written with enough digits to be read back exactly. Regenerate the header
whenever the bundle changes. Nothing checks at run time that the profile matches the board's
real energy costs — if it does not, the policy runs on wrong inputs — so always generate the
header from the bundle that was validated in simulation.

## Equivalence with Python

`tests/test_firmware.py` compiles `eea_node.c` and `test/eea_harness.c` with headers
generated from a rule-based bundle of each of the three domain profiles (agriculture,
indoor air, industrial — different energies, radio tables and danger side) and from network
bundles (random ones and the shipped `ppo_default.json`), replays thousands of tracker events
on both sides and checks that:

* the 18 observation values are **bit-exact** (`float32`) — the tracker and the observation
  builder compute in `double`, like Python, and round to `float` at the end, like numpy;
* rule-based actions are identical, on realistic event streams and on random observations
  covering every branch;
* `plan_execution` results are identical (levels, modes, energies, rejection reasons);
* network outputs agree to about 1e-6 and actions are identical, except when two outputs
  are equal to that precision (a tie), which the test tolerates.

Run it with `pytest tests/test_firmware.py`; it is skipped when no C compiler (`cc`, `gcc`
or `clang`) is found.

## Porting notes

* Keep the `eea_tracker_t` in RAM that survives sleep; it is the node's whole memory.
* `eea_tracker_begin_step` wants the *average harvesting power of the interval that just
  ended*: the energy counted by the harvester monitor since the previous wake-up, divided by
  `timestep_s`. Never a forecast. Pass `priority = -1` when no new downlink was received.
* `eea_tracker_on_transmission` wants the outcome of the uplink: `1` = acknowledged, `0` =
  not acknowledged, `-1` = the link gives no confirmations. When an ACK is available, also
  pass the **link margin** the gateway measured (in LoRaWAN: the downlink SNR relative to the
  demodulation floor of the spreading factor used, or the value returned by a `LinkCheckReq`);
  the runtime converts it into the mode-independent path-loss estimate with the link-budget
  table of the profile. Pass `has_margin = false` if the value is not available.
* Sizes: the runtime is ~4 kB of code; the 18→64→64→7 PPO network is 5 831 `float` weights
  (~23 kB of flash) and as many multiply-adds per decision — negligible at one decision every
  15 minutes. `EEA_MAX_MODES` (4), `EEA_MAX_LAYERS` (4) and `EEA_MAX_UNITS` (128) bound the
  static buffers; raise them if your bundle needs more.
* `EEA_REAL` defaults to `double`. Define it as `float` for a target with a single-precision
  FPU only; the observation may then differ from Python in the last bit (the equivalence test
  does not cover that build).
