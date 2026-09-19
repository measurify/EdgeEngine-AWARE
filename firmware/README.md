# EdgeEngine AWARE — node-side C runtime

A dependency-free C99 port of the part of the Python package that runs *on the node*:

| C | Python reference |
|---|---|
| `eea_tracker_init / reset / begin_step / on_measurement / on_transmission / set_priority / state` | `observation.NodeStateTracker` |
| `eea_observation_build` | `observation.ObservationBuilder.build` (18 float32 values in [0, 1]) |
| `eea_plan_execution`, `eea_plan_execution_profile` | `actions.plan_execution` |
| `eea_rule_based_act`, `eea_rule_based_tx` | `policies.RuleBasedPolicy.act / _tx` |
| `eea_mlp_forward`, `eea_mlp_act` | `rl.NumpyMLPPolicy.forward / act` |

`eea_policy_data.h` is **generated** from a `PolicyBundle`:

```bash
python tools/export_c.py examples/rl_runs/ppo_default_bundle.json -o firmware/eea_policy_data.h
cc -std=c99 -O2 -I firmware firmware/eea_node.c firmware/main_example.c -lm -o node_example && ./node_example
```

It contains `EEA_PROFILE` (every `NodeProfile` constant) and either `EEA_RULE_PARAMS`
(`policy_type == "rule_based"`) or `EEA_MLP` with the weight arrays (`"mlp"`), plus
`EEA_BUNDLE_VERSION / EXPORTED_AT / NOTES` strings. Regenerate it whenever the bundle changes;
nothing checks it at run time, so a profile that does not match the board's real energy costs
means the policy runs on wrong inputs.

## Equivalence with Python

`tests/test_firmware.py` compiles `eea_node.c` + `test/eea_harness.c` with a header generated
from a rule-based bundle and from random / exported MLP bundles, replays thousands of
tracker events on both sides and checks:

* observations bit-exact (float32) — the tracker and the builder use `double` and round at the
  end exactly as numpy does;
* rule-based actions identical, on realistic event streams and on random observations;
* `plan_execution` identical (levels, modes, energies, rejection reasons);
* MLP logits within 1e-6 and actions identical except at numerical ties.

Run it with `pytest tests/test_firmware.py` (skipped when no C compiler is on the PATH).

## Porting notes

* Keep the tracker in retained RAM across sleep cycles; it is the node's whole memory.
* `eea_tracker_begin_step` takes the *average harvesting power of the interval that just
  elapsed* (energy counted by the harvester monitor since the previous wake-up divided by
  `timestep_s`), never a forecast.
* `eea_tracker_on_transmission` wants the ACK outcome (1/0, or −1 on an unconfirmed link) and,
  when available, the link margin measured from the ACK (LoRaWAN: downlink SNR relative to
  the SF's demodulation floor, or `LinkCheckAns`); it converts it to a mode-independent path-loss
  estimate with the flash link-budget table.
* Sizes: the runtime is ~4 kB of code at `-Os`; the 18→64→64→7 PPO actor is 5.7 k `float`
  weights (~23 kB flash) and 5.7 k multiply-adds per decision — negligible at one decision per
  15 minutes. `EEA_MAX_MODES` (4), `EEA_MAX_LAYERS` (4) and `EEA_MAX_UNITS` (128) bound the
  static buffers.
* Define `EEA_REAL float` for single-precision-only targets; the observation may then differ in
  the last bit from the Python reference.
