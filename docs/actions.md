# Actions and the feasibility rule

## The action

At every decision the policy returns two integers, `(sensing_level, transmit)`, i.e. an
element of the Gymnasium space `MultiDiscrete([3, 1 + n_modes])` — `MultiDiscrete([3, 4])`
with the default three-mode radio.

| component | value | what the node does | energy (default) | sensor noise / radio link (default) |
|---|---|---|---|---|
| `sensing_level` | 0 | the sensor stays off; the stored reading is kept and keeps ageing | 0 J | — |
| | 1 | **cheap reading**: one sample, short warm-up | 0.10 J | noise σ = 0.040 (fraction of field capacity) |
| | 2 | **accurate reading**: averaging, longer warm-up | 0.60 J | noise σ = 0.010 |
| `transmit` | 0 | the radio stays off | 0 J | — |
| | 1 | send the **latest stored reading** in the *fast* mode (SF7-like) | 0.30 J | receiver sensitivity −123 dBm → mean margin −2 dB |
| | 2 | send it in the *standard* mode (SF9-like) | 0.60 J | sensitivity −129 dBm → mean margin +4 dB |
| | 3 | send it in the *robust* mode (SF12-like) | 1.20 J | sensitivity −137 dBm → mean margin +12 dB |

The "mean margin" column assumes the default transmit power of 14 dBm and the default mean
path loss of 139 dB: `margin = 14 − 139 − sensitivity`. Averaged over the channel's fading
the three modes deliver about 36 % / 75 % / 98 % of the uplinks; when the channel happens to
be good, even the fast mode gets through, which is what makes the choice interesting.

Within one step, **sensing happens before transmission**, so `(2, 2)` means "take an accurate
reading and send it now in the standard mode", and `(0, 3)` means "re-send what I already
have, robustly". A transmission always carries the most recent stored reading.

### How delivery is decided

An uplink in mode *k* is delivered with probability

    p_k = 1 / (1 + exp(−(margin_k + fast_fading) / 1.5 dB))
    margin_k = tx_power_k − path_loss(t) − sensitivity_k

where `path_loss(t)` = 139 dB + a slowly varying term (AR(1) shadowing, σ = 5 dB, correlation
time ≈ 8 h) and `fast_fading` is redrawn at every attempt (σ = 2 dB). The 1.5 dB scale makes
the transition from "almost never" to "almost always" about 6 dB wide. Details in
`docs/modeling.md` §2. The node never sees `path_loss(t)`; it estimates it from the margin
reported in acknowledgements and from lost uplinks (observation component `path_loss_est`,
see `docs/observation.md`).

### Changing the radio

The mode set is configuration: `CommunicationConfig.modes` is a tuple of `RadioModeConfig(name,
energy_j, tx_power_dbm, sensitivity_dbm)`; the action space grows with it. For a radio with a
single setting use the whole configuration object returned by
`config.single_mode_radio(energy_j, mean_margin_db)` — it also sets `reference_mode = 0` and
the path loss consistent with the requested margin:

```python
import edgeengine_aware as ea

cfg = ea.default_config()
cfg.communication = ea.config.single_mode_radio(energy_j=0.6, mean_margin_db=4.0)
# action space is now MultiDiscrete([3, 2]): transmit is a yes/no decision
```

(Assigning only `.modes` and leaving `reference_mode = 1` fails validation, because the
reference mode must exist.)

## Flat encoding (for DQN and tabular methods)

Some algorithms need a single integer instead of a pair:

```
flat = sensing_level * (1 + n_modes) + transmit          # 0 .. 11 with three modes
sensing_level, transmit = divmod(flat, 1 + n_modes)
```

`actions.flatten_action` / `actions.unflatten_action` implement it and `rl.FlatActionWrapper`
exposes the environment with the space `Discrete(3 · (1 + n_modes))` = `Discrete(12)`.

## The feasibility rule

The policy *requests* an action; the node executes only the part it can afford. The rule,
`actions.plan_execution`, uses only the energy the node can read from its gauge and is the
same in the simulator, in the Python controller and in the C runtime:

```
available = E_stored − E_baseline − E_reserve
    E_baseline = always-on power × interval   (200 µW × 900 s = 0.18 J by default)
    E_reserve  = reserve_soc × E_max           (2 % × 300 J = 6 J by default)

sensing  is executed if E_sense[level] ≤ available;      then available −= E_sense[level]
         otherwise rejected with reason "sensing:insufficient_energy"
transmit is rejected with reason "transmit:no_measurement" if there is nothing to send
         (no stored reading and no sensing in this step);
         executed if E_tx[mode] ≤ available;
         otherwise rejected with reason "transmit:insufficient_energy"
```

Rejected operations cost no energy, appear in `info["rejected"]` (the reason strings above)
and cost `lambda_reject` = 0.2 each in the reward. The rule deliberately mimics the brown-out
protection of a real power path: optional loads are shed first, the microcontroller keeps its
baseline. Sensing has priority over transmission because a transmission without a fresh
reading is worth little.

The rule returns an `ExecutionPlan` (executed sensing level, whether a transmission happens
and in which mode, the energies, the rejection reasons). The C twin is
`eea_plan_execution` / `eea_plan_execution_profile` in `firmware/eea_node.c`.

## Why this action space

* It is small: 12 combinations, fine for tabular methods, DQN and hand-written C.
* Each value is a concrete firmware branch: `sensor_read(level)`, `radio_send(mode)`.
* It captures the three levers of an energy-harvesting monitoring node — *how well to look*,
  *whether to speak*, *how loudly* — the last one being where a fixed rule is hardest to tune
  by hand, because the right mode depends on a link state the node observes only indirectly.

## Extensions that keep the contract

More radio settings are more entries in `CommunicationConfig.modes`. A genuinely new
decision (payload size, on-device compression, running an edge-inference model) is a new
column in the `MultiDiscrete` vector: a branch in `plan_execution`, an energy entry in the
hardware profile, and a `Radio` / `Sensor` implementation that understands the new
parameter. Policies and the observation are untouched.
