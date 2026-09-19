# Action encoding

`action_space = gymnasium.spaces.MultiDiscrete([3, 1 + n_modes])` — a pair
`(sensing_level, transmit)`; `[3, 4]` with the default three-mode radio.

| component | value | operation on the hardware | default energy | default noise σ / link budget |
|---|---|---|---|---|
| `sensing_level` | 0 | sensor stays off, the stored sample is kept (it ages) | 0 J | — |
| | 1 | low-cost acquisition: single sample, short warm-up | 0.10 J | σ = 0.040 |
| | 2 | high-quality acquisition: averaging, longer warm-up | 0.60 J | σ = 0.010 |
| `transmit` | 0 | radio off | 0 J | |
| | 1 | uplink of the **latest stored sample** in the *fast* mode (SF7-like) | 0.30 J | sensitivity −123 dBm, mean margin −2 dB |
| | 2 | uplink in the *standard* mode (SF9-like) | 0.60 J | sensitivity −129 dBm, mean margin +4 dB |
| | 3 | uplink in the *robust* mode (SF12-like) | 1.20 J | sensitivity −137 dBm, mean margin +12 dB |

Within a step, **sensing is executed before transmission**, so `(2, 2)` means "take a good
sample and send it now in the standard mode" and `(0, 3)` means "re-send what I already have,
robustly". A transmission is delivered with probability `1 / (1 + exp(-margin / 1.5 dB))`,
where the margin depends on the slowly varying path loss (see `docs/modeling.md`); the node
learns the path loss from the margin reported in the ACK (observation `path_loss_est`) and
from lost uplinks (a loss in mode k means the path loss is at least mode k's link budget).

The mode set is configuration (`CommunicationConfig.modes`); `config.single_mode_radio()` gives
the binary `MultiDiscrete([3, 2])` action of a radio with one setting.

## Flat encoding

Tabular methods and DQN-like agents need a single discrete index:

```
flat = sensing_level * (1 + n_modes) + transmit          # in [0, 12) for three modes
sensing_level, transmit = divmod(flat, 1 + n_modes)
```

`edgeengine_aware.actions.flatten_action` / `unflatten_action` implement it and
`rl.FlatActionWrapper` exposes it as `spaces.Discrete(12)`.

## Feasibility rule (identical in simulation and firmware)

`edgeengine_aware.actions.plan_execution` decides which part of a requested action is
executed, using only the stored energy the node can read:

```
available = E_stored - baseline_energy_of_the_interval - reserve
sensing executed   if  E_sense[level] <= available          (else rejected, available unchanged)
transmit executed  if  E_tx[mode] <= available - E_sense_executed and a sample exists
```

Rejected sub-actions cost no energy, are reported in `info["rejected"]` and incur
`lambda_reject` (0.2) in the reward. The rule is deliberately simple: the brown-out
protection of a real power path behaves the same way (optional loads are shed first, the
MCU keeps its baseline).

## Why this action space

* Small enough for tabular/DQN agents (12 actions) and for hand-written C.
* Each value is a concrete firmware branch (`sensor_read(mode)`, `radio_send(sf, power)`).
* It captures the three levers of an energy-harvesting monitoring node — *how well to look*,
  *whether to speak* and *how loudly* — the last one being where a fixed rule is hardest to
  tune by hand, because the right mode depends on a link state the node only observes
  indirectly.

## Extensions that fit without changing the contract

More radio settings are entries in `CommunicationConfig.modes` (the action space grows with
them). A genuinely new decision (`payload_size`, `compress ∈ {0,1}`, `run_edge_inference ∈
{0,1}`) is a new column in the `MultiDiscrete` vector: a branch in `plan_execution`, the
corresponding energy entry in the hardware profile and a `Radio`/`Sensor` implementation that
understands the new parameter. Policies and the observation contract are untouched.
