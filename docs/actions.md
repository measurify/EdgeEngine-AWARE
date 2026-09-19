# Action encoding

`action_space = gymnasium.spaces.MultiDiscrete([3, 2])` — a pair `(sensing_level, transmit)`.

| component | value | operation on the hardware | default energy | default noise σ |
|---|---|---|---|---|
| `sensing_level` | 0 | sensor stays off, the stored sample is kept (it ages) | 0 J | — |
| | 1 | low-cost acquisition: single sample, short warm-up | 0.10 J | 0.040 |
| | 2 | high-quality acquisition: averaging, longer warm-up | 0.60 J | 0.010 |
| `transmit` | 0 | radio off | 0 J | |
| | 1 | one uplink attempt carrying the **latest stored sample** | 0.60 J | success p ≈ 0.9 |

Within a step, **sensing is executed before transmission**, so `(2, 1)` means "take a
good sample and send it now" and `(0, 1)` means "re-send what I already have".

## Flat encoding

Tabular methods and DQN-like agents need a single discrete index:

```
flat = sensing_level * 2 + transmit          # in [0, 6)
sensing_level, transmit = divmod(flat, 2)
```

`edgeengine_aware.actions.flatten_action` / `unflatten_action` implement it; a Gymnasium
wrapper with `spaces.Discrete(6)` is a few lines (see README).

## Feasibility rule (identical in simulation and firmware)

`edgeengine_aware.actions.plan_execution` decides which part of a requested action is
executed, using only the stored energy the node can read:

```
available = E_stored - baseline_energy_of_the_interval - reserve
sensing executed   if  E_sense[level] <= available          (else rejected, available unchanged)
transmit executed  if  E_tx <= available - E_sense_executed and a sample exists
```

Rejected sub-actions cost no energy, are reported in `info["rejected"]` and incur
`lambda_reject` (0.2) in the reward. The rule is deliberately simple: the brown-out
protection of a real power path behaves the same way (optional loads are shed first, the
MCU keeps its baseline).

## Why this action space

* Small enough for tabular/DQN agents (6 actions) and for hand-written C.
* Each value is a concrete firmware branch (`sensor_read(mode)`, `radio_send()`).
* It captures the two levers of an energy-harvesting monitoring node — *how well to look*
  and *whether to speak* — without committing to a radio technology.

## Extensions that fit without changing the contract

Adding a component to the `MultiDiscrete` vector (e.g. `tx_power ∈ {0,1,2}`, `spreading_factor`,
`payload_size`, `compress ∈ {0,1}`, `run_edge_inference ∈ {0,1}`) requires: a new column in
`ACTION_NVEC`, a branch in `plan_execution`, the corresponding energy entry in the hardware
profile and a `Radio`/`Sensor` implementation that understands the new parameter. Policies
and the observation contract are untouched.
