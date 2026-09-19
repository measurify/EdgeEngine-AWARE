"""Helpers for training and evaluating RL agents on EdgeEngine AWARE.

Nothing here depends on a specific RL library except :class:`SB3Policy`, which
only needs an object with a ``predict(obs, deterministic=True)`` method (the
Stable-Baselines3 convention). Import of Stable-Baselines3 itself is left to
the caller, so the core package stays dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import N_FLAT_ACTIONS, flatten_action, unflatten_action
from .env import EdgeEngineAwareEnv
from .interfaces import Policy
from .policies import run_episode
from .scenarios import SCENARIOS, get_scenario


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------
class FlatActionWrapper(gym.ActionWrapper):
    """Expose the ``MultiDiscrete([3, 2])`` action as ``Discrete(6)``.

    ``flat = sensing_level * 2 + transmit`` (see ``actions.py``). Needed by
    value-based agents such as DQN; PPO/A2C work on the MultiDiscrete space directly.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(N_FLAT_ACTIONS)

    def action(self, action):
        return unflatten_action(int(action))


class MixedScenarioEnv(EdgeEngineAwareEnv):
    """EdgeEngine AWARE environment that draws a *scenario* at every reset.

    Training on a mixture of scenarios (plus domain randomisation inside each)
    is the simplest way to obtain a policy that is robust to weather, storage
    size, link quality and application behaviour, instead of one tuned to the
    nominal configuration. The scenario of the current episode is reported in
    ``info["scenario"]``.
    """

    def __init__(self, scenarios: Iterable[str], *, randomize: bool = True, render_mode: str | None = None):
        self.scenario_names = list(scenarios)
        if not self.scenario_names:
            raise ValueError("at least one scenario name is required")
        self._configs = {n: get_scenario(n, randomize=randomize) for n in self.scenario_names}
        self.current_scenario = self.scenario_names[0]
        super().__init__(self._configs[self.current_scenario], render_mode=render_mode)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:  # seed the scenario draw as well, for reproducibility
            super().reset(seed=seed)
        self.current_scenario = self.scenario_names[int(self.np_random.integers(len(self.scenario_names)))]
        self.base_config = self._configs[self.current_scenario]
        obs, info = super().reset(seed=None, options=options)
        info["scenario"] = self.current_scenario
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["scenario"] = self.current_scenario
        return obs, reward, terminated, truncated, info


def make_env(scenario: str | Iterable[str] = "default", *, randomize: bool = False, flat_actions: bool = False, seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    """Build an environment for a named scenario, or for a mixture of scenarios
    (``"mixed"`` = all of them, or an explicit list of names), optionally with
    domain randomisation and flat actions. ``seed`` seeds the first reset."""
    if scenario == "mixed":
        scenario = list(SCENARIOS)
    if isinstance(scenario, str):
        env: gym.Env = EdgeEngineAwareEnv(get_scenario(scenario, randomize=randomize), render_mode=render_mode)
    else:
        env = MixedScenarioEnv(scenario, randomize=randomize, render_mode=render_mode)
    if flat_actions:
        env = FlatActionWrapper(env)
    if seed is not None:
        env.reset(seed=seed)
    return env


def make_env_fn(scenario: str | Iterable[str] = "default", *, randomize: bool = True, flat_actions: bool = False, seed: int = 0) -> Callable[[], gym.Env]:
    """Factory usable with SB3 ``DummyVecEnv`` / ``SubprocVecEnv``."""

    def _init() -> gym.Env:
        return make_env(scenario, randomize=randomize, flat_actions=flat_actions, seed=seed)

    return _init


# ---------------------------------------------------------------------------
# Policy adapter
# ---------------------------------------------------------------------------
class SB3Policy:
    """Wrap a trained Stable-Baselines3 model as an EdgeEngine AWARE ``Policy``.

    The adapter always returns the MultiDiscrete action array, undoing the flat
    encoding when the model was trained with :class:`FlatActionWrapper`.
    """

    def __init__(self, model: Any, flat_actions: bool = False, deterministic: bool = True):
        self.model = model
        self.flat_actions = flat_actions
        self.deterministic = deterministic
        self.reset()

    def reset(self) -> None:
        self._state = None  # recurrent policies would keep their hidden state here

    def act(self, observation) -> np.ndarray:
        action, self._state = self.model.predict(np.asarray(observation, dtype=np.float32), state=self._state, deterministic=self.deterministic)
        if self.flat_actions:
            return unflatten_action(int(np.asarray(action).reshape(-1)[0]))
        return np.asarray(action, dtype=np.int64).reshape(-1)



# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------
@dataclass
class EvalRow:
    policy: str
    scenario: str
    seed: int
    reward: float
    utility: float
    harvested_j: float
    consumed_j: float
    n_sensing: int
    n_high_quality: int
    n_tx: int
    n_delivered: int
    min_soc: float
    low_battery_frac: float
    depletions: int
    aoi_h: float
    max_aoi_h: float
    components: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "components"}
        d.update({f"c_{k}": v for k, v in self.components.items()})
        return d


def evaluate(
    policies: dict[str, Callable[[], Policy]],
    scenarios: Iterable[str] = ("default",),
    seeds: Iterable[int] = range(1000, 1010),
    *,
    randomize: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[EvalRow]:
    """Run every policy on every scenario and seed; return one row per episode.

    Policies are given as factories so that stateful policies start fresh.
    Seeds default to a held-out range (1000+) that training never touches.
    """
    rows: list[EvalRow] = []
    seeds = list(seeds)
    for scenario in scenarios:
        env = make_env(scenario, randomize=randomize)
        for name, factory in policies.items():
            if progress:
                progress(f"{scenario:22s} {name}")
            for seed in seeds:
                res = run_episode(env, factory(), seed=seed)
                m = env.unwrapped.metrics
                rows.append(
                    EvalRow(
                        policy=name,
                        scenario=scenario,
                        seed=seed,
                        reward=res.total_reward,
                        utility=m.total_application_utility,
                        harvested_j=m.total_harvested_energy_j,
                        consumed_j=m.total_consumed_energy_j,
                        n_sensing=m.n_sensing,
                        n_high_quality=m.n_high_quality_sensing,
                        n_tx=m.n_transmissions,
                        n_delivered=m.n_successful_transmissions,
                        min_soc=m.min_battery_soc,
                        low_battery_frac=m.fraction_low_battery,
                        depletions=m.battery_depletion_events,
                        aoi_h=m.average_aoi_s / 3600.0,
                        max_aoi_h=m.max_aoi_s / 3600.0,
                        components=dict(m.reward_components),
                    )
                )
        env.close()
    return rows


def summarize(rows: list[EvalRow], metric: str = "reward") -> dict[str, dict[str, tuple[float, float]]]:
    """``{scenario: {policy: (mean, std)}}`` for one metric."""
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for r in rows:
        out.setdefault(r.scenario, {}).setdefault(r.policy, [])  # type: ignore[arg-type]
        out[r.scenario][r.policy].append(getattr(r, metric))  # type: ignore[union-attr]
    return {s: {p: (float(np.mean(v)), float(np.std(v))) for p, v in d.items()} for s, d in out.items()}


__all__ = ["FlatActionWrapper", "MixedScenarioEnv", "make_env", "make_env_fn", "SB3Policy", "EvalRow", "evaluate", "summarize", "flatten_action"]


# ---------------------------------------------------------------------------
# Export of small MLP policies (SB3 -> plain numbers -> numpy runtime)
# ---------------------------------------------------------------------------
def export_sb3_mlp(model: Any) -> dict[str, Any]:
    """Extract the actor of an SB3 ``PPO``/``A2C`` (MultiDiscrete) or ``DQN``
    (flat Discrete) MlpPolicy as nested lists, ready for ``PolicyBundle.model``.

    Returned dict::

        {"type": "mlp", "input_dim": 17,
         "layers": [{"W": [[...]], "b": [...], "activation": "tanh"|"relu"|"linear"}, ...],
         "output": "multidiscrete_logits" | "flat_q_values",
         "output_split": [3, 2]}            # only for multidiscrete_logits
    """
    import torch  # local import: torch is only needed when exporting

    policy = model.policy
    layers: list[dict[str, Any]] = []

    def add_sequential(seq, final_activation: str | None = None):
        mods = [m for m in seq if not isinstance(m, torch.nn.Flatten)]
        pending: dict | None = None
        for m in mods:
            if isinstance(m, torch.nn.Linear):
                if pending is not None:
                    pending["activation"] = "linear"
                    layers.append(pending)
                pending = {"W": m.weight.detach().cpu().numpy().tolist(), "b": m.bias.detach().cpu().numpy().tolist()}
            elif isinstance(m, (torch.nn.Tanh, torch.nn.ReLU)):
                assert pending is not None, "activation without a preceding Linear layer"
                pending["activation"] = "tanh" if isinstance(m, torch.nn.Tanh) else "relu"
                layers.append(pending)
                pending = None
            else:
                raise TypeError(f"unsupported module in policy network: {type(m).__name__}")
        if pending is not None:
            pending["activation"] = final_activation or "linear"
            layers.append(pending)

    if hasattr(policy, "q_net"):  # DQN
        add_sequential(policy.q_net.q_net, final_activation="linear")
        output, split = "flat_q_values", None
    else:  # on-policy actor-critic
        add_sequential(policy.mlp_extractor.policy_net)
        add_sequential(torch.nn.Sequential(policy.action_net), final_activation="linear")
        output = "multidiscrete_logits"
        split = [int(n) for n in model.action_space.nvec]
    return {"type": "mlp", "input_dim": len(layers[0]["W"][0]), "layers": layers, "output": output, "output_split": split}


class NumpyMLPPolicy:
    """Dependency-free runtime for an exported MLP (what a C port would do).

    Runs the forward pass in float32 with numpy only and applies the argmax
    decoding of the action encoding, so a bundle exported with
    :func:`export_sb3_mlp` can be executed without torch or SB3 - and compared
    action-by-action with the original model (see the RL notebook).
    """

    def __init__(self, model: dict[str, Any]):
        self.layers = [(np.asarray(l["W"], dtype=np.float32), np.asarray(l["b"], dtype=np.float32), l["activation"]) for l in model["layers"]]
        self.output = model["output"]
        self.split = model.get("output_split")

    def reset(self) -> None:
        pass

    def forward(self, observation) -> np.ndarray:
        h = np.asarray(observation, dtype=np.float32).reshape(-1)
        for W, b, act in self.layers:
            h = W @ h + b
            if act == "tanh":
                h = np.tanh(h)
            elif act == "relu":
                h = np.maximum(h, 0.0)
        return h

    def act(self, observation) -> np.ndarray:
        out = self.forward(observation)
        if self.output == "flat_q_values":
            return unflatten_action(int(np.argmax(out)))
        n1 = self.split[0]
        return np.array([int(np.argmax(out[:n1])), int(np.argmax(out[n1:]))], dtype=np.int64)


__all__ += ["export_sb3_mlp", "NumpyMLPPolicy"]
