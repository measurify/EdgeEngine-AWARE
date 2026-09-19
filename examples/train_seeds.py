"""Train one agent (one algorithm, one seed) and evaluate it on every scenario.

    python examples/train_seeds.py --algo ppo --seed 0 --steps 1000000
    python examples/train_seeds.py --algo ppo_stack --seed 0        # PPO + frame stacking (4)
    python examples/train_seeds.py --algo dqn --seed 1 --steps 500000
    python examples/train_seeds.py --algo rppo --seed 0             # RecurrentPPO (needs sb3-contrib)

Every run writes ``examples/rl_runs/seeds/<algo>_seed<N>.json`` (evaluation rows,
learning curve, metadata) and the best checkpoint next to it. The notebook
``examples/train_rl.ipynb`` launches the missing runs and aggregates the JSON
files, so this script is also what you call to add seeds or algorithms later.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from stable_baselines3 import DQN, PPO  # noqa: E402
from stable_baselines3.common.callbacks import EvalCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack  # noqa: E402

import edgeengine_aware as ea  # noqa: E402
from edgeengine_aware.rl import SB3Policy, StackedPolicy, evaluate, make_env, make_env_fn  # noqa: E402
from edgeengine_aware.scenarios import SCENARIOS  # noqa: E402

ALGOS = ("ppo", "dqn", "ppo_stack", "rppo")
N_STACK = 4
NET = [64, 64]


def build(algo: str, seed: int, n_envs: int, log_dir: Path):
    """Return (model, eval_callback, policy_factory) for ``algo``."""
    flat = algo == "dqn"
    scenarios = list(SCENARIOS)

    def vec(seed_offset: int):
        v = DummyVecEnv([make_env_fn(scenarios, randomize=True, flat_actions=flat, seed=seed * 1000 + seed_offset + i) for i in range(n_envs)])
        return VecFrameStack(v, n_stack=N_STACK) if algo == "ppo_stack" else v

    train_env = vec(0)
    eval_env = DummyVecEnv([lambda: Monitor(make_env(scenarios, randomize=True, flat_actions=flat, seed=10_000 + seed))])
    if algo == "ppo_stack":
        eval_env = VecFrameStack(eval_env, n_stack=N_STACK)

    common = dict(policy_kwargs=dict(net_arch=NET), seed=seed, device="cpu", verbose=0)
    if algo in ("ppo", "ppo_stack"):
        model = PPO("MlpPolicy", train_env, n_steps=256, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2, **common)
    elif algo == "dqn":
        model = DQN("MlpPolicy", train_env, learning_rate=5e-4, buffer_size=200_000, learning_starts=20_000, batch_size=256, gamma=0.99, train_freq=4, gradient_steps=1, target_update_interval=5_000, exploration_fraction=0.3, exploration_final_eps=0.05, **common)
    elif algo == "rppo":
        from sb3_contrib import RecurrentPPO

        common["policy_kwargs"] = dict(net_arch=NET, lstm_hidden_size=32, n_lstm_layers=1, shared_lstm=False, enable_critic_lstm=True)
        model = RecurrentPPO("MlpLstmPolicy", train_env, n_steps=256, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95, learning_rate=3e-4, ent_coef=0.01, clip_range=0.2, **common)
    else:
        raise ValueError(algo)

    def policy_factory(m):
        if algo == "dqn":
            return lambda: SB3Policy(m, flat_actions=True)
        if algo == "ppo_stack":
            return lambda: StackedPolicy(SB3Policy(m), N_STACK)
        return lambda: SB3Policy(m)  # PPO and RecurrentPPO (SB3Policy carries the LSTM state)

    return model, eval_env, policy_factory


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=ALGOS, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--eval-episodes", type=int, default=12)
    ap.add_argument("--eval-seeds", type=int, default=20, help="held-out seeds per scenario for the final evaluation")
    ap.add_argument("--out", type=Path, default=ROOT / "examples" / "rl_runs" / "seeds")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.algo}_seed{args.seed}"
    log_dir = args.out / tag
    log_dir.mkdir(exist_ok=True)

    model, eval_env, policy_factory = build(args.algo, args.seed, args.n_envs, log_dir)
    cb = EvalCallback(eval_env, n_eval_episodes=args.eval_episodes, eval_freq=max(1, args.eval_every // args.n_envs), best_model_save_path=str(log_dir), log_path=str(log_dir), deterministic=True, verbose=0)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=cb, progress_bar=False)
    train_s = time.time() - t0
    model.save(log_dir / "last_model.zip")
    best = type(model).load(log_dir / "best_model.zip", device="cpu")

    rows = evaluate({args.algo: policy_factory(best)}, SCENARIOS, seeds=range(1000, 1000 + args.eval_seeds))
    curve = np.load(log_dir / "evaluations.npz")
    result = {
        "algo": args.algo,
        "seed": args.seed,
        "steps": args.steps,
        "train_seconds": train_s,
        "n_stack": N_STACK if args.algo == "ppo_stack" else 1,
        "curve": {"timesteps": curve["timesteps"].tolist(), "mean": curve["results"].mean(axis=1).tolist(), "std": curve["results"].std(axis=1).tolist()},
        "rows": [r.as_dict() for r in rows],
        "package_version": ea.__version__,
    }
    (args.out / f"{tag}.json").write_text(json.dumps(result))
    by_scen = {}
    for r in rows:
        by_scen.setdefault(r.scenario, []).append(r.reward)
    print(f"{tag}: trained {args.steps:,} steps in {train_s:.0f} s; reward per scenario: " + ", ".join(f"{s}={np.mean(v):.1f}" for s, v in by_scen.items()))


if __name__ == "__main__":
    main()
