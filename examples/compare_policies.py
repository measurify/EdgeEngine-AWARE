"""Headless comparison of the built-in baseline policies.

    python examples/compare_policies.py --seeds 10
    python examples/compare_policies.py --randomize        # domain randomisation on
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edgeengine_aware as ea  # noqa: E402
from edgeengine_aware.policies import AlwaysOnPolicy, PeriodicPolicy, RandomPolicy, RuleBasedPolicy, run_episode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--randomize", action="store_true", help="enable domain randomisation")
    ap.add_argument("--days", type=float, default=None, help="override episode length in days")
    args = ap.parse_args()

    cfg = ea.default_config()
    cfg.randomization.enabled = args.randomize
    if args.days:
        cfg.time.episode_days = args.days
    env = ea.EdgeEngineAwareEnv(cfg)
    profile = ea.NodeProfile.from_config(cfg)  # gives the rule-based policy its link-budget table (radio-mode choice)

    policies = {
        "rule-based": lambda: RuleBasedPolicy(profile=profile),
        "random": lambda: RandomPolicy(seed=0),
        "periodic 1h high": lambda: PeriodicPolicy(4, 2),
        "periodic 1h low": lambda: PeriodicPolicy(4, 1),
        "periodic 3h high": lambda: PeriodicPolicy(12, 2),
        "always on": AlwaysOnPolicy,
    }
    print(f"{'policy':18s} {'reward':>8s} {'utility':>8s} {'harv J':>7s} {'cons J':>7s} {'sens':>5s} {'tx':>5s} {'deliv':>5s} {'minSoC':>7s} {'low%':>5s} {'depl':>5s} {'AoI h':>6s}")
    for name, factory in policies.items():
        rows = []
        for seed in range(args.seeds):
            res = run_episode(env, factory(), seed=seed)
            m = env.metrics
            rows.append([res.total_reward, m.total_application_utility, m.total_harvested_energy_j, m.total_consumed_energy_j, m.n_sensing, m.n_transmissions, m.n_successful_transmissions, m.min_battery_soc, 100 * m.fraction_low_battery, m.battery_depletion_events, m.average_aoi_s / 3600])
        a = np.mean(rows, axis=0)
        print(f"{name:18s} {a[0]:8.1f} {a[1]:8.1f} {a[2]:7.0f} {a[3]:7.0f} {a[4]:5.0f} {a[5]:5.0f} {a[6]:5.0f} {a[7]:7.2f} {a[8]:5.0f} {a[9]:5.1f} {a[10]:6.2f}")


if __name__ == "__main__":
    main()
