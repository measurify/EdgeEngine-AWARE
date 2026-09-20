#!/usr/bin/env bash
# Long training runs for EdgeEngine AWARE, meant to be launched once on a laptop
# and left alone for a few hours (macOS or Linux).
#
#   cd "<repository folder>"
#   bash examples/run_long_training.sh            # PPO 5M steps x 3 seeds, then DQN 2M x 3 seeds
#   bash examples/run_long_training.sh --ppo-only # skip DQN
#   bash examples/run_long_training.sh --dry-run  # print the commands only
#
# What it does
#   1. creates a virtual environment in .venv (if missing) and installs the package with the
#      RL extras (stable-baselines3, sb3-contrib, torch) — first run only, ~2-5 min;
#   2. runs examples/train_seeds.py for every (algorithm, seed) pair, sequentially, writing
#      to examples/rl_runs/seeds_long/ (already-finished runs are skipped, so the script can be
#      re-launched after an interruption);
#   3. on macOS wraps everything in `caffeinate` so the laptop does not sleep.
#
# Output: examples/rl_runs/seeds_long/<algo>_seed<N>.json (evaluation rows + learning curve),
#         <algo>_seed<N>/best_model.zip, <algo>_seed<N>_bundle.json (exported policy),
#         and log.txt with the progress. Expected duration: PPO ~30-60 min per 5M-step run on a
#         recent laptop CPU, DQN ~20-40 min per 2M-step run — about 3-5 h in total.
#
# Monitor from another terminal:   tail -f examples/rl_runs/seeds_long/log.txt
# Stop:                             kill the python process (the finished runs are kept)

set -euo pipefail
cd "$(dirname "$0")/.."

PPO_STEPS=${PPO_STEPS:-5000000}
DQN_STEPS=${DQN_STEPS:-2000000}
SEEDS=${SEEDS:-"0 1 2"}
OUT=${OUT:-examples/rl_runs/seeds_long}
THREADS=${THREADS:-4}
RUN_DQN=1
DRY=0
for a in "$@"; do
  case "$a" in
    --ppo-only) RUN_DQN=0 ;;
    --dry-run) DRY=1 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

# --- environment -----------------------------------------------------------------------
if [ ! -x .venv/bin/python ]; then
  echo "creating .venv and installing the package with the RL extras (first run only)"
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip >/dev/null
  .venv/bin/pip install -e ".[dev,rl]"
fi
PY=.venv/bin/python
$PY -c "import stable_baselines3, torch, edgeengine_aware" || { echo "RL extras missing: run  .venv/bin/pip install -e '.[rl]'"; exit 1; }

mkdir -p "$OUT"
LOG="$OUT/log.txt"
CAFF=""
if [ "$(uname)" = "Darwin" ] && command -v caffeinate >/dev/null; then CAFF="caffeinate -i"; fi

run() {  # run <algo> <seed> <steps> <eval-every>
  local algo=$1 seed=$2 steps=$3 every=$4
  if [ -f "$OUT/${algo}_seed${seed}.json" ]; then
    echo "skip ${algo} seed ${seed} (already done)" | tee -a "$LOG"
    return
  fi
  local cmd="$PY examples/train_seeds.py --algo $algo --seed $seed --steps $steps --eval-every $every --eval-seeds 20 --threads $THREADS --out $OUT"
  echo "$(date '+%F %T') START $cmd" | tee -a "$LOG"
  if [ "$DRY" = 1 ]; then return; fi
  $CAFF $cmd 2>&1 | grep -v -i warning | tee -a "$LOG"
  echo "$(date '+%F %T') DONE  ${algo} seed ${seed}" | tee -a "$LOG"
}

echo "$(date '+%F %T') queue start: PPO ${PPO_STEPS} steps x seeds [${SEEDS}]$( [ $RUN_DQN = 1 ] && echo ", DQN ${DQN_STEPS} steps x seeds [${SEEDS}]" )" | tee -a "$LOG"
for s in $SEEDS; do run ppo "$s" "$PPO_STEPS" 250000; done
if [ "$RUN_DQN" = 1 ]; then
  for s in $SEEDS; do run dqn "$s" "$DQN_STEPS" 100000; done
fi
echo "$(date '+%F %T') QUEUE_DONE" | tee -a "$LOG"
