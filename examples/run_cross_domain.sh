#!/usr/bin/env bash
# Cross-domain training runs for EdgeEngine AWARE: one PPO policy per domain plus a
# "universal" policy trained on the mixture of every domain, each evaluated on the
# scenarios of *all* domains. Feeds section 3 of examples/domains.ipynb.
#
#   cd "<repository folder>"
#   bash examples/run_cross_domain.sh             # PPO 5M steps, 2 seeds, 4 training sets = 8 runs (~2 h on a recent laptop)
#   SEEDS="0" bash examples/run_cross_domain.sh   # one seed only
#   bash examples/run_cross_domain.sh --dry-run
#
# Output: examples/rl_runs/seeds_domains/ppo_<domain>_seed<N>.json (evaluation rows on all 16
# scenarios + learning curve), checkpoints and exported bundles; log.txt with the progress.
# Already-finished runs are skipped, so the script can be re-launched after an interruption.
# The environment (.venv with the RL extras) is created by examples/run_long_training.sh; this
# script creates it too if missing.

set -euo pipefail
cd "$(dirname "$0")/.."

STEPS=${STEPS:-5000000}
SEEDS=${SEEDS:-"0 1"}
DOMAINS=${DOMAINS:-"agriculture indoor_air industrial all"}
OUT=${OUT:-examples/rl_runs/seeds_domains}
THREADS=${THREADS:-4}
DRY=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    *) echo "unknown option $a"; exit 2 ;;
  esac
done

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

echo "$(date '+%F %T') queue start: PPO ${STEPS} steps, domains [${DOMAINS}], seeds [${SEEDS}], evaluated on all domains" | tee -a "$LOG"
for d in $DOMAINS; do
  for s in $SEEDS; do
    tag="ppo_${d}_seed${s}"
    if [ -f "$OUT/${tag}.json" ]; then
      echo "skip ${tag} (already done)" | tee -a "$LOG"; continue
    fi
    cmd="$PY examples/train_seeds.py --algo ppo --domain $d --eval-domain all --tag $tag --seed $s --steps $STEPS --eval-every 250000 --eval-seeds 20 --threads $THREADS --out $OUT"
    echo "$(date '+%F %T') START $cmd" | tee -a "$LOG"
    if [ "$DRY" = 1 ]; then continue; fi
    $CAFF $cmd 2>&1 | grep -v -i warning | tee -a "$LOG"
    echo "$(date '+%F %T') DONE  ${tag}" | tee -a "$LOG"
  done
done
echo "$(date '+%F %T') QUEUE_DONE" | tee -a "$LOG"
