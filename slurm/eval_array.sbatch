#!/usr/bin/env bash
#SBATCH --job-name=lucy-eval
#SBATCH --partition=cpu
#SBATCH --account=pi_machta_umass_edu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err
# Pairwise eval array: each Lucy variant vs OpenSpiel baseline at matched
# iteration budgets. Index layout:
#
#   index = ((lucy_variant_idx * len(ITERS) + iter_idx) * len(SEEDS) + seed_idx)
#
# At matched iters lucy_iters == baseline_iters; mismatched eval (frontier
# table) is run via a second eval_array_full.sbatch invocation.
set -euo pipefail

ITERS_LIST=(${LUCY_BENCH_ITERS:-1000 10000 100000 1000000})
SEEDS=(${LUCY_BENCH_SEEDS:-1 2 3})
LUCY_VARIANTS=(${LUCY_BENCH_LUCY_VARIANTS:-lucy-legacy lucy-cuda})

PAIRS=${LUCY_BENCH_PAIRS:-2500}   # 2500 pairs = 5000 hands

NI=${#ITERS_LIST[@]}
NS=${#SEEDS[@]}
NV=${#LUCY_VARIANTS[@]}

TASK=${SLURM_ARRAY_TASK_ID:-0}
SEED_IDX=$((TASK % NS))
ITER_IDX=$(((TASK / NS) % NI))
VAR_IDX=$(((TASK / (NS * NI)) % NV))

VARIANT=${LUCY_VARIANTS[$VAR_IDX]}
ITERS=${ITERS_LIST[$ITER_IDX]}
SEED=${SEEDS[$SEED_IDX]}

LUCY_BENCH_ROOT="${LUCY_BENCH_ROOT:-$(pwd)}"
cd "$LUCY_BENCH_ROOT"

if command -v module >/dev/null 2>&1; then
  module load conda/latest || true
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LUCY_BENCH_ROOT/envs/lucy"

case "$VARIANT" in
  lucy-legacy) BUILD_DIR="$LUCY_BENCH_ROOT/build-legacy" ;;
  lucy-cuda)   BUILD_DIR="$LUCY_BENCH_ROOT/build-cuda" ;;
  *) echo "[eval] unknown variant: $VARIANT" >&2; exit 2 ;;
esac
LUCY_BIN="$BUILD_DIR/bin/PokerBotMAIF"

LUCY_MODEL="$LUCY_BENCH_ROOT/models/$VARIANT/iters=${ITERS}/seed=${SEED}.dat"
OS_MODEL="$LUCY_BENCH_ROOT/models/openspiel/iters=${ITERS}/seed=${SEED}.pkl"
OUT="$LUCY_BENCH_ROOT/results/${VARIANT}-vs-openspiel/lucy_iters=${ITERS}/baseline_iters=${ITERS}/seed=${SEED}.json"

if [[ ! -f "$LUCY_MODEL" ]]; then
  echo "[eval] missing $LUCY_MODEL — did training finish?" >&2
  exit 3
fi
if [[ ! -f "$OS_MODEL" ]]; then
  echo "[eval] missing $OS_MODEL — did training finish?" >&2
  exit 3
fi

echo "[eval] task=$TASK variant=$VARIANT iters=$ITERS seed=$SEED pairs=$PAIRS"
echo "[eval] node=$(hostname)"

mkdir -p "$(dirname "$OUT")"

python -m bench.play_match \
  --bot-a "lucy:bin=$LUCY_BIN,model=$LUCY_MODEL,abstraction=fcpa" \
  --bot-b "openspiel:$OS_MODEL" \
  --label-a "$VARIANT@${ITERS}" \
  --label-b "openspiel@${ITERS}" \
  --pairs "$PAIRS" \
  --seed "$((SEED * 100 + ITER_IDX))" \
  --out "$OUT"

echo "[eval] done."
