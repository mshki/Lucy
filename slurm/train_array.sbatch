#!/usr/bin/env bash
#SBATCH --job-name=lucy-train
#SBATCH --partition=cpu
#SBATCH --account=pi_machta_umass_edu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
# Job array of training runs across (variant × iter_budget × seed). Index
# layout: idx = ((var * len(ITERS) + iter) * len(SEEDS) + seed).
#
# Variants on bench/v2 / v0.2 sweep:
#   lucy-legacy     pre-NodeMatrix Lucy (worktree at a8deeee)
#   lucy-cuda       NodeMatrix Lucy on main (no OMP/DCFR/OS)
#   lucy-omp        bench/v2 binary, vanilla CFR, external sampling
#   lucy-dcfr       bench/v2, DCFR, external sampling
#   lucy-os         bench/v2, vanilla CFR, outcome sampling
#   lucy-dcfr-os    bench/v2, DCFR, outcome sampling
#   openspiel       OpenSpiel OutcomeSamplingMCCFRSolver
set -euo pipefail

ITERS_LIST=(${LUCY_BENCH_ITERS:-1000 10000 100000 1000000})
SEEDS=(${LUCY_BENCH_SEEDS:-1 2 3})
VARIANTS=(${LUCY_BENCH_VARIANTS:-lucy-omp lucy-dcfr lucy-os lucy-dcfr-os openspiel})

NI=${#ITERS_LIST[@]}
NS=${#SEEDS[@]}
NV=${#VARIANTS[@]}

TASK=${SLURM_ARRAY_TASK_ID:-0}
SEED_IDX=$((TASK % NS))
ITER_IDX=$(((TASK / NS) % NI))
VAR_IDX=$(((TASK / (NS * NI)) % NV))

VARIANT=${VARIANTS[$VAR_IDX]}
ITERS=${ITERS_LIST[$ITER_IDX]}
SEED=${SEEDS[$SEED_IDX]}

LUCY_BENCH_ROOT="${LUCY_BENCH_ROOT:-$(pwd)}"
cd "$LUCY_BENCH_ROOT"

# shellcheck disable=SC1091
source "$LUCY_BENCH_ROOT/envs/lucy/bin/activate"

echo "[train] task=$TASK variant=$VARIANT iters=$ITERS seed=$SEED"
echo "[train] node=$(hostname) cpus=$SLURM_CPUS_PER_TASK"

# Resolve (binary, cfr_variant, sampler) for each Lucy variant.
LUCY_BIN=""
CFR_VARIANT="vanilla"
SAMPLER="external"
case "$VARIANT" in
  lucy-legacy)   LUCY_BIN="$LUCY_BENCH_ROOT/lucy-legacy/build/bin/PokerBotMAIF" ;;
  lucy-cuda)     LUCY_BIN="$LUCY_BENCH_ROOT/lucy-cuda/build/bin/PokerBotMAIF" ;;
  lucy-omp)      LUCY_BIN="$LUCY_BENCH_ROOT/lucy-v2/build/bin/PokerBotMAIF" ;;
  lucy-dcfr)     LUCY_BIN="$LUCY_BENCH_ROOT/lucy-v2/build/bin/PokerBotMAIF"
                 CFR_VARIANT="dcfr" ;;
  lucy-os)       LUCY_BIN="$LUCY_BENCH_ROOT/lucy-v2/build/bin/PokerBotMAIF"
                 SAMPLER="outcome" ;;
  lucy-dcfr-os)  LUCY_BIN="$LUCY_BENCH_ROOT/lucy-v2/build/bin/PokerBotMAIF"
                 CFR_VARIANT="dcfr"
                 SAMPLER="outcome" ;;
  openspiel) ;;
  *) echo "[train] unknown variant: $VARIANT" >&2; exit 2 ;;
esac

if [[ "$VARIANT" == "openspiel" ]]; then
  OUT="$LUCY_BENCH_ROOT/models/openspiel/iters=${ITERS}/seed=${SEED}.pkl"
  mkdir -p "$(dirname "$OUT")"
  echo "[train] running OpenSpiel OutcomeSamplingMCCFR..."
  python -m bench.train_openspiel --iters "$ITERS" --seed "$SEED" --out "$OUT"
else
  if [[ ! -x "$LUCY_BIN" ]]; then
    echo "[train] missing $LUCY_BIN — was lucy-v2 built? Run setup_env.sh" >&2
    exit 3
  fi
  OUT="$LUCY_BENCH_ROOT/models/${VARIANT}/iters=${ITERS}/seed=${SEED}.dat"
  mkdir -p "$(dirname "$OUT")"
  echo "[train] $LUCY_BIN --train $ITERS --cfr-variant $CFR_VARIANT --sampler $SAMPLER ..."
  "$LUCY_BIN" \
    --train "$ITERS" \
    --players 2 \
    --abstraction fcpa \
    --seed "$SEED" \
    --cfr-variant "$CFR_VARIANT" \
    --sampler "$SAMPLER" \
    --out "$OUT"
fi

echo "[train] done."
