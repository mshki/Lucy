#!/usr/bin/env bash
#SBATCH --job-name=lucy-train
#SBATCH --partition=cpu
#SBATCH --account=pi_machta_umass_edu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
# Job array of training runs across (variant × iter_budget × seed). The total
# array size is len(VARIANTS) * len(ITERS_LIST) * len(SEEDS). The submitter
# (slurm/submit_all.sh) overrides --array=... to match.
#
# Layout:
#   index = ((variant_idx * len(ITERS) + iter_idx) * len(SEEDS) + seed_idx)
#
# Variants:
#   0 = lucy-legacy
#   1 = lucy-cuda
#   2 = openspiel
set -euo pipefail

# ---------- Parameter grid (kept in lockstep with submit_all.sh) ----------
ITERS_LIST=(${LUCY_BENCH_ITERS:-1000 10000 100000 1000000})
SEEDS=(${LUCY_BENCH_SEEDS:-1 2 3})
VARIANTS=(${LUCY_BENCH_VARIANTS:-lucy-legacy lucy-cuda openspiel})

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

# ---------- Environment ----------
LUCY_BENCH_ROOT="${LUCY_BENCH_ROOT:-$(pwd)}"
cd "$LUCY_BENCH_ROOT"

# Activate the venv built by slurm/setup_env.sh
# shellcheck disable=SC1091
source "$LUCY_BENCH_ROOT/envs/lucy/bin/activate"

# ---------- Dispatch ----------
echo "[train] task=$TASK variant=$VARIANT iters=$ITERS seed=$SEED"
echo "[train] node=$(hostname) cpus=$SLURM_CPUS_PER_TASK"

case "$VARIANT" in
  lucy-legacy)
    BIN="$LUCY_BENCH_ROOT/lucy-legacy/build/bin/PokerBotMAIF"
    OUT="$LUCY_BENCH_ROOT/models/lucy-legacy/iters=${ITERS}/seed=${SEED}.dat"
    mkdir -p "$(dirname "$OUT")"
    echo "[train] running $BIN --train $ITERS --seed $SEED ..."
    "$BIN" \
      --train "$ITERS" \
      --players 2 \
      --abstraction fcpa \
      --seed "$SEED" \
      --out "$OUT"
    ;;
  lucy-cuda)
    BIN="$LUCY_BENCH_ROOT/lucy-cuda/build/bin/PokerBotMAIF"
    OUT="$LUCY_BENCH_ROOT/models/lucy-cuda/iters=${ITERS}/seed=${SEED}.dat"
    mkdir -p "$(dirname "$OUT")"
    echo "[train] running $BIN --train $ITERS --seed $SEED ..."
    "$BIN" \
      --train "$ITERS" \
      --players 2 \
      --abstraction fcpa \
      --seed "$SEED" \
      --out "$OUT"
    ;;
  openspiel)
    OUT="$LUCY_BENCH_ROOT/models/openspiel/iters=${ITERS}/seed=${SEED}.pkl"
    mkdir -p "$(dirname "$OUT")"
    echo "[train] running OpenSpiel external-sampling MCCFR ..."
    python -m bench.train_openspiel \
      --iters "$ITERS" \
      --seed "$SEED" \
      --out "$OUT"
    ;;
  *)
    echo "[train] unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac

echo "[train] done."
