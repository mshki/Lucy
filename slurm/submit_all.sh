#!/usr/bin/env bash
# End-to-end driver: submits the training array and a dependent eval array,
# then prints the chained job IDs and a hint on how to follow the logs.
#
# Usage (from /work/pi_machta_umass_edu/$USER/lucy-bench, after setup_env.sh):
#
#   bash slurm/submit_all.sh                  # default sweep
#   ITERS="1000 10000" SEEDS="1 2" \
#     bash slurm/submit_all.sh                # smaller smoke sweep
#
# Knobs (env vars, all optional):
#   ITERS         space-separated list of iteration budgets
#   SEEDS         space-separated list of seeds
#   VARIANTS      training variants  (default: lucy-legacy lucy-cuda openspiel)
#   LUCY_VARIANTS eval variants     (default: lucy-legacy lucy-cuda)
#   PAIRS         number of duplicate pairs per match (default: 2500)
#   PARTITION     SLURM partition  (default: cpu)
#   ACCOUNT       SLURM account    (default: pi_machta_umass_edu)
#   TIME_TRAIN    walltime per training task (default: 12:00:00)
#   TIME_EVAL     walltime per eval task     (default: 04:00:00)
set -euo pipefail

LUCY_BENCH_ROOT="${LUCY_BENCH_ROOT:-$(pwd)}"
cd "$LUCY_BENCH_ROOT"
mkdir -p logs

ITERS=${ITERS:-"1000 10000 100000 1000000"}
SEEDS=${SEEDS:-"1 2 3"}
VARIANTS=${VARIANTS:-"lucy-omp lucy-dcfr lucy-os lucy-dcfr-os openspiel"}
LUCY_VARIANTS=${LUCY_VARIANTS:-"lucy-omp lucy-dcfr lucy-os lucy-dcfr-os"}
PAIRS=${PAIRS:-2500}
PARTITION=${PARTITION:-cpu}
ACCOUNT=${ACCOUNT:-pi_machta_umass_edu}
TIME_TRAIN=${TIME_TRAIN:-12:00:00}
TIME_EVAL=${TIME_EVAL:-04:00:00}

read -ra ITERS_ARR     <<<"$ITERS"
read -ra SEEDS_ARR     <<<"$SEEDS"
read -ra VAR_ARR       <<<"$VARIANTS"
read -ra LVAR_ARR      <<<"$LUCY_VARIANTS"

NI=${#ITERS_ARR[@]}
NS=${#SEEDS_ARR[@]}
NVT=${#VAR_ARR[@]}
NLE=${#LVAR_ARR[@]}

NTRAIN=$((NVT * NI * NS))
NEVAL=$((NLE  * NI * NS))

echo "[submit] LUCY_BENCH_ROOT = $LUCY_BENCH_ROOT"
echo "[submit] ITERS  = $ITERS  (${NI})"
echo "[submit] SEEDS  = $SEEDS  (${NS})"
echo "[submit] TRAIN  variants = $VARIANTS  (${NVT})"
echo "[submit] EVAL   variants = $LUCY_VARIANTS (${NLE})"
echo "[submit] PAIRS  = $PAIRS"
echo "[submit] => $NTRAIN train tasks, $NEVAL eval tasks"

# Pass the same parameter grid to the array scripts via env vars.
EXPORT_VARS="ALL,LUCY_BENCH_ROOT=$LUCY_BENCH_ROOT,LUCY_BENCH_ITERS=$ITERS,LUCY_BENCH_SEEDS=$SEEDS,LUCY_BENCH_VARIANTS=$VARIANTS,LUCY_BENCH_LUCY_VARIANTS=$LUCY_VARIANTS,LUCY_BENCH_PAIRS=$PAIRS"

TRAIN_JOBID=$(sbatch --parsable \
  --partition="$PARTITION" \
  --account="$ACCOUNT" \
  --time="$TIME_TRAIN" \
  --array=0-$((NTRAIN - 1))%32 \
  --export="$EXPORT_VARS" \
  slurm/train_array.sbatch)

echo "[submit] training array submitted: $TRAIN_JOBID  (size $NTRAIN)"

EVAL_JOBID=$(sbatch --parsable \
  --partition="$PARTITION" \
  --account="$ACCOUNT" \
  --time="$TIME_EVAL" \
  --dependency=afterok:"$TRAIN_JOBID" \
  --array=0-$((NEVAL - 1))%32 \
  --export="$EXPORT_VARS" \
  slurm/eval_array.sbatch)

echo "[submit] eval array submitted:     $EVAL_JOBID  (size $NEVAL)"
echo
echo "[submit] follow with:"
echo "  squeue --me"
echo "  tail -f logs/train_${TRAIN_JOBID}_*.out"
echo "  tail -f logs/eval_${EVAL_JOBID}_*.out"
echo
echo "[submit] when eval finishes, aggregate + plot:"
echo "  python -m bench.aggregate --results-dir results --out results/summary.csv"
echo "  python -m bench.plot --csv results/summary.csv --out results/plot.png"
