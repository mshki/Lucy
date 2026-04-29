#!/usr/bin/env bash
#SBATCH --job-name=lucy-setup
#SBATCH --partition=cpu
#SBATCH --account=pi_machta_umass_edu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err
# One-shot env-creation + dependency-install + Lucy build job.
# Submit from /work/pi_machta_umass_edu/$USER/lucy-bench:
#
#     sbatch slurm/setup.sbatch
#
# Re-runnable: setup_env.sh is idempotent.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
exec bash slurm/setup_env.sh
