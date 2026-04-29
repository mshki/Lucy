#!/usr/bin/env bash
# Deploy the bench harness + both Lucy worktrees to Unity. Run from the
# *local* machine (where the worktrees live):
#
#     bash slurm/deploy.sh
#
# Pushes:
#   - lucy-cuda/   (current bench/v1 branch)
#   - lucy-legacy/ (bench/v1-legacy worktree)
#   - bench/       (Python harness)
#   - slurm/       (this directory)
#
# Default destination: /work/pi_machta_umass_edu/$USER/lucy-bench
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LOCAL_LEGACY="${LOCAL_LEGACY:-$LOCAL_ROOT/../Lucy-worktrees/legacy}"
SSH_TARGET="${SSH_TARGET:-unity}"
REMOTE_USER="${REMOTE_USER:-$(ssh "$SSH_TARGET" 'echo $USER')}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/pi_machta_umass_edu/$REMOTE_USER/lucy-bench}"

echo "[deploy] LOCAL  : $LOCAL_ROOT"
echo "[deploy] LEGACY : $LOCAL_LEGACY"
echo "[deploy] REMOTE : $SSH_TARGET:$REMOTE_ROOT"

ssh "$SSH_TARGET" "mkdir -p '$REMOTE_ROOT'"

# Use rsync with --exclude to keep build dirs / models / results out of the
# initial sync. They will be regenerated on Unity.
COMMON_EXCLUDES=(
  --exclude='build/'
  --exclude='build-cuda/'
  --exclude='build-legacy/'
  --exclude='envs/'
  --exclude='.git/'
  --exclude='__pycache__/'
  --exclude='*.dat'
  --exclude='models/'
  --exclude='results/'
  --exclude='logs/'
)

# 1. cuda variant (the main repo) -> $REMOTE_ROOT/lucy-cuda
rsync -avh "${COMMON_EXCLUDES[@]}" \
  "$LOCAL_ROOT/" "$SSH_TARGET:$REMOTE_ROOT/lucy-cuda/" | tail -10

# 2. legacy worktree -> $REMOTE_ROOT/lucy-legacy
rsync -avh "${COMMON_EXCLUDES[@]}" \
  "$LOCAL_LEGACY/" "$SSH_TARGET:$REMOTE_ROOT/lucy-legacy/" | tail -10

# 3. bench/ + slurm/ at the root for direct convenience.
rsync -avh "$LOCAL_ROOT/bench/"  "$SSH_TARGET:$REMOTE_ROOT/bench/"  | tail -5
rsync -avh "$LOCAL_ROOT/slurm/"  "$SSH_TARGET:$REMOTE_ROOT/slurm/"  | tail -5

echo "[deploy] done."
echo
echo "Next steps on Unity:"
echo "  ssh $SSH_TARGET"
echo "  cd $REMOTE_ROOT"
echo "  bash slurm/setup_env.sh"
echo "  ITERS='1000 10000' SEEDS='1' bash slurm/submit_all.sh   # smoke run"
