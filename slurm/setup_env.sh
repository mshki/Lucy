#!/usr/bin/env bash
# One-shot setup on Unity. Run once (interactively or via srun) before any
# job arrays:
#
#     ssh unity
#     cd /work/pi_machta_umass_edu/$USER/lucy-bench
#     bash slurm/setup_env.sh
#
# What it does:
#   1. Loads Unity modules (conda).
#   2. Creates a conda env at /work/pi_machta_umass_edu/$USER/lucy-bench/envs/lucy
#      (out-of-tree to avoid burning /home quota).
#   3. Installs open_spiel + numpy + matplotlib into the env.
#   4. Builds both Lucy variants (cuda + legacy) into separate build dirs.
set -euo pipefail
set -o errtrace

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LUCY_BENCH_ROOT="${LUCY_BENCH_ROOT:-$REPO_ROOT}"
ENV_PREFIX="${ENV_PREFIX:-$LUCY_BENCH_ROOT/envs/lucy}"
PY_VER="${PY_VER:-3.11}"

echo "[setup] REPO_ROOT     = $REPO_ROOT"
echo "[setup] LUCY_BENCH    = $LUCY_BENCH_ROOT"
echo "[setup] ENV_PREFIX    = $ENV_PREFIX"
echo "[setup] PY_VER        = $PY_VER"

# 1. Unity modules ----------------------------------------------------------
if command -v module >/dev/null 2>&1; then
  module load conda/latest || true
fi

# 2. Conda env --------------------------------------------------------------
if [[ ! -d "$ENV_PREFIX" ]]; then
  echo "[setup] creating conda env at $ENV_PREFIX"
  mkdir -p "$(dirname "$ENV_PREFIX")"
  conda create --yes --prefix "$ENV_PREFIX" "python=$PY_VER" pip
else
  echo "[setup] conda env already exists, skipping create"
fi

# Activate via path (works both inside/outside conda init).
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# 3. Python deps ------------------------------------------------------------
echo "[setup] installing python deps"
pip install --upgrade pip
pip install "open_spiel>=1.6.10" numpy matplotlib

# 4. Build both Lucy variants -----------------------------------------------
build_variant() {
  local src="$1" build="$2" tag="$3"
  echo "[setup] building $tag from $src -> $build"
  cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$build" -j"$(nproc)"
  echo "[setup] $tag binary: $build/bin/PokerBotMAIF"
}

CUDA_SRC="${CUDA_SRC:-$LUCY_BENCH_ROOT/lucy-cuda}"
LEGACY_SRC="${LEGACY_SRC:-$LUCY_BENCH_ROOT/lucy-legacy}"

CUDA_BUILD="${CUDA_BUILD:-$LUCY_BENCH_ROOT/build-cuda}"
LEGACY_BUILD="${LEGACY_BUILD:-$LUCY_BENCH_ROOT/build-legacy}"

if [[ -d "$CUDA_SRC" ]]; then
  build_variant "$CUDA_SRC" "$CUDA_BUILD" "lucy-cuda"
else
  echo "[setup] WARNING: $CUDA_SRC not found, skipping cuda variant"
fi

if [[ -d "$LEGACY_SRC" ]]; then
  build_variant "$LEGACY_SRC" "$LEGACY_BUILD" "lucy-legacy"
else
  echo "[setup] WARNING: $LEGACY_SRC not found, skipping legacy variant"
fi

# 5. Workspace dirs ---------------------------------------------------------
mkdir -p "$LUCY_BENCH_ROOT/models" "$LUCY_BENCH_ROOT/results" "$LUCY_BENCH_ROOT/logs"

echo
echo "[setup] done. To activate the env in any future shell:"
echo "  module load conda/latest && conda activate $ENV_PREFIX"
echo
echo "[setup] quick smoke check:"
python -c "import pyspiel, numpy; print('pyspiel', pyspiel.__name__, 'numpy', numpy.__version__)" || true
