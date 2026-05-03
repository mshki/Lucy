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

# 1. Locate a Python with venv. Prefer Lmod's python module (pre-built on
# Unity, with a sane site layout), fall back to system /usr/bin/python3.
SYSTEM_PYTHON=""
if command -v module >/dev/null 2>&1; then
  for cand in "python/$PY_VER" "python/3.11.7" "python/3.10.14" "python/3.12.3"; do
    if module load "$cand" 2>/dev/null; then
      SYSTEM_PYTHON="$(command -v python3)"
      echo "[setup] using module $cand -> $SYSTEM_PYTHON"
      break
    fi
  done
fi
if [[ -z "$SYSTEM_PYTHON" ]]; then
  SYSTEM_PYTHON="$(command -v python3 || true)"
  echo "[setup] fallback python: $SYSTEM_PYTHON"
fi
if [[ -z "$SYSTEM_PYTHON" ]]; then
  echo "[setup] FATAL: no python3 found" >&2; exit 1
fi

# 2. venv  -----------------------------------------------------------------
# venv is much faster than conda on shared filesystems (no transaction phase).
if [[ ! -x "$ENV_PREFIX/bin/pip" ]]; then
  if [[ -d "$ENV_PREFIX" ]]; then
    echo "[setup] env exists but is incomplete — recreating"
    rm -rf "$ENV_PREFIX"
  fi
  mkdir -p "$(dirname "$ENV_PREFIX")"
  echo "[setup] creating venv at $ENV_PREFIX"
  "$SYSTEM_PYTHON" -m venv "$ENV_PREFIX"
else
  echo "[setup] venv already complete, skipping create"
fi

PY="$ENV_PREFIX/bin/python"
PIP="$ENV_PREFIX/bin/pip"
if [[ ! -x "$PY" ]]; then
  echo "[setup] FATAL: $PY missing — venv create failed" >&2; exit 1
fi

# 3. Python deps ------------------------------------------------------------
echo "[setup] installing python deps via $PIP"
"$PIP" install --upgrade pip
"$PIP" install "open_spiel>=1.6.10" numpy matplotlib

# 4. Build both Lucy variants -----------------------------------------------
build_variant() {
  local src="$1" build="$2" tag="$3"
  echo "[setup] building $tag from $src"
  # Lucy's CMakeLists hardcodes RUNTIME_OUTPUT_DIRECTORY=<src>/build/bin so
  # the actual binary path is $src/build/bin/PokerBotMAIF regardless of -B.
  cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release >/dev/null
  cmake --build "$build" -j"$(nproc)"
  echo "[setup] $tag binary: $src/build/bin/PokerBotMAIF"
}

for tag in lucy-cuda lucy-legacy lucy-v2; do
  src="$LUCY_BENCH_ROOT/$tag"
  if [[ -d "$src" ]]; then
    build_variant "$src" "$src/build" "$tag"
  else
    echo "[setup] WARNING: $src not found, skipping $tag"
  fi
done

# 5. Workspace dirs ---------------------------------------------------------
mkdir -p "$LUCY_BENCH_ROOT/models" "$LUCY_BENCH_ROOT/results" "$LUCY_BENCH_ROOT/logs"

echo
echo "[setup] done. To activate the env in any future shell:"
echo "  source $ENV_PREFIX/bin/activate"
echo
echo "[setup] quick smoke check:"
"$PY" -c "import pyspiel, numpy; print('pyspiel ok, numpy', numpy.__version__)" || true
