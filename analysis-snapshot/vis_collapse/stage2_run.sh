#!/usr/bin/env bash
# Run ANY python script inside the stage2 env (same interpreter / PYTHONPATH / CUDA libs as run_room.sh):
#   bash stage2_run.sh precision_sweep.py --deliver-dir ... 
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PKG=${STAGE2_PKG:-/scratch/zr2245/shared-collapse-stage2-v2}   # v12: stage2-v2 (constant vertex_weight + ctx index coder); STAGE2_PKG=... overrides
[[ -d "$PKG" ]] || { echo "stage2 package not found: $PKG (apply stage2-v2-patch-v1/apply_stage2_v2.sh or set STAGE2_PKG)"; exit 1; }
echo "stage2 package: $PKG ($(cat "$PKG/STAGE2_VERSION" 2>/dev/null || echo v1, no STAGE2_VERSION file))"
ENV_PATH=${STAGE2_ENV:-/scratch/zr2245/conda-envs/pmsh-cu126}
LAB=${STAGE2_LAB:-/scratch/zr2245/meshsplatting-original-lossless-v1}
PYTHON=${STAGE2_PYTHON:-$LAB/.venv/bin/python}
[[ -x "$PYTHON" ]] || PYTHON="$ENV_PATH/bin/python"
export CUDA_HOME="$ENV_PATH"; export PATH="$ENV_PATH/bin:$PATH"
export PYTHONPATH="$PKG/renderer:$PKG:$PKG/codec:$LAB/upstream/submodules/effrdel/src:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ENV_PATH/targets/x86_64-linux/lib:$ENV_PATH/lib:$ENV_PATH/lib64:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8
SCRIPT=$1; shift
[[ "$SCRIPT" = /* ]] || SCRIPT="$HERE/$SCRIPT"
cd "$PKG"; "$PYTHON" -u "$SCRIPT" --stage2 "$PKG" "$@"
