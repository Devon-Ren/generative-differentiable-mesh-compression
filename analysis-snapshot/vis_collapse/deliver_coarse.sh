#!/usr/bin/env bash
# Same interpreter / PYTHONPATH / CUDA library env as shared-collapse-stage2-v1/run_room.sh, then deliver_coarse.py "$@".
# Usage: bash deliver_coarse.sh --coarse <best.pt> --output <dir> [--png] [--shared-steps 400]
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PKG=${STAGE2_PKG:-/scratch/zr2245/shared-collapse-stage2-v2}   # v12: stage2-v2 (constant vertex_weight + ctx index coder); STAGE2_PKG=... overrides
[[ -d "$PKG" ]] || { echo "stage2 package not found: $PKG (apply stage2-v2-patch-v1/apply_stage2_v2.sh or set STAGE2_PKG)"; exit 1; }
echo "stage2 package: $PKG ($(cat "$PKG/STAGE2_VERSION" 2>/dev/null || echo v1, no STAGE2_VERSION file))"
ENV_PATH=${STAGE2_ENV:-/scratch/zr2245/conda-envs/pmsh-cu126}
LAB=${STAGE2_LAB:-/scratch/zr2245/meshsplatting-original-lossless-v1}
PYTHON=${STAGE2_PYTHON:-$LAB/.venv/bin/python}
[[ -x "$PYTHON" ]] || PYTHON="$ENV_PATH/bin/python"
export CUDA_HOME="$ENV_PATH"
export PATH="$ENV_PATH/bin:$PATH"
export PYTHONPATH="$PKG/renderer:$PKG:$PKG/codec:$LAB/upstream/submodules/effrdel/src:${PYTHONPATH:-}"
CUDA_LIB_DIRS="$ENV_PATH/targets/x86_64-linux/lib:$ENV_PATH/lib:$ENV_PATH/lib64"
export LD_LIBRARY_PATH="$CUDA_LIB_DIRS:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=8
cd "$PKG"
"$PYTHON" -c 'from stage2_triangle_rasterization import _C; print("stage2 renderer OK, revision", _C.shared_stage2_backward_revision())'
"$PYTHON" -u "$HERE/deliver_coarse.py" --stage2 "$PKG" "$@"
