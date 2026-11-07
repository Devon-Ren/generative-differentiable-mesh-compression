#!/usr/bin/env bash
# Real bytes of the neighbour-context index coder on the index_rd lambda sweep (no training, seconds).
set -o pipefail
W=/scratch/zr2245/room-vis-collapse
PKG=/scratch/zr2245/analysis/vis_collapse-v12
BASE=${BASE:-$W/deliver-geo-only}
ls $PKG/native/_index_ctx*.so >/dev/null 2>&1 || bash $PKG/stage2_run.sh build_index_ctx.py
bash $PKG/stage2_run.sh index_context_codec.py --deliver-dir $BASE --lambdas ${LAMBDAS:-0,0.1,0.3,1,3} 2>&1 | tee $W/index_ctx.log
cat $BASE/index_ctx/sweep.txt
