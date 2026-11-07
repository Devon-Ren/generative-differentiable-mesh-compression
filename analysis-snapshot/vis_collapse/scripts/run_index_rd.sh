#!/usr/bin/env bash
# Rate-aware step 2: neighbour-context RD reassignment of the SH indices on the geo-only delivery (no training, then +200-step refit).
set -o pipefail
W=/scratch/zr2245/room-vis-collapse
PKG=/scratch/zr2245/analysis/vis_collapse-v12
BASE=${BASE:-$W/deliver-geo-only}
bash $PKG/stage2_run.sh index_rd.py --deliver-dir $BASE --lambdas ${LAMBDAS:-0,0.1,0.3,1,3} --refit-steps ${REFIT:-200} 2>&1 | tee $W/index_rd.log
cat $BASE/index_rd/sweep.txt
