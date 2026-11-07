#!/usr/bin/env bash
# v12: byte / quality effect of the two stage2-v2 format changes (constant vertex_weight, ctx index coder), no training,
# on the fitted v0.8 shared mesh of axis B at q14/q7.  Seconds per variant plus the test render.  Table in $BASE/codec-variants/sweep.txt.
set -o pipefail
W=/scratch/zr2245/room-vis-collapse
PKG=/scratch/zr2245/analysis/vis_collapse-v12
BASE=${BASE:-$W/axisB/v0.8/deliver}              # has shared-final.pt + config.json (deliver_coarse.py output)
XYZ=${XYZ:-q14}; DC=${DC:-q7}
[ -f $BASE/shared-final.pt ] || { echo "shared-final.pt not found in $BASE"; exit 1; }
bash $PKG/stage2_run.sh codec_variants.py --deliver-dir $BASE --xyz $XYZ --dc $DC 2>&1 | tee -a $W/codec_variants.log
echo "===== $BASE/codec-variants/sweep.txt ====="; cat $BASE/codec-variants/sweep.txt
