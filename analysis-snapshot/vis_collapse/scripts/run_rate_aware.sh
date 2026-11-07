#!/usr/bin/env bash
# Rate-aware step 1: precision sweep WITHOUT training on the existing geo-only delivery (how many bits does XYZ/DC need?).
# Step 2: QAT deliveries (geometry pre-quantised + STE DC) at the candidate precisions, same mesh, same 400-step fit,
# to be compared with the no-training rows of step 1 and with the q18/q8 baseline (26.695 MB / test 27.624).
set -o pipefail
W=/scratch/zr2245/room-vis-collapse
PKG=/scratch/zr2245/analysis/vis_collapse-v12
BASE=${BASE:-$W/deliver-geo-only}                       # has shared-final.pt + config.json
MESH=${MESH:-$W/finetune-geo-only/best.pt}
MODES=${MODES:-"q16/q8,q15/q8,q14/q8,q13/q8,q14/q7,q15/q7"}
QAT=${QAT:-""}   # v8: DC QAT recorded as a negative result; set QAT="q14/q7" to retry with the fixed grid
echo "=== step 1: precision sweep (no training) on $BASE ==="
bash $PKG/stage2_run.sh precision_sweep.py --deliver-dir $BASE --modes $MODES 2>&1 | tee -a $W/precision_sweep.log
cat $BASE/precision/sweep.txt
echo "=== step 2: QAT deliveries ==="
for m in $QAT; do xyz=${m%/*}; dc=${m#*/}; name=$W/qat_${xyz}_${dc}; tag=qat-$xyz-$dc; echo "--- $tag ---"
  if [ ! -f $name/summary.json ]; then rm -rf $name
    bash $PKG/deliver_coarse.sh --coarse $MESH --output $name --xyz $xyz --dc $dc --qat > $name.log 2>&1 || { echo "FAILED $tag"; tail -3 $name.log; continue; }
  fi
  python - <<PY | tee -a $W/rate_aware.txt
import json; d=json.load(open('$name/summary.json')); t=d['delivery_test']; s={r['stage']:r['validation'] for r in d['stages']}
print('QAT $xyz/$dc | val shared_initial', round(s['shared_initial']['psnr'],3), 'fitted', round(s['shared_fitted']['psnr'],3), 'decoded', round(s['decoded_delivery']['psnr'],3),
      '| delivered', d['file_MB'], 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| flips', d['codec']['quantization_geometry']['flipped_faces'])
PY
done
echo "===== rate_aware.txt ====="; cat $W/rate_aware.txt 2>/dev/null
