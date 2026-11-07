#!/usr/bin/env bash
# v12: the candidate default operating point on Room, measured end to end in the stage2-v2 format:
#   axis-B 80 % vertex mesh (finetuned) -> k-means K4096 x R256 dictionaries -> 400-step shared fit -> q14 XYZ / q7 DC
#   -> constant vertex_weight + ctx-coded indices -> MB + decoded test metrics.   Restartable (skips when summary exists).
set -o pipefail
W=/scratch/zr2245/room-vis-collapse
PKG=/scratch/zr2245/analysis/vis_collapse-v12
MESH=${MESH:-$W/axisB/v0.8/finetune/best.pt}
K=${K:-4096}; R=${R:-256}; XYZ=${XYZ:-q14}; DC=${DC:-q7}
TAG=${TAG:-$(basename $(dirname $(dirname $MESH)))}
OUT=${OUT:-$W/default-$TAG-k${K}-r${R}-${XYZ}${DC}}
[ -f $MESH ] || { echo "mesh not found: $MESH"; exit 1; }
echo "=== default point: $MESH  K$K R$R $XYZ/$DC -> $OUT ==="
if [ ! -f $OUT/summary.json ]; then rm -rf $OUT
  bash $PKG/deliver_coarse.sh --coarse $MESH --output $OUT --k $K --r $R --xyz $XYZ --dc $DC --dictionary-rounds 4 --png 2>&1 | tee $OUT.log
  [ -f $OUT/summary.json ] || { echo "FAILED (see $OUT.log)"; exit 1; }
fi
python - <<PY | tee -a $W/default.txt
import json; d=json.load(open('$OUT/summary.json')); t=d['delivery_test']; s={r['stage']:r['validation'] for r in d['stages']}; pb=d['codec']['payload_bytes']
print('$TAG K$K R$R $XYZ/$DC', d.get('stage2_version','v1'), d.get('weight','keep'), d.get('index_codec','zstd'),
      '| vertices', d['vertices'], 'faces', d['faces'], '| val fitted', round(s['shared_fitted']['psnr'],3), 'decoded', round(s['decoded_delivery']['psnr'],3),
      '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4))
print('   payload MB:', {k: round(b/1e6,3) for k,b in sorted(pb.items(), key=lambda kv:-kv[1]) if b > 20000})
PY
echo "===== $W/default.txt ====="; cat $W/default.txt
