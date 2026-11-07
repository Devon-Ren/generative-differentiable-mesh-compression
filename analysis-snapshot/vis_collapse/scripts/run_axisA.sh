#!/usr/bin/env bash
# Axis A (appearance redundancy): dictionary-size sweep K x R on ONE mesh through the same stage2 pipeline as axis B
# (k-means dictionaries built on the mesh, 400-step shared fit, q14/q8 export; v12: stage2-v2 format, i.e. constant vertex_weight + ctx-coded indices).  Default mesh = the finetuned 80%-vertex
# level of axis B (candidate default operating point); override with MESH=... .  Restartable; table in $W/axisA.txt.
set -o pipefail
W=/scratch/zr2245/room-vis-collapse
PKG=/scratch/zr2245/analysis/vis_collapse-v12
MESH=${MESH:-$W/axisB/v0.8/finetune/best.pt}
TAG=${TAG:-$(basename $(dirname $(dirname $MESH)))}   # axis B level name, e.g. v0.8 -> outputs $W/axisA-$TAG/, table $W/axisA-$TAG.txt
OUT=$W/axisA-$TAG
KS=${KS:-"1024 4096 16384"}
RS=${RS:-"1 256 1024 4096"}
[ -f $MESH ] || { echo "mesh not found: $MESH (run axis B first, or set MESH=)"; exit 1; }
echo "mesh $MESH -> $OUT"; mkdir -p $OUT
for K in $KS; do for R in $RS; do
  name=$OUT/k${K}_r${R}; tag=axisA-k${K}-r${R}; echo "=== $tag ==="
  if [ ! -f $name/summary.json ]; then rm -rf $name
    bash $PKG/deliver_coarse.sh --coarse $MESH --output $name --k $K --r $R --dictionary-rounds 4 > $name.log 2>&1 || { echo "FAILED $tag"; tail -3 $name.log; continue; }
  fi
  python - <<PY | tee -a $OUT.txt
import json; d=json.load(open('$name/summary.json')); t=d['delivery_test']; s={r['stage']:r['validation'] for r in d['stages']}; pb=d['codec']['payload_bytes']
print('K $K R $R | val shared_initial', round(s['shared_initial']['psnr'],3), 'fitted', round(s['shared_fitted']['psnr'],3), 'decoded', round(s['decoded_delivery']['psnr'],3),
      '| delivered', d['file_MB'], 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4),
      '| mapping MB', round(sum(pb.get(k,0) for k in ('app_mapping','app_residual_mapping','ctx_mapping','ctx_residual'))/1e6,3), 'banks MB', round((pb['app_bank']+pb['app_residual_bank'])/1e6,3), '|', d.get('stage2_version','v1'), d.get('weight','keep'), d.get('index_codec','zstd'))
PY
done; done
echo "===== $OUT.txt ====="; cat $OUT.txt
