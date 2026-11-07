#!/usr/bin/env bash
# axisA-levels-v1.sh (2026-10-16): the K x R dictionary sweep (3 x 4 = 12 points) on EVERY axis-B vertex level, all in the
# final delivery format (stage2-v2 const vertex_weight + ctx indices, XYZ q14 / DC q7).  Same pipeline as run_axisA.sh
# (k-means dictionaries 4 rounds + 400-step shared fit + export + decoded val/test) with --dc q7 made explicit.
# Restartable: a point with summary.json is skipped.  Outputs $W/axisA-q14q7-<level>/k<K>_r<R>/, table $W/axisA-levels.txt.
# Usage (GPU node, env activated):  bash /scratch/zr2245/analysis/axisA-levels-v1.sh
set -o pipefail
W=/scratch/zr2245/room-vis-collapse
PKG=/scratch/zr2245/analysis/vis_collapse-v12
XYZ=${XYZ:-q14}; DC=${DC:-q7}
LEVELS=${LEVELS:-"1.0 0.9 0.7 0.6 0.5 0.8"}      # v0.8 last: its final-format numbers also come from hull-v2.txt
KS=${KS:-"1024 4096 16384"}
RS=${RS:-"1 256 1024 4096"}
WAIT_FOR=${WAIT_FOR:-$W/hull-v2.txt}             # do not share the GPU with the running hull-v2 job
[ -f $PKG/deliver_coarse.sh ] || { echo "MISSING $PKG/deliver_coarse.sh"; exit 2; }
for L in $LEVELS; do [ -f $W/axisB/v$L/finetune/best.pt ] || { echo "MISSING mesh for level $L: $W/axisB/v$L/finetune/best.pt"; exit 2; }; done
while [ -n "$WAIT_FOR" ] && [ ! -f "$WAIT_FOR" ]; do echo "$(date +%H:%M) waiting for $WAIT_FOR (hull-v2 still running)"; sleep 120; done
TABLE=$W/axisA-levels.txt; touch $TABLE
for L in $LEVELS; do
  MESH=$W/axisB/v$L/finetune/best.pt; OUT=$W/axisA-q14q7-v$L; mkdir -p $OUT
  for K in $KS; do for R in $RS; do
    name=$OUT/k${K}_r${R}; tag="v$L K$K R$R"; t0=$(date +%s)
    if [ ! -f $name/summary.json ]; then rm -rf $name
      bash $PKG/deliver_coarse.sh --coarse $MESH --output $name --k $K --r $R --xyz $XYZ --dc $DC --dictionary-rounds 4 > $name.log 2>&1 || { echo "FAILED $tag"; tail -3 $name.log; continue; }
    fi
    grep -q "^level $L | K $K R $R " $TABLE && { echo "done earlier: $tag"; continue; }
    python - "$name" "$L" "$K" "$R" "$XYZ" "$DC" "$t0" >> $TABLE <<'PY'
import json, sys, time
name, L, K, R, xyz, dc, t0 = sys.argv[1:8]
d = json.load(open(name + '/summary.json')); t = d['delivery_test']; s = {r['stage']: r['validation'] for r in d['stages']}
pb = d['codec']['payload_bytes']
print(f"level {L} | K {K} R {R} | vertices {d.get('vertices', d.get('n_vertices', '?'))} | {xyz}/{dc} {d.get('stage2_version','v1')} {d.get('weight','keep')}/{d.get('index_codec','zstd')} | "
      f"{d['file_MB']:.3f} MB test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f} | val fitted {s['shared_fitted']['psnr']:.3f} decoded {s['decoded_delivery']['psnr']:.3f} | "
      f"mapping {sum(pb.get(k,0) for k in ('app_mapping','app_residual_mapping','ctx_mapping','ctx_residual'))/1e6:.3f} banks {(pb['app_bank']+pb['app_residual_bank'])/1e6:.3f} | {int(time.time())-int(t0)} s")
PY
    tail -1 $TABLE
  done; done
done
echo "===== $TABLE ====="; cat $TABLE
