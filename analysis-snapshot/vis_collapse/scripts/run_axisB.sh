#!/usr/bin/env bash
# Axis B (geometry redundancy): vertex-count sweep with the geo-only render-sensitivity collapse (vis_collapse-v5),
# each level: collapse -> 3000-step appearance finetune -> stage2 sharing + fit + codec export -> MB + test metrics.
# Restartable: every stage is skipped when its summary already exists. Output table: $W/axisB.txt
set -o pipefail
W=/scratch/zr2245/room-vis-collapse
C=/scratch/zr2245/room-fixed-sharing-0N4OSU/run/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
OLD=/scratch/zr2245/room-shared-collapse-QGQgyr/run/selected-collapse-map.npz
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
MODEL=/scratch/zr2245/room-fixed-sharing-0N4OSU/run/original
DATA=/scratch/zr2245/datasets/m360/room
PKG=/scratch/zr2245/analysis/vis_collapse-v12
FRACS=${FRACS:-"1.0 0.9 0.8 0.7 0.6 0.5"}
CRIT=${CRIT:-"--geo-weight sens_pos --app-weight none --weight-power 0.5 --weight-clip 0.1,10 --weight-divide-pixel-scale"}

V0=$(python -c "import torch; sd=torch.load('$C',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))")
echo "original vertices: $V0"; mkdir -p $W/axisB

level () { frac=$1; nv=$(python -c "print(round($frac*$V0))"); name=$W/axisB/v$frac; tag=axisB-v$frac
  echo "=== $tag: target $nv vertices ==="
  if [ ! -f $name/report.json ]; then rm -rf $name
    (cd $PKG && python coarse_vis.py --source $C --footprint $W/footprint-sens.npz --base uniform --target-vertices $nv --old-map $OLD --output $name $CRIT \
       > $name.build.log 2>&1) || { echo "BUILD FAILED $tag"; tail -3 $name.build.log; return; }
  fi
  if [ ! -f $name/finetune/summary.json ]; then rm -rf $name/finetune
    (cd /scratch/zr2245/analysis && python coarse_finetune.py --lab $LAB --model $MODEL --dataset $DATA --images images_2 \
       --coarse $name/coarse-initial.pt --steps 3000 --out $name/finetune > $name.finetune.log 2>&1) || { echo "FINETUNE FAILED $tag"; tail -3 $name.finetune.log; return; }
  fi
  if [ ! -f $name/deliver/summary.json ]; then rm -rf $name/deliver
    bash $PKG/deliver_coarse.sh --coarse $name/finetune/best.pt --output $name/deliver > $name.deliver.log 2>&1 || { echo "DELIVER FAILED $tag"; tail -3 $name.deliver.log; return; }
  fi
  python - <<PY | tee -a $W/axisB.txt
import json
b=json.load(open('$name/report.json')); f=json.load(open('$name/finetune/summary.json')); d=json.load(open('$name/deliver/summary.json')); t=d['delivery_test']; h=f['history']
print('frac $frac | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3), 'worst', round(h[-1]['worst_view_psnr'],2),
      '| test', round(f['test_best']['psnr'],3), '| delivered', d['file_MB'], 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4))
PY
}
for f in $FRACS; do level $f; done
echo "===== axisB.txt ====="; cat $W/axisB.txt
