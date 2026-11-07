#!/usr/bin/env bash
# v12: second scene (MipNeRF360 garden, outdoor -> images_4 as in the original full_eval.py), the whole pipeline in ONE
# GPU session, same protocol as Room:
#   0. train the original MeshSplatting model (upstream/train.py, 30000 iterations, --eval split)      ~1-1.5 h
#   1. vertex_footprint.py (per-vertex pixel scale + render sensitivities)                              minutes
#   2. per level in FRACS: coarse_vis.py (geo-only criterion, --target-vertices frac*V0; CPU, >= 48 GB RAM)
#      -> coarse_finetune.py (appearance-only, 3000 steps) -> deliver_coarse.sh (k-means K4096 x R256, 400-step
#      shared fit, q14/q7, stage2-v2 format)                                                            ~30-40 min / level
#   FRACS 1.0 is the dense anchor (no merge, same finetune + delivery), 0.8 the candidate default, 0.7 the next hull point.
# Restartable: every stage is skipped when its output exists.  Table: $G/garden.txt.  Logs next to each output dir.
set -o pipefail
G=${G:-/scratch/zr2245/garden-vis-collapse}
DATA=${DATA:-/scratch/zr2245/datasets/m360/garden}
IMAGES=${IMAGES:-images_4}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
AN=/scratch/zr2245/analysis                     # coarse_finetune.py lives here (outside the package, as for Room)
PKG=/scratch/zr2245/analysis/vis_collapse-v12
FRACS=${FRACS:-"1.0 0.8 0.7"}
K=${K:-4096}; R=${R:-256}; XYZ=${XYZ:-q14}; DC=${DC:-q7}
CRIT=${CRIT:-"--geo-weight sens_pos --app-weight none --weight-power 0.5 --weight-clip 0.1,10 --weight-divide-pixel-scale"}
C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt

[ -d $DATA/$IMAGES ] || { echo "dataset images not found: $DATA/$IMAGES"; exit 1; }
[ -f $AN/coarse_finetune.py ] || { echo "coarse_finetune.py not found in $AN"; exit 1; }
ls $PKG/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "native module missing in $PKG/native (cp from vis_collapse-v11/native or: cd $PKG && python build_vis.py)"; exit 1; }
mkdir -p $G

echo "=== stage 0: original model -> $G/original ==="
if [ ! -f $C ]; then
  (cd $LAB/upstream && python train.py -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 > $G/train.log 2>&1) \
    || { echo "TRAIN FAILED"; tail -5 $G/train.log; exit 1; }
  [ -f $C ] || { echo "TRAIN produced no $C"; tail -5 $G/train.log; exit 1; }
else echo "exists: $C"; fi

echo "=== stage 1: footprint -> $G/footprint-sens.npz ==="
if [ ! -f $G/footprint-sens.npz ]; then
  (cd $PKG && python vertex_footprint.py --lab $LAB --model $G/original --dataset $DATA --images $IMAGES --out $G/footprint-sens.npz > $G/footprint.log 2>&1) \
    || { echo "FOOTPRINT FAILED"; tail -5 $G/footprint.log; exit 1; }
else echo "exists: $G/footprint-sens.npz"; fi

V0=$(python -c "import torch; sd=torch.load('$C',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))")
echo "original vertices: $V0"; mkdir -p $G/axisB

level () { frac=$1; nv=$(python -c "print(round($frac*$V0))"); name=$G/axisB/v$frac; tag=garden-v$frac
  echo "=== $tag: target $nv vertices ==="
  if [ ! -f $name/report.json ]; then rm -rf $name
    (cd $PKG && python coarse_vis.py --source $C --footprint $G/footprint-sens.npz --base uniform --target-vertices $nv --output $name $CRIT \
       > $name.build.log 2>&1) || { echo "BUILD FAILED $tag"; tail -3 $name.build.log; return; }
  fi
  if [ ! -f $name/finetune/summary.json ]; then rm -rf $name/finetune
    (cd $AN && python coarse_finetune.py --lab $LAB --model $G/original --dataset $DATA --images $IMAGES \
       --coarse $name/coarse-initial.pt --steps 3000 --out $name/finetune > $name.finetune.log 2>&1) || { echo "FINETUNE FAILED $tag"; tail -3 $name.finetune.log; return; }
  fi
  if [ ! -f $name/deliver/summary.json ]; then rm -rf $name/deliver
    bash $PKG/deliver_coarse.sh --coarse $name/finetune/best.pt --output $name/deliver --historical-config none \
       --source $C --dataset $DATA --lab $LAB --images $IMAGES --k $K --r $R --xyz $XYZ --dc $DC --dictionary-rounds 4 \
       > $name.deliver.log 2>&1 || { echo "DELIVER FAILED $tag"; tail -3 $name.deliver.log; return; }
  fi
  python - <<PY | tee -a $G/garden.txt
import json
b=json.load(open('$name/report.json')); f=json.load(open('$name/finetune/summary.json')); d=json.load(open('$name/deliver/summary.json')); t=d['delivery_test']; h=f['history']
print('garden frac $frac | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3), 'worst', round(h[-1]['worst_view_psnr'],2),
      '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4),
      '| K$K R$R $XYZ/$DC', d.get('stage2_version','v1'), d.get('weight','keep'), d.get('index_codec','zstd'))
PY
}
for f in $FRACS; do level $f; done
echo "===== $G/garden.txt ====="; cat $G/garden.txt
