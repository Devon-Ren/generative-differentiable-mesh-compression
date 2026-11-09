#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=g0rb
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/zr2245/analysis/g0rb-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/g0rb-%x-%j.err
# budget-sweep-blender-v1.sbatch (2026-11-09, VERSION g0rb-v1): the G0 comparison (train-at-budget vs post-hoc collapse) on ONE NeRF-Synthetic scene,
# Blender protocol: --indoor, black background, checkpoints every 1k from 12k (train_ckpt_v2 wrapping upstream train.py --max_points B), the best
# OFFICIAL-val checkpoint chosen by val_select_v3 (importance recomputed at the final floor / 4x), then the level-1.0 protocol on $G-best.
# CONTROL: the scene's val-selected dense original ($ORIG, default $SCENE-blender4-best) collapsed by coarse_vis to the vertex count the budget
# run ended with, same 3000-step finetune, same K4096 R256 delivery.  Budgets come from LAMS_AUTO: for each lambda L the delivered vertex count of
# /scratch/zr2245/$SCENE-rd3b4-lam$L-mu1.0-best3 (the val_select_v3 re-delivery, the hull's points) divided by 0.9, so the three paths meet at ~equal V;
# a lambda whose -best3 delivery is missing is SKIPPED with a note (never a failure).  Explicit BUDGETS="..." are appended.  Rows -> $AN/g0-refresh-blender-$SCENE.txt
# ('TRAIN-AT-BUDGET-BLENDER' / 'POST-HOC-COLLAPSE-BLENDER').  A failing CONTROL stage prints FAILED and moves on (the budget row is kept).  Restartable.
#   sbatch --export=ALL,SCENE=lego,XYZ=q14,LAMS_AUTO="0.00062 0.00185 0.0062 0.0185" --job-name=lego-b21-g0rb .../budget-sweep-blender-v1.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; INDOOR=${INDOOR:-1}; LAMS_AUTO=${LAMS_AUTO:-"0.00062 0.00185 0.0062 0.0185"}; BUDGETS=${BUDGETS:-}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v3.py; CK=$AN/blender/train_ckpt_v2.py
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; ORIG=${ORIG:-/scratch/zr2245/$SCENE-blender4-best}; TAB=$AN/g0-refresh-blender-$SCENE.txt
CORIG=$ORIG/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; MORIG=$ORIG/original; FOOT=$ORIG/footprint-sens.npz
CRIT=${CRIT:-"--geo-weight sens_pos --app-weight none --weight-power 0.5 --weight-clip 0.1,10 --weight-divide-pixel-scale"}
echo "g0-refresh-blender | scene $SCENE (black bg, val-selected) | lams_auto '$LAMS_AUTO' budgets '$BUDGETS' | indoor flag '$FLAG' | original $ORIG | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $PKG/scripts/run_garden.sh $PKG/coarse_vis.py $PKG/vertex_footprint.py $PKG/deliver_coarse.sh $AN/coarse_finetune.py $DATA/transforms_train.json $DATA/transforms_val.json $DATA/transforms_test.json $CORIG $VS $CK $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
ls $PKG/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "FAILED: native module missing in $PKG/native"; exit 1; }
grep -q "VERSION = 'val_select_v3" $VS || { echo "FAILED: $VS is not val_select_v3"; exit 1; }
grep -q "VERSION = 'train_ckpt_v2" $CK || { echo "FAILED: $CK is not train_ckpt_v2"; exit 1; }
(cd $LAB/upstream && python train.py -h 2>&1 | grep -q -- "--max_points") || { echo "FAILED: train.py does not accept --max_points"; exit 1; }
grep -A 20 "def update_indoor" $LAB/upstream/arguments/__init__.py | grep -q "max_points" && { echo "FAILED: this upstream's update_indoor() sets max_points; the CLI budget would be overwritten"; exit 1; }
echo "preflight ok: --max_points accepted, update_indoor() leaves it alone; original $(python -c "import torch; sd=torch.load('$CORIG',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))") vertices"
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
for L in $LAMS_AUTO; do
  RJ=/scratch/zr2245/$SCENE-rd3b4-lam$L-mu1.0-best3/axisB/v1.0/report.json
  if [ -f $RJ ]; then V=$(python -c "import json; print(json.load(open('$RJ'))['vertices'])"); B=$(( (V * 10 + 8) / 9 )); BUDGETS="$BUDGETS $B"; echo "lambda $L: delivered V $V -> budget $B"
  else echo "note: lambda $L has no -best3 delivery at $RJ; skipped"; fi
done
BUDGETS=$(echo $BUDGETS | tr ' ' '\n' | sort -nr | uniq | tr '\n' ' '); [ -n "$(echo $BUDGETS)" ] || { echo "FAILED: no budgets (no lambda delivery found and BUDGETS empty)"; exit 1; }
echo "budgets for $SCENE: $BUDGETS"
if [ ! -f $FOOT ]; then
  echo "=== footprint of the original (missing) -> $FOOT ($(date)) ==="
  (cd $PKG && python vertex_footprint.py --lab $LAB --model $MORIG --dataset $DATA --images $IMAGES --out $FOOT > $ORIG/footprint.log 2>&1) || { echo "FAILED: FOOTPRINT of the original (controls will be skipped)"; tail -8 $ORIG/footprint.log; }
fi
row () { # $1 tag  $2 report.json  $3 finetune/summary.json  $4 deliver/summary.json  $5 extra
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$2')); f=json.load(open('$3')); d=json.load(open('$4')); t=d['delivery_test']; h=f['history']
print('$1 | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | $IMAGES | blackbg $FLAG | $5')
PY
}
nfail=0
for B in $BUDGETS; do
  G=/scratch/zr2245/$SCENE-budgetb-$B; GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== A: $SCENE budget $B: train.py --max_points $B $FLAG, checkpoints every 1k from 12k ($(date)) ==="
  if [ ! -f $CB ]; then
    python $CK --saves $SAVES --script $LAB/upstream/train.py -- -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --max_points $B --test_iterations 7000 30000 $FLAG > $G/train.log 2>&1 \
      || { echo "FAILED: TRAIN $SCENE budget $B"; grep -n -m 6 -i 'error\|Traceback' $G/train.log; tail -10 $G/train.log; nfail=$((nfail+1)); continue; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE budget $B"; tail -8 $G/train.log; nfail=$((nfail+1)); continue; }
  else echo "exists: $CB"; fi
  grep -h "^\[ckpt\] all\|FAILED" $G/train.log | tail -2; grep -h "\[ITER 30000\] Evaluating test" $G/train.log | tail -1
  echo "=== A: $SCENE budget $B: val_select_v3 ($(date)) ==="
  if [ ! -f $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then
    rm -rf $GB
    python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT $SCENE budget $B"; grep -n -m 6 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -10 $G/val_select.log; nfail=$((nfail+1)); continue; }
  fi
  grep -h "BEST\|VAL_SELECT_RESULT" $G/val_select.log | tail -2
  [ -f $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "FAILED: val_select output missing for $SCENE budget $B"; nfail=$((nfail+1)); continue; }
  VB=$(nverts $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt); echo "budget $B -> val-selected pruned vertices $VB"
  echo "=== A: $SCENE budget $B: level-1.0 protocol on $GB ($(date)) ==="
  if [ ! -f $GB/axisB/v1.0/deliver/summary.json ]; then
    G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 || { echo "FAILED: LEVEL-1.0 $SCENE budget $B"; tail -8 $GB/level.log; nfail=$((nfail+1)); continue; }
  fi
  NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE budget $B"; tail -8 $GB/level.log; nfail=$((nfail+1)); continue; }
  row "$SCENE budget $B TRAIN-AT-BUDGET-BLENDER" $NA/report.json $NA/finetune/summary.json $NA/deliver/summary.json "max_points $B"
  [ -f $FOOT ] || { echo "skip control for budget $B: no footprint"; continue; }
  echo "=== B: $SCENE control: collapse the dense original to $VB vertices ($(date)) ==="
  NC=$ORIG/axisB/g0ctrl-$B; mkdir -p $ORIG/axisB
  if [ ! -f $NC/report.json ]; then rm -rf $NC
    (cd $PKG && python coarse_vis.py --source $CORIG --footprint $FOOT --base uniform --target-vertices $VB --output $NC $CRIT > $NC.build.log 2>&1) || { echo "FAILED: CONTROL BUILD $SCENE $B"; tail -8 $NC.build.log; nfail=$((nfail+1)); continue; }
  fi
  if [ ! -f $NC/finetune/summary.json ]; then rm -rf $NC/finetune
    (cd $AN && python coarse_finetune.py --lab $LAB --model $MORIG --dataset $DATA --images $IMAGES --coarse $NC/coarse-initial.pt --steps 3000 --out $NC/finetune > $NC.finetune.log 2>&1) || { echo "FAILED: CONTROL FINETUNE $SCENE $B"; tail -8 $NC.finetune.log; nfail=$((nfail+1)); continue; }
  fi
  if [ ! -f $NC/deliver/summary.json ]; then rm -rf $NC/deliver
    bash $PKG/deliver_coarse.sh --coarse $NC/finetune/best.pt --output $NC/deliver --historical-config none --source $CORIG --dataset $DATA --lab $LAB --images $IMAGES --k $K --r $R --xyz $XYZ --dc $DC --dictionary-rounds 4 > $NC.deliver.log 2>&1 || { echo "FAILED: CONTROL DELIVER $SCENE $B"; tail -8 $NC.deliver.log; nfail=$((nfail+1)); continue; }
  fi
  row "$SCENE budget $B POST-HOC-COLLAPSE-BLENDER" $NC/report.json $NC/finetune/summary.json $NC/deliver/summary.json "target-vertices $VB"
done
echo "===== $TAB ====="; cat $TAB 2>/dev/null
echo "g0-refresh-blender $SCENE exit 0 ($nfail stage failures)  $(date)"
