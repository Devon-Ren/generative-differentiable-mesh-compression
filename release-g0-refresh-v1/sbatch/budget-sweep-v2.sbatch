#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=g0r
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/zr2245/analysis/g0r-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/g0r-%x-%j.err
# budget-sweep-v2.sbatch (2026-11-08, VERSION g0r-v2): the G0 comparison on one scene, three vertex budgets, current protocol.
#   = budget-sweep-v1.sbatch (train from scratch with upstream --max_points B, then the level-1.0 protocol) + the CONTROL of
#   g0-budget-v1.sbatch (collapse the scene's dense FP32 original to exactly the vertex count the budget run ended with, same
#   3000-step finetune, same K4096 R256 delivery) + DATA / ORIG overrides (T&T scenes) + explicit XYZ.  One row per path and budget
#   in $AN/g0-refresh-$SCENE.txt: 'TRAIN-AT-BUDGET' and 'POST-HOC-COLLAPSE'; the lambda-grown assets of the same scene (rd-prune3 tables)
#   are the third path, compared at equal bytes.  Budgets are chosen as V(lambda asset) / 0.9 so the three paths meet at ~equal V.
#   INDOOR=1 is REFUSED here: upstream's update_indoor() overwrites --max_points after parsing (known pitfall), so an indoor budget run
#   needs the train_ckpt_v4 --override path, which this file does not use.  Restartable: every stage is skipped when its output exists.
#   sbatch --export=ALL,SCENE=truck,IMAGES=images,XYZ=q18,DATA=/scratch/zr2245/datasets/tandt/truck,BUDGETS="1934000 853000 354000" --job-name=truck-g0r .../budget-sweep-v2.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_4}; BUDGETS=${BUDGETS:?set BUDGETS="2000000 900000 400000" via --export}; INDOOR=${INDOOR:-0}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
[ "$INDOOR" = "0" ] || { echo "FAILED: INDOOR=1 is not supported by this sbatch (update_indoor overrides --max_points); indoor scenes need the train_ckpt_v4 --override path"; exit 1; }
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; ORIG=${ORIG:-/scratch/zr2245/$SCENE-vis-collapse}; TAB=$AN/g0-refresh-$SCENE.txt
CORIG=$ORIG/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; MORIG=$ORIG/original; FOOT=$ORIG/footprint-sens.npz
CRIT=${CRIT:-"--geo-weight sens_pos --app-weight none --weight-power 0.5 --weight-clip 0.1,10 --weight-divide-pixel-scale"}
echo "g0-refresh | scene $SCENE | IMAGES $IMAGES | budgets $BUDGETS | original $ORIG | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $PKG/scripts/run_garden.sh $PKG/coarse_vis.py $PKG/vertex_footprint.py $PKG/deliver_coarse.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $DATA/sparse/0/cameras.bin $CORIG $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
ls $PKG/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "FAILED: native module missing in $PKG/native"; exit 1; }
(cd $LAB/upstream && python train.py -h 2>&1 | grep -q -- "--max_points") || { echo "FAILED: train.py does not accept --max_points"; (cd $LAB/upstream && python train.py -h 2>&1 | tail -5); exit 1; }
echo "preflight ok: --max_points accepted; original $(python -c "import torch; sd=torch.load('$CORIG',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))") vertices"
if [ ! -f $FOOT ]; then
  echo "=== footprint of the original (missing) -> $FOOT ($(date)) ==="
  (cd $PKG && python vertex_footprint.py --lab $LAB --model $MORIG --dataset $DATA --images $IMAGES --out $FOOT > $ORIG/footprint.log 2>&1) || { echo "FAILED: FOOTPRINT of the original"; tail -8 $ORIG/footprint.log; exit 1; }
fi
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
row () { # $1 tag  $2 report.json  $3 finetune/summary.json  $4 deliver/summary.json
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$2')); f=json.load(open('$3')); d=json.load(open('$4')); t=d['delivery_test']; h=f['history']
print('$1 | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | $IMAGES')
PY
}
for B in $BUDGETS; do
  G=/scratch/zr2245/$SCENE-budget-$B; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== A: $SCENE budget $B: train with --max_points $B ($(date)) ==="
  if [ ! -f $CB ]; then
    (cd $LAB/upstream && python train.py -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --max_points $B > $G/train.log 2>&1) \
      || { echo "FAILED: TRAIN $SCENE budget $B"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE budget $B"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  VB=$(nverts $CB); echo "budget $B -> trained vertices $VB ($(date))"; grep -h "\[ITER 30000\] Evaluating test" $G/train.log | tail -1
  echo "=== A: $SCENE budget $B: level-1.0 protocol ($(date)) ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 || { echo "FAILED: LEVEL-1.0 $SCENE budget $B"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE budget $B"; tail -8 $G/level.log; exit 1; }
  row "$SCENE budget $B TRAIN-AT-BUDGET" $NA/report.json $NA/finetune/summary.json $NA/deliver/summary.json
  echo "=== B: $SCENE control: collapse the dense original to $VB vertices ($(date)) ==="
  NC=$ORIG/axisB/g0ctrl-$B; mkdir -p $ORIG/axisB
  if [ ! -f $NC/report.json ]; then rm -rf $NC
    (cd $PKG && python coarse_vis.py --source $CORIG --footprint $FOOT --base uniform --target-vertices $VB --output $NC $CRIT > $NC.build.log 2>&1) || { echo "FAILED: CONTROL BUILD $SCENE $B"; tail -8 $NC.build.log; exit 1; }
  fi
  if [ ! -f $NC/finetune/summary.json ]; then rm -rf $NC/finetune
    (cd $AN && python coarse_finetune.py --lab $LAB --model $MORIG --dataset $DATA --images $IMAGES --coarse $NC/coarse-initial.pt --steps 3000 --out $NC/finetune > $NC.finetune.log 2>&1) || { echo "FAILED: CONTROL FINETUNE $SCENE $B"; tail -8 $NC.finetune.log; exit 1; }
  fi
  if [ ! -f $NC/deliver/summary.json ]; then rm -rf $NC/deliver
    bash $PKG/deliver_coarse.sh --coarse $NC/finetune/best.pt --output $NC/deliver --historical-config none --source $CORIG --dataset $DATA --lab $LAB --images $IMAGES --k $K --r $R --xyz $XYZ --dc $DC --dictionary-rounds 4 > $NC.deliver.log 2>&1 || { echo "FAILED: CONTROL DELIVER $SCENE $B"; tail -8 $NC.deliver.log; exit 1; }
  fi
  row "$SCENE budget $B POST-HOC-COLLAPSE" $NC/report.json $NC/finetune/summary.json $NC/deliver/summary.json
done
echo "===== $TAB ====="; cat $TAB
echo "g0-refresh $SCENE exit 0  $(date)"
