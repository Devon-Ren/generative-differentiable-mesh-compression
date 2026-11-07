#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=budget-sweep
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/zr2245/analysis/budget-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/budget-%x-%j.err
# budget-sweep-v1.sbatch (2026-10-17): follow-up of G0 (train-at-budget 1.35M v: test 28.165, delivered 13.56 MB / 27.984 vs post-hoc
# collapse 13.93 MB / 26.66 = +1.3 dB).  For SCENE and each B in BUDGETS: train from scratch with --max_points B, then the standard
# level-1.0 protocol (footprint -> no-merge -> 3000-step finetune -> K4096 R256 XYZ/DC stage2-v2 delivery) via run_garden.sh FRACS=1.0.
# -> /scratch/zr2245/$SCENE-budget-<B>/, one row per budget in /scratch/zr2245/analysis/budget-sweep-$SCENE.txt.  Restartable.
# Submit:  sbatch --export=ALL,SCENE=room,IMAGES=images_2,BUDGETS="700000 500000 350000",XYZ=q14 --job-name=room-budget /scratch/zr2245/analysis/budget-sweep-v1.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; BUDGETS=${BUDGETS:?set BUDGETS="700000 500000" via --export}
XYZ=${XYZ:-q14}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=/scratch/zr2245/datasets/m360/$SCENE; TAB=$AN/budget-sweep-$SCENE.txt
echo "scene $SCENE | IMAGES $IMAGES | budgets $BUDGETS | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
(cd $LAB/upstream && python train.py -h 2>&1 | grep -q -- "--max_points") || { echo "train.py does not accept --max_points"; exit 1; }
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
for B in $BUDGETS; do
  G=/scratch/zr2245/$SCENE-budget-$B; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE budget $B: train ($(date)) ==="
  if [ ! -f $CB ]; then
    (cd $LAB/upstream && python train.py -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --max_points $B > $G/train.log 2>&1) \
      || { echo "TRAIN FAILED $SCENE $B"; tail -8 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "no checkpoint for $SCENE $B"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  VB=$(nverts $CB); echo "budget $B -> trained vertices $VB ($(date))"
  echo "=== $SCENE budget $B: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "LEVEL-1.0 FAILED $SCENE $B"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "no delivery for $SCENE $B"; tail -8 $G/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']
print('$SCENE budget $B TRAIN-AT-BUDGET | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC')
PY
done
echo "===== $TAB ====="; cat $TAB
echo "budget-sweep $SCENE exit 0  $(date)"
