#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --job-name=g0-budget
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/zr2245/analysis/g0-%j.out
#SBATCH --error=/scratch/zr2245/analysis/g0-%j.err
# g0-budget-v1.sbatch (2026-10-17): G0 = "vertex budget at TRAINING time" vs "post-hoc collapse to the same vertex count".
# For each BUDGET in BUDGETS:
#   A. train Room from scratch with --max_points BUDGET (add_new_gs caps vertices at min(cap, 1.27 x current)),
#      then the standard level-1.0 protocol (no merge -> 3000-step finetune -> K4096 R256 q14/q7 stage2-v2 delivery)
#      via run_garden.sh with FRACS=1.0  -> /scratch/zr2245/room-budget-<BUDGET>/
#   B. control: collapse the DENSE Room original (3.19M v) to exactly the vertex count A ended with, same finetune
#      + same delivery  -> /scratch/zr2245/room-vis-collapse/axisB/g0ctrl-<BUDGET>/
# Table: /scratch/zr2245/analysis/g0-budget-v1.txt (one A row + one B row per budget).  Restartable (stages skip when
# their output exists).  64G because coarse_vis.py (control collapse) needs >= 48 GB RAM.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
BUDGETS=${BUDGETS:-"1600000 1000000"}        # ~50 % and ~31 % of the dense 3,193,592 vertices
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
AN=/scratch/zr2245/analysis
PKG=/scratch/zr2245/analysis/vis_collapse-v12
DATA=/scratch/zr2245/datasets/m360/room
IMAGES=images_2
W=/scratch/zr2245/room-vis-collapse
CROOM=/scratch/zr2245/room-fixed-sharing-0N4OSU/run/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
MROOM=/scratch/zr2245/room-fixed-sharing-0N4OSU/run/original
CRIT="--geo-weight sens_pos --app-weight none --weight-power 0.5 --weight-clip 0.1,10 --weight-divide-pixel-scale"
K=4096; R=256; XYZ=q14; DC=q7
TAB=$AN/g0-budget-v1.txt
echo "job $SLURM_JOB_ID on $(hostname) start $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $CROOM $W/footprint-sens.npz $DATA/$IMAGES/. /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do
  [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
ls $PKG/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "native module missing in $PKG/native"; exit 1; }
(cd $LAB/upstream && python train.py -h 2>&1 | grep -q -- "--max_points") || { echo "train.py does not accept --max_points (check arguments/__init__.py ParamGroup)"; (cd $LAB/upstream && python train.py -h 2>&1 | tail -5); exit 1; }
echo "preflight ok: --max_points accepted"

nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
row () { # $1 tag  $2 report.json  $3 finetune/summary.json  $4 deliver/summary.json
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$2')); f=json.load(open('$3')); d=json.load(open('$4')); t=d['delivery_test']; h=f['history']
print('$1 | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC')
PY
}

for B in $BUDGETS; do
  G=/scratch/zr2245/room-budget-$B; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== A: train Room with --max_points $B -> $G/original  ($(date)) ==="
  if [ ! -f $CB ]; then
    (cd $LAB/upstream && python train.py -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --max_points $B > $G/train.log 2>&1) \
      || { echo "TRAIN FAILED budget $B"; tail -8 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "TRAIN produced no checkpoint for budget $B"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  VB=$(nverts $CB); echo "budget $B -> trained vertices $VB  ($(date))"
  grep -i -m 3 -E "psnr|test" $G/train.log || true
  echo "=== A: level-1.0 protocol on the budget model (footprint -> no-merge -> finetune -> deliver) ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "LEVEL-1.0 FAILED budget $B"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0
  [ -f $NA/deliver/summary.json ] || { echo "no delivery for budget $B"; tail -8 $G/level.log; exit 1; }
  row "G0 budget $B TRAIN-AT-BUDGET" $NA/report.json $NA/finetune/summary.json $NA/deliver/summary.json

  echo "=== B: control = collapse dense Room to $VB vertices  ($(date)) ==="
  NC=$W/axisB/g0ctrl-$B; mkdir -p $W/axisB
  if [ ! -f $NC/report.json ]; then rm -rf $NC
    (cd $PKG && python coarse_vis.py --source $CROOM --footprint $W/footprint-sens.npz --base uniform --target-vertices $VB --output $NC $CRIT \
       > $NC.build.log 2>&1) || { echo "CONTROL BUILD FAILED $B"; tail -5 $NC.build.log; exit 1; }
  fi
  if [ ! -f $NC/finetune/summary.json ]; then rm -rf $NC/finetune
    (cd $AN && python coarse_finetune.py --lab $LAB --model $MROOM --dataset $DATA --images $IMAGES \
       --coarse $NC/coarse-initial.pt --steps 3000 --out $NC/finetune > $NC.finetune.log 2>&1) || { echo "CONTROL FINETUNE FAILED $B"; tail -5 $NC.finetune.log; exit 1; }
  fi
  if [ ! -f $NC/deliver/summary.json ]; then rm -rf $NC/deliver
    bash $PKG/deliver_coarse.sh --coarse $NC/finetune/best.pt --output $NC/deliver --historical-config none \
       --source $CROOM --dataset $DATA --lab $LAB --images $IMAGES --k $K --r $R --xyz $XYZ --dc $DC --dictionary-rounds 4 \
       > $NC.deliver.log 2>&1 || { echo "CONTROL DELIVER FAILED $B"; tail -5 $NC.deliver.log; exit 1; }
  fi
  row "G0 budget $B POST-HOC-COLLAPSE" $NC/report.json $NC/finetune/summary.json $NC/deliver/summary.json
done
echo "===== $TAB ====="; cat $TAB
echo "reference: dense Room original test 28.263; v1.0 K4096R256 q14/q7 31.87 MB / 28.091; v0.5 16.89 MB / 27.090"
echo "g0 exit 0  end $(date)"
