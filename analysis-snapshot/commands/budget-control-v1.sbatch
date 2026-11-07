#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=budget-ctrl
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zr2245/analysis/budgetctrl-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/budgetctrl-%x-%j.err
# budget-control-v1.sbatch (2026-10-17): the G0 control for any scene = collapse the DENSE model to exactly the vertex count each
# budget run ended with (read from /scratch/zr2245/$SCENE-budget-<B>/original ckpt), 3000-step finetune, same delivery
# (K4096 R256 XYZ/q7 const/ctx) -> /scratch/zr2245/$SCENE-vis-collapse/axisB/g0ctrl-<B>/, rows appended to
# /scratch/zr2245/analysis/budget-sweep-$SCENE.txt as POST-HOC-COLLAPSE.  64G: coarse_vis.py needs >= 48 GB RAM.  Restartable.
# Submit after the scene's budget-sweep job finished:
#   sbatch --export=ALL,SCENE=garden,IMAGES=images_4,BUDGETS="1600000 1000000",XYZ=q15 --job-name=garden-ctrl /scratch/zr2245/analysis/budget-control-v1.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; BUDGETS=${BUDGETS:?set BUDGETS via --export}
XYZ=${XYZ:-q14}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=/scratch/zr2245/datasets/m360/$SCENE; W=/scratch/zr2245/$SCENE-vis-collapse
C=$W/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; M=$W/original
CRIT="--geo-weight sens_pos --app-weight none --weight-power 0.5 --weight-clip 0.1,10 --weight-divide-pixel-scale"
TAB=$AN/budget-sweep-$SCENE.txt
echo "scene $SCENE | budgets $BUDGETS | $XYZ/$DC | job $SLURM_JOB_ID on $(hostname) $(date)"
for f in $C $W/footprint-sens.npz $AN/coarse_finetune.py $PKG/coarse_vis.py $DATA/$IMAGES/.; do [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
rc=0
for B in $BUDGETS; do
  CB=/scratch/zr2245/$SCENE-budget-$B/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  [ -f $CB ] || { echo "MISSING budget checkpoint $CB (budget-sweep for $SCENE $B not finished?)"; rc=1; continue; }
  VB=$(nverts $CB); echo "=== $SCENE control for budget $B: collapse dense to $VB vertices ($(date)) ==="
  NC=$W/axisB/g0ctrl-$B; mkdir -p $W/axisB
  if [ ! -f $NC/report.json ]; then rm -rf $NC
    (cd $PKG && python coarse_vis.py --source $C --footprint $W/footprint-sens.npz --base uniform --target-vertices $VB --output $NC $CRIT > $NC.build.log 2>&1) \
      || { echo "CONTROL BUILD FAILED $SCENE $B"; tail -5 $NC.build.log; rc=1; continue; }
  fi
  if [ ! -f $NC/finetune/summary.json ]; then rm -rf $NC/finetune
    (cd $AN && python coarse_finetune.py --lab $LAB --model $M --dataset $DATA --images $IMAGES --coarse $NC/coarse-initial.pt --steps 3000 --out $NC/finetune > $NC.finetune.log 2>&1) \
      || { echo "CONTROL FINETUNE FAILED $SCENE $B"; tail -5 $NC.finetune.log; rc=1; continue; }
  fi
  if [ ! -f $NC/deliver/summary.json ]; then rm -rf $NC/deliver
    bash $PKG/deliver_coarse.sh --coarse $NC/finetune/best.pt --output $NC/deliver --historical-config none --source $C --dataset $DATA --lab $LAB --images $IMAGES \
       --k $K --r $R --xyz $XYZ --dc $DC --dictionary-rounds 4 > $NC.deliver.log 2>&1 || { echo "CONTROL DELIVER FAILED $SCENE $B"; tail -5 $NC.deliver.log; rc=1; continue; }
  fi
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NC/report.json')); f=json.load(open('$NC/finetune/summary.json')); d=json.load(open('$NC/deliver/summary.json')); t=d['delivery_test']; h=f['history']
print('$SCENE budget $B POST-HOC-COLLAPSE | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC')
PY
done
echo "===== $TAB ====="; cat $TAB
echo "budget-control $SCENE exit $rc  $(date)"; exit $rc
