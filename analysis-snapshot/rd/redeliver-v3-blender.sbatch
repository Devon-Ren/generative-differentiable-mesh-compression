#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=redeliv3
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zr2245/analysis/redeliv3-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/redeliv3-%x-%j.err
# redeliver-v3-blender.sbatch (2026-11-06, VERSION rd3): re-delivers EXISTING Blender runs through val_select_v3 (importance recomputed at the chosen
# checkpoint) + the level-1.0 protocol, no retraining.  For each RUN (a model dir /scratch/zr2245/$RUN with original/point_cloud/iteration_{best,30000}
# kept by --delete_unselected), the new delivery goes to /scratch/zr2245/$RUN-best3 (the v2 delivery in $RUN-best is left untouched) and one row is
# appended to $AN/redeliver-v3-blender.txt: v2 vs v3 delivered MB, faces, test PSNR.  ~4-6 min per run.  Restartable (finished stages reused).
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
RUNS=${RUNS:?set RUNS="lego-rd3b4-lam0.0062-mu1.0 ..." via --export (model dir basenames under /scratch/zr2245, scene = text before the first dash)}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; IMAGES=train
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; STAGE2=/scratch/zr2245/shared-collapse-stage2-v2; VS=$AN/blender/val_select_v3.py
TAB=$AN/redeliver-v3-blender.txt
echo "redeliver-v3-blender | runs: $RUNS | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $VS $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $LAB/upstream/train.py $STAGE2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "VERSION = 'val_select_v3" $VS || { echo "FAILED: $VS is not val_select_v3"; exit 1; }
grep -q "def recompute_importance" $VS || { echo "FAILED: $VS lacks recompute_importance"; exit 1; }
python -m py_compile $VS || { echo "FAILED: $VS does not compile"; exit 1; }
for RUN in $RUNS; do
  SCENE=${RUN%%-*}; DATA=/scratch/zr2245/datasets/nerf_synthetic/$SCENE; G=/scratch/zr2245/$RUN; GB=$G-best; GB3=$G-best3
  echo "=== $RUN (scene $SCENE): val_select_v3 + level-1.0 -> $GB3 ($(date)) ==="
  [ -d $G/original/point_cloud ] || { echo "FAILED: $G/original/point_cloud missing (wrong RUN name?)"; exit 1; }
  [ -f $DATA/transforms_val.json ] || { echo "FAILED: $DATA/transforms_val.json missing"; exit 1; }
  echo "checkpoints: $(ls $G/original/point_cloud | tr '\n' ' ')"
  if [ ! -f $GB3/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then
    rm -rf $GB3
    python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB3/original --out $G/original/val_select_v3.json > $G/val_select_v3.log 2>&1 \
      || { echo "FAILED: VAL_SELECT_V3 $RUN"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select_v3.log; tail -15 $G/val_select_v3.log; exit 1; }
  else echo "exists: $GB3/original/point_cloud/iteration_30000/point_cloud_state_dict.pt"; fi
  grep -h "BEST\|state dict scalars\|cleaning render\|importance over\|pruned best\|pruned val\|VAL_SELECT_RESULT" $G/val_select_v3.log
  [ -f $GB3/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "FAILED: $GB3/original/point_cloud/iteration_30000/point_cloud_state_dict.pt missing (val_select_v3 output)"; exit 1; }
  if [ ! -f $GB3/axisB/v1.0/deliver/summary.json ]; then
    G=$GB3 DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB3/level.log 2>&1 \
      || { echo "FAILED: LEVEL-1.0 $RUN"; tail -8 $GB3/level.log; exit 1; }
  else echo "exists: $GB3/axisB/v1.0/deliver/summary.json"; fi
  NA=$GB3/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $RUN"; tail -8 $GB3/level.log; exit 1; }
  python - "$RUN" "$G" "$GB" "$NA" "$XYZ" "$DC" <<'PY' | tee -a $TAB
import json, os, sys
RUN, G, GB, NA, xyz, dc = sys.argv[1:7]
vs = json.load(open(G + '/original/val_select_v3.json')); pr = vs.get('pruned') or {}
b = json.load(open(NA + '/report.json')); d = json.load(open(NA + '/deliver/summary.json')); t = d['delivery_test']
old = GB + '/axisB/v1.0'
if os.path.exists(old + '/deliver/summary.json'):
    d0 = json.load(open(old + '/deliver/summary.json')); b0 = json.load(open(old + '/report.json')); t0 = d0['delivery_test']
    o = f"v2 {d0['file_MB']:.3f} MB V {b0['vertices']} F {b0['faces']} test {t0['psnr']:.3f} -> v3 {d['file_MB']:.3f} MB V {b['vertices']} F {b['faces']} test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f} (bytes x{d['file_MB'] / d0['file_MB']:.3f}, dPSNR {t['psnr'] - t0['psnr']:+.3f})"
else:
    o = f"v2 n/a -> v3 {d['file_MB']:.3f} MB V {b['vertices']} F {b['faces']} test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f}"
print(f"{RUN} | best iter {vs['best_iteration']} val {vs['best_val_psnr']:.3f} (pruned val {pr.get('val_psnr', float('nan')):.3f}) | kept faces recomputed {pr.get('kept_faces_recomputed')} (floor {pr.get('clean_floor')} ss {pr.get('clean_supersample')}) / final-floor-4x {pr.get('kept_faces_final_floor_4x')} / running-max {pr.get('kept_faces_saved_running_max')} | {o} | K4096 R256 {xyz}/{dc}")
PY
done
echo "===== $TAB ====="; cat $TAB
echo "redeliver-v3-blender exit 0  $(date)"
