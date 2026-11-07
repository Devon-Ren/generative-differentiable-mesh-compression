#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd3hi
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/rd3hi-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd3hi-%x-%j.err
# rd-hirate-blender-v1.sbatch (2026-11-04, VERSION v1): HIGH-RATE point for one NeRF-Synthetic scene = rd-prune-blender-v5 with ONE lambda and the
# upstream growth limiter raised through train_ckpt_v4 --override (add_percentage, max_points survive --indoor). Everything else identical:
# --indoor, black bg, checkpoints every 1k from 12k, val_select_v2 (official val split, importance prune), level-1.0 delivery, then viewer .msv export.
# Output /scratch/zr2245/$SCENE-rd3hi-lam$LAM-ap$AP-mp$MP (+ -best), table rd-hirate-blender-v1.txt, msv $AN/viewer-v2/msv/$SCENE-hi-lam$LAM.msv.
# 48G host RAM because the mesh is ~3x the lambda 6.2e-4 one (val_select + delivery hold several copies).
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; LAM=${LAM:?set LAM=0.000062 via --export}; AP=${AP:?set AP=1.5 (add_percentage) via --export}; MP=${MP:?set MP=1500000 (max_points) via --export}; INDOOR=${INDOOR:-1}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v2.py; CK=$AN/blender/train_ckpt_v4.py; EXP=$AN/viewer-v1/export_viewer_v1.py; MSV=$AN/viewer-v2/msv
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; TAB=$AN/rd-hirate-blender-v1.txt; TR=$AN/rdgrowth/train_rd_v3.py
echo "scene $SCENE | HIGH-RATE blender, black bg, val-selected | flag '$FLAG' | lam $LAM add_percentage $AP max_points $MP | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $VS $CK $EXP $PKG/stage2_run.sh $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/transforms_train.json $DATA/transforms_val.json $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "'--indoor'" $TR || { echo "FAILED: $TR has no --indoor flag"; exit 1; }
grep -q "update_indoor(ops)" $TR || { echo "FAILED: $TR does not call update_indoor(ops); override wrapper assumption broken"; exit 1; }
grep -q "opt.max_points, int(opt.add_percentage \* V)" $TR || { echo "FAILED: growth limiter line not found in $TR (expected min(opt.max_points, add_percentage*V))"; exit 1; }
grep -q "'--override'" $CK || { echo "FAILED: $CK is not train_ckpt_v4 (no --override)"; exit 1; }
mkdir -p $MSV
for L in $LAM; do
  G=/scratch/zr2245/$SCENE-rd3hi-lam$L-ap$AP-mp$MP; GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L ap $AP mp $MP $FLAG: train+prune (saves every 1k from 12k) ($(date)) ==="
  if [ ! -f $CB ]; then
    python $CK --saves $SAVES --override add_percentage=$AP max_points=$MP --script $TR -- --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --test_iterations 7000 11000 14000 20000 25000 30000 $FLAG > $G/train.log 2>&1 \
      || { echo "FAILED: TRAIN $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback' $G/train.log; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  grep -h "overrides applied after update_indoor" $G/train.log | tail -1 | grep -q "'add_percentage': $AP" || { echo "FAILED: override not confirmed in train.log (expected add_percentage $AP after update_indoor)"; grep -h "\[ckpt\] override" $G/train.log; exit 1; }
  echo "--- growth (first 2 / last 3 densify lines; split == limit means the limiter, not lambda, stopped growth) ---"; grep -h "^\[RD [0-9]*\] V" $G/train.log | head -2; grep -h "^\[RD [0-9]*\] V" $G/train.log | tail -3
  grep -h "^\[RD prune" $G/train.log; grep -h "^\[ckpt\] all\|FAILED" $G/train.log | tail -2
  echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
  echo "=== $SCENE lam $L: val_select ($(date)) ==="
  python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT $SCENE lam $L"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
  grep -h "BEST\|VAL_SELECT_RESULT" $G/val_select.log
  [ -f $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "FAILED: $GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt missing"; exit 1; }
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol on $GB ($(date)) ==="
  G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 \
    || { echo "FAILED: LEVEL-1.0 $SCENE lam $L"; tail -8 $GB/level.log; exit 1; }
  NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE lam $L"; tail -8 $GB/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']; v=json.load(open('$G/original/val_select.json'))
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
last=[r for r in v['rows'] if r['iteration']==30000]
print('$SCENE lam $L HIGH-RATE ap$AP mp$MP RD-GROWTH+PRUNE-mu$MU-BLENDER-VALSEL | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3), 'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'pairs', e['n_pairs'], 'cap', e['max_pairs']) for e in pr], '| blackbg $FLAG')
PY
  echo "=== $SCENE lam $L: viewer .msv export ($(date)) ==="
  A=$NA/deliver/delivery/asset.mslb; [ -f $A ] || A=$(find $NA/deliver -name asset.mslb 2>/dev/null | head -1)
  [ -n "$A" ] && [ -f "$A" ] || { echo "FAILED: no asset.mslb under $NA/deliver"; exit 1; }
  CAM=$G/original/cameras.json; [ -f $CAM ] || CAM=""
  bash $PKG/stage2_run.sh $EXP --asset $A ${CAM:+--cameras $CAM} --out $MSV/$SCENE-hi-lam$L.msv --label "$SCENE hi lam$L" --native-cache $AN/viewer-v2/native-cache > $G/export_msv.log 2>&1 \
    || { echo "FAILED: msv export"; tail -8 $G/export_msv.log; exit 1; }
  grep -h "^VIEWER" $G/export_msv.log
done
echo "===== $TAB ====="; cat $TAB
echo "rd-hirate-blender-v1 $SCENE exit 0  $(date)"
