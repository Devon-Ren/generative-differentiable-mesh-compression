#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=room-topo4
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/topo4-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/topo4-%x-%j.err
# rd-topo-v4.sbatch (2026-10-26): direction A probe.  For SCENE and each L in LAMS: train_rd_v4 (= v3 + late quantised
# Delaunay rebuild at REBUILD, XYZ frozen) -> decoder-side check on the checkpoint -> level-1.0 protocol (unchanged) ->
# check on shared-final.pt + delivered asset (coverage %, label entropy, topology bytes now vs estimated).
# Also runs the check on the existing v3 checkpoint as the baseline (expected coverage ~1-2 %).
# -> /scratch/zr2245/$SCENE-rd4-lam$L/, rows in /scratch/zr2245/analysis/rd-topo-v4-$SCENE.txt.  Restartable.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; LAMS=${LAMS:?set LAMS="0.0062 0.00185" via --export}
XYZ=${XYZ:?set XYZ=q14 via --export (xyz_rule_v1 value of this scene)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}
BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}; REBUILD=${REBUILD:-26000}; JIT=${JIT:-0.02}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; S2=/scratch/zr2245/shared-collapse-stage2-v2
DATA=/scratch/zr2245/datasets/m360/$SCENE; TAB=$AN/rd-topo-v4-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v4.py; CK=$AN/rdgrowth/delaunay_check_v1.py
echo "scene $SCENE | IMAGES $IMAGES | lams $LAMS | rebuild $REBUILD jitter $JIT | $XYZ/$DC K$K R$R | prune $PRUNE mu $MU | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $CK $AN/rdgrowth/rd_rebuild_v1.py $AN/rdgrowth/delaunay_canon_v1.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py $S2/STAGE2_VERSION; do [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
(cd $AN/rdgrowth && python delaunay_canon_v1.py && python test_rd_rebuild_cpu.py | tail -1) || { echo "CPU SELFTEST FAILED"; exit 1; }
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
# --- baseline: the existing v3 checkpoint of this scene (first lambda), decoder-side coverage ---
L0=$(echo $LAMS | awk '{print $1}'); V3CK=/scratch/zr2245/$SCENE-rd3-lam$L0-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
if [ -f $V3CK ]; then
  echo "=== baseline check on v3 checkpoint ($(date)) ==="
  python $CK --state $V3CK --xyz $XYZ --jitter $JIT --label "$SCENE v3 lam$L0 (baseline)" --out $AN/topo4-baseline-$SCENE-lam$L0.json | tee -a $TAB || echo "baseline check FAILED (non-fatal)"
else echo "no v3 checkpoint at $V3CK (baseline skipped)"; fi
# --- smoke: 1300 iterations, prune at 1200 (forced), rebuild at 1250, then 50 iterations on the rebuilt mesh + save; check must give 100 % ---
SM=$AN/topo4-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
echo "=== smoke ($(date)) ==="
python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1300 --test_iterations 1300 --lam 0.0062 --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
  --prune_iters 1200 --allow_any_prune_iter --prune_mu 1e6 --prune_max_frac 0.1 --rebuild_iter 1250 --allow_any_rebuild_iter --xyz $XYZ --rebuild_jitter $JIT > $SM/smoke.log 2>&1 \
  || { echo "SMOKE FAILED"; tail -40 $SM/smoke.log; exit 1; }
grep -h "^\[RD rebuild\|^\[RD v4\|^\[RD prune" $SM/smoke.log | tail -8
python $CK --state $SM/original/point_cloud/iteration_1300/point_cloud_state_dict.pt --xyz $XYZ --jitter $JIT --label "smoke" --out $SM/check.json || { echo "SMOKE CHECK FAILED"; exit 1; }
python - <<PY || { echo "SMOKE CHECK FAILED (coverage)"; exit 1; }
import json; c=json.load(open('$SM/check.json')); r=json.load(open('$SM/original/rebuild.json'))
print('smoke rebuild:', {k:r[k] for k in ('vertices_before','vertices_after_clean','vertices','faces','unused_kept')}, '| rounds', len(r['rounds']), '| check coverage', c['coverage'], 'on_grid', c['on_grid'], 'H1', c['H1_bits_per_vertex'])
assert c['on_grid'] and c['quantiser_idempotent'], 'saved vertices are not on the grid'
assert c['coverage'] == 1.0, 'smoke coverage %s: delivered faces are not all Delaunay faces' % c['coverage']
PY
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd4-lam$L; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L: train v4 (rebuild $REBUILD) ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --rebuild_iter $REBUILD --xyz $XYZ --rebuild_jitter $JIT > $G/train.log 2>&1 \
      || { echo "TRAIN FAILED $SCENE lam $L"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  echo "lam $L -> trained vertices $(nverts $CB) ($(date))"; grep -h "^\[RD prune\|^\[RD rebuild\|^\[RD v4" $G/train.log; grep -h "ITER 30000\] Evaluating test" $G/train.log | tail -1
  python $CK --state $CB --xyz $XYZ --jitter $JIT --label "$SCENE v4 lam$L checkpoint" --out $G/check-ckpt.json | tee -a $TAB || { echo "CHECK FAILED (checkpoint)"; exit 1; }
  echo "=== $SCENE lam $L: level-1.0 protocol ($(date)) ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "LEVEL-1.0 FAILED $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "no delivery for $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  python $CK --state $NA/deliver/shared-final.pt --xyz $XYZ --jitter $JIT --label "$SCENE v4 lam$L delivered" --asset $NA/deliver/delivery/asset.mslb --stage2 $S2 --out $G/check-delivered.json | tee -a $TAB \
    || { echo "CHECK FAILED (delivered)"; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']
r=json.load(open('$G/original/rebuild.json')); c=json.load(open('$G/check-delivered.json'))
est=c.get('est_MB_with_labels_H1'); print('$SCENE lam $L RD-V4-REBUILD$REBUILD | vertices', b['vertices'], 'faces', b['faces'], '| val', round(f['validation_best']['psnr'],3), '| test', round(f['test_best']['psnr'],3),
      '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC',
      '| rebuild: V', r['vertices_before'], '->', r['vertices'], 'faces', r['faces_before'], '->', r['faces'], 'rounds', len(r['rounds']), 'dup', r['rounds'][0]['duplicates_merged'], 'unused kept', r['unused_kept'],
      '| coverage %.2f %%' % (100*c['coverage']), '| labels H0/H1', c['H0_bits_per_vertex'], c['H1_bits_per_vertex'], 'bits/v | topology now', c.get('topology_bits_per_vertex'), 'bits/v',
      '| est', est, 'MB with labels (%.0fx of 742 MB)' % (742/est) if est else '')
PY
done
echo "===== $TAB ====="; cat $TAB
echo "v3 reference (rd-prune3-room): lam 0.0062 5.228 MB / test 27.382 (142x); lam 0.00185 12.572 MB / 28.004 (59x)"
echo "rd-topo-v4 $SCENE exit 0  $(date)"
