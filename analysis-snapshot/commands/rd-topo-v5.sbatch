#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=room-topo5
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/topo5-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/topo5-%x-%j.err
# rd-topo-v5.sbatch (2026-10-26): direction A, second design.  For SCENE / LAM and each MODE in MODES:
#   v5   train_rd_v5 = v3 + early fixed-soup restricted Delaunay at REBUILDS[0] (XYZ keeps training) + final canonical
#        rebuild on the delivery grid at REBUILDS[-1] (XYZ frozen) -> decoder-side check -> level-1.0 protocol -> check on
#        the delivered asset (coverage %, label entropy, estimated MB with labels)
#   snap control = v3 + clean + snap + freeze at REBUILDS[-1], no Delaunay (isolates the quantise+freeze cost)
# -> /scratch/zr2245/$SCENE-rd5-$MODE-lam$LAM/, rows in /scratch/zr2245/analysis/rd-topo-v5-$SCENE.txt.  Restartable.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; LAM=${LAM:-0.0062}; MODES=${MODES:-"v5 snap"}
XYZ=${XYZ:?set XYZ=q14 via --export (xyz_rule_v1 value of this scene)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}
BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}; REBUILDS=${REBUILDS:-"20000 26000"}; JIT=${JIT:-0.02}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; S2=/scratch/zr2245/shared-collapse-stage2-v2
DATA=/scratch/zr2245/datasets/m360/$SCENE; TAB=$AN/rd-topo-v5-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v5.py; CK=$AN/rdgrowth/delaunay_check_v1.py
echo "scene $SCENE | IMAGES $IMAGES | lam $LAM | modes $MODES | rebuilds $REBUILDS jitter $JIT | $XYZ/$DC K$K R$R | prune $PRUNE mu $MU | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $CK $AN/rdgrowth/rd_rebuild_v2.py $AN/rdgrowth/test_rd_rebuild_cpu_v2.py $AN/rdgrowth/delaunay_canon_v1.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py $S2/STAGE2_VERSION; do [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
(cd $AN/rdgrowth && python delaunay_canon_v1.py && python test_rd_rebuild_cpu_v2.py | tail -1) || { echo "CPU SELFTEST FAILED"; exit 1; }
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
# --- smoke: 1300 iterations, prune at 1200 (forced), early rebuild at 1240, final at 1250, 50 frozen iterations + save; the check must give 100 % ---
SM=$AN/topo5-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
echo "=== smoke ($(date)) ==="
python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $SM/original --iterations 1300 --test_iterations 1300 --lam 0.0062 --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
  --prune_iters 1200 --allow_any_prune_iter --prune_mu 1e6 --prune_max_frac 0.1 --rebuild_iters 1240 1250 --allow_any_rebuild_iter --xyz $XYZ --rebuild_jitter $JIT > $SM/smoke.log 2>&1 \
  || { echo "SMOKE FAILED"; tail -40 $SM/smoke.log; exit 1; }
grep -h "^\[RD rebuild\|^\[RD v5\|^\[RD prune" $SM/smoke.log | tail -12
python $CK --state $SM/original/point_cloud/iteration_1300/point_cloud_state_dict.pt --xyz $XYZ --jitter $JIT --label "smoke" --out $SM/check.json || { echo "SMOKE CHECK FAILED"; exit 1; }
python - <<PYSMOKE || { echo "SMOKE CHECK FAILED (coverage)"; exit 1; }
import json; c=json.load(open('$SM/check.json')); e=json.load(open('$SM/original/rebuild-1240.json')); r=json.load(open('$SM/original/rebuild.json'))
print('smoke early:', {k:e[k] for k in ('stage','vertices_before','vertices_after_clean','vertices','faces','restriction_calls')}, e['rounds'][0])
print('smoke final:', {k:r[k] for k in ('stage','vertices_before','vertices_after_clean','duplicates_merged','vertices','faces','unused_kept','restriction_calls')}, '| rounds', len(r['rounds']), r['rounds'][0], '| check coverage', c['coverage'], 'on_grid', c['on_grid'], 'H1', c['H1_bits_per_vertex'])
assert e['stage'] == 'early' and r['stage'] == 'final', 'stages'
assert c['on_grid'] and c['quantiser_idempotent'], 'saved vertices are not on the grid'
assert c['coverage'] == 1.0, 'smoke coverage %s: delivered faces are not all Delaunay faces' % c['coverage']
PYSMOKE
for MODE in $MODES; do
  case $MODE in v5) EXTRA="";; snap) EXTRA="--snap_only";; *) echo "unknown MODE $MODE"; exit 1;; esac
  G=/scratch/zr2245/$SCENE-rd5-$MODE-lam$LAM; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $LAM mode $MODE: train v5 (rebuilds $REBUILDS $EXTRA) ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $LAM --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --rebuild_iters $REBUILDS --xyz $XYZ --rebuild_jitter $JIT $EXTRA > $G/train.log 2>&1 \
      || { echo "TRAIN FAILED $SCENE $MODE"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "no checkpoint for $SCENE $MODE"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  echo "$MODE -> trained vertices $(nverts $CB) ($(date))"; grep -h "^\[RD prune\|^\[RD rebuild\|^\[RD v5" $G/train.log; grep -h "ITER 30000\] Evaluating test" $G/train.log | tail -1
  python $CK --state $CB --xyz $XYZ --jitter $JIT --label "$SCENE $MODE lam$LAM checkpoint" --out $G/check-ckpt.json | tee -a $TAB || { echo "CHECK FAILED (checkpoint)"; exit 1; }
  echo "=== $SCENE $MODE: level-1.0 protocol ($(date)) ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "LEVEL-1.0 FAILED $SCENE $MODE"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "no delivery for $SCENE $MODE"; tail -8 $G/level.log; exit 1; }
  python $CK --state $NA/deliver/shared-final.pt --xyz $XYZ --jitter $JIT --label "$SCENE $MODE lam$LAM delivered" --asset $NA/deliver/delivery/asset.mslb --stage2 $S2 --out $G/check-delivered.json | tee -a $TAB \
    || { echo "CHECK FAILED (delivered)"; exit 1; }
  python - <<PYROW | tee -a $TAB
import json, os, glob
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']
r=json.load(open('$G/original/rebuild.json')); c=json.load(open('$G/check-delivered.json'))
early=[json.load(open(p)) for p in sorted(glob.glob('$G/original/rebuild-*.json')) if json.load(open(p))['stage']=='early']
e0=(early[0]['rounds'][0] if early and early[0]['rounds'] else {}); r0=(r['rounds'][0] if r['rounds'] else {})
est=c.get('est_MB_with_labels_H1')
print('$SCENE lam $LAM RD-V5-$MODE rebuilds $REBUILDS | vertices', b['vertices'], 'faces', b['faces'], '| val', round(f['validation_best']['psnr'],3), '| test', round(f['test_best']['psnr'],3),
      '| delivered', round(d['file_MB'],3), 'MB test', round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC',
      '| early: V', (early[0]['vertices_before'] if early else None), '->', (early[0]['vertices'] if early else None), 'faces', (early[0]['faces_before'] if early else None), '->', (early[0]['faces'] if early else None),
      'retained', e0.get('soup_faces_retained'), 'new', e0.get('output_faces_new'), 'post-clean -', (early[0].get('faces_removed_by_post_clean') if early else None),
      '| final (' + r['stage'] + '): V', r['vertices_before'], '->', r['vertices'], 'faces', r['faces_before'], '->', r['faces'], 'rounds', len(r['rounds']), 'dup', r['duplicates_merged'],
      'retained', r0.get('soup_faces_retained'), 'new', r0.get('output_faces_new'), 'unused kept', r.get('unused_kept', r.get('unused_final_event')), 'post-clean -', r.get('faces_removed_by_post_clean'),
      '| coverage %.2f %%' % (100*c['coverage']), '| labels H0/H1', c['H0_bits_per_vertex'], c['H1_bits_per_vertex'], 'bits/v | topology now', c.get('topology_bits_per_vertex'), 'bits/v',
      '| est', est, 'MB with labels (%.0fx of 742 MB)' % (742/est) if est else '')
PYROW
done
echo "===== $TAB ====="; cat $TAB
echo "references (Room lam 0.0062): v3 5.228 MB / test 27.382 (142x) | v4 (single rebuild, drifting soup) 4.56 MB / 25.61, est 3.53 MB with labels"
echo "v3/v1 Room RD curve near 3.5 MB: 3.01 MB / 26.51 and 5.70 MB / 27.37 (v1 lambda sweep) -> interpolated ~26.7 at 3.5 MB; v3 ~+0.1 above v1"
echo "pass rule: v5 delivered test >= 27.28 very viable | 26.98..27.28 compare est-MB point with the v3 curve (needs >= +0.15 dB above it) | < 26.98 dead; snap row = cost of quantise+freeze alone"
echo "rd-topo-v5 $SCENE exit 0  $(date)"
