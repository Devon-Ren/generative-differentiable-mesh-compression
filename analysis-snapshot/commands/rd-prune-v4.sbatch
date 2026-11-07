#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd4
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zr2245/analysis/rd4-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd4-%x-%j.err
# rd-prune-v4.sbatch (2026-10-31): = rd-prune-v3b.sbatch with the v4 trainer (rdgrowth/train_rd_v4.py: no per-1000 evaluation, render_v4, fused Adam)
# and --data_device cuda; outputs under /scratch/zr2245/$SCENE-rd4-lam$L-mu$MU, rows in analysis/rd-prune4-$SCENE.txt, train wall time printed.  XYZ must be explicit (qN).
# rd-prune-v3b.sbatch (2026-10-20, batch B): = rd-prune-v3.sbatch + DATA override (T&T), SMOKE=0 to skip the smoke stage.
# rd-prune-v3.sbatch (2026-10-19): for SCENE and each L in LAMS: train with rate-controlled growth AND phase-B lambda-prune
# (rdgrowth/train_rd_v3.py: --lam L --dd_mode error --dd_norm mpx --bytes_per_vertex BPV --prune_iters PRUNE --prune_mu MU
# --prune_max_frac MAXF --bytes_per_face BPF), then the level-1.0 protocol (run_garden.sh FRACS=1.0, K4096 R256 XYZ/DC stage2-v2).
# -> /scratch/zr2245/$SCENE-rd3-lam$L-mu$MU/, one row per lambda in /scratch/zr2245/analysis/rd-prune3-$SCENE.txt.  Restartable.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; LAMS=${LAMS:?set LAMS="0.0062 0.00185" via --export}
XYZ=${XYZ:-q14}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/rd-prune4-$SCENE.txt; TR=$AN/rdgrowth/train_rd_v4.py
SMOKE=${SMOKE:-0}; EXTRA=${EXTRA:-}; DEV=${DEV:-cuda}
[ "$XYZ" = "auto" ] && { echo "XYZ=auto is not supported here: pass XYZ=qN (xyz_rule_v1.py)"; exit 1; }
echo "scene $SCENE | IMAGES $IMAGES | lams $LAMS | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/render_v4.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "MISSING: $f"; exit 1; }; done
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
# --- smoke: 1300 iterations, one forced prune round at 1250 (mu 1e6 -> every edge is a candidate, the per-round cap binds),
#     then 50 more iterations of rendering/training on the collapsed mesh and the final save ---
if [ "$SMOKE" = "1" ]; then
SM=$AN/rd4-smoke-$SCENE-$SLURM_JOB_ID; rm -rf $SM; mkdir -p $SM
echo "=== smoke ($(date)) ==="
python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device $DEV -m $SM/original --iterations 1300 --test_iterations 1300 --lam 0.0062 --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
  --prune_iters 1250 --allow_any_prune_iter --prune_mu 1e6 --prune_max_frac 0.1 > $SM/smoke.log 2>&1 \
  || { echo "SMOKE FAILED"; tail -40 $SM/smoke.log; exit 1; }
python - <<PY || { echo "SMOKE CHECK FAILED"; grep -h "RD \|Traceback\|Error" $SM/smoke.log | tail -20; exit 1; }
import json
ev=[json.loads(l) for l in open('$SM/original/rd_log.jsonl')]; p=[e for e in ev if e['event']=='prune']; fin=[e for e in ev if e['event']=='final']
assert p, 'no prune event in smoke'; e=p[0]
print('smoke prune:', {k:e[k] for k in ('iteration','views','vertices_before','vertices_after','pairs','vertices_isolated_removed','faces_before','faces_after','faces_degenerate','faces_duplicate','n_cand','n_pairs','max_pairs','n_flip_rejected','passes','dR_mean_sel')})
assert e['n_pairs'] > 0 and e['vertices_after'] < e['vertices_before'], 'collapse did nothing'
assert e['faces_after'] == e['faces_before'] - e['faces_degenerate'] - e['faces_duplicate'], 'face bookkeeping'
assert fin, 'training did not finish after the prune (render on collapsed mesh failed?)'
print('smoke final:', fin[0])
PY
grep -h "^\[RD" $SM/smoke.log | tail -3
else echo "smoke skipped (SMOKE=0)"; fi
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd4-lam$L-mu$MU; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L mu $MU: train+prune ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device $DEV -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF $EXTRA > $G/train.log 2>&1 \
      || { echo "TRAIN FAILED $SCENE lam $L"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  VB=$(nverts $CB); echo "lam $L mu $MU -> trained vertices $VB ($(date))"; grep -h "^\[RD prune\|^\[v4\]" $G/train.log
  python - <<PY
import json,os
for tag,p in (('v4','$G/original/rd_log.jsonl'),('v3','/scratch/zr2245/$SCENE-rd3-lam$L-mu$MU/original/rd_log.jsonl')):
    if os.path.exists(p):
        fin=[json.loads(l) for l in open(p) if '"final"' in l]
        print('TRAINTIME', tag, '$SCENE lam $L', 'train_wall_s', fin[-1]['elapsed_s'] if fin else None, 'min', round(fin[-1]['elapsed_s']/60,1) if fin else None)
PY
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "LEVEL-1.0 FAILED $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "no delivery for $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dn=[e for e in ev if e['event']=='densify']; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
print('$SCENE lam $L RD-GROWTH+PRUNE-v4-mu$MU | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'cand%.3f' % e['cand_frac'], 'pairs', e['n_pairs'], 'cap', e['max_pairs'], 'flip', e['n_flip_rejected'],
                          'q50/thr %.2f' % (e['score_q50'] / e['thr'] if e['thr'] else float('nan')), 'sel', e['faces_removed_share_sel'], 'all', e['faces_removed_share_all']) for e in pr])
PY
done
echo "===== $TAB ====="; cat $TAB
echo "rd-prune4 $SCENE exit 0  $(date)"
