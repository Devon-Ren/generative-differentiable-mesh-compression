#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rd4reg
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zr2245/analysis/rd4reg-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rd4reg-%x-%j.err
# rd-reg-blender-v1.sbatch (2026-11-05, VERSION v1): ONE NeRF-Synthetic scene, ONE lambda, ONE regulariser config (CFG) with train_rd_v4
# (= train_rd_v3 + mesh_reg_v1: sliver loss WS, Laplacian WL + neighbour-normal WN after the Delaunay rebuild, hard jump filter FA/FK).
# Everything else = rd-prune-blender-v5: --indoor, black bg, checkpoints every 1k from 12k (train_ckpt_v4, no --override), val_select_v2,
# level-1.0 delivery, mesh_stats_v2 on the selected checkpoint, viewer .msv export.
# Output /scratch/zr2245/$SCENE-rd4reg-$CFG-lam$LAM (+ -best), table rd-reg-blender-v1.txt, msv $AN/viewer-v2/msv/$SCENE-reg-$CFG-lam$LAM.msv.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=train; LAM=${LAM:?set LAM=0.00062 via --export}; CFG=${CFG:?set CFG=a|b|c (name only) via --export}; INDOOR=${INDOOR:-1}
WS=${WS:?set WS (w_sliver) via --export}; WL=${WL:?set WL (w_lap) via --export}; WN=${WN:?set WN (w_normal) via --export}; FA=${FA:-20}; FK=${FK:-6}; A0=${A0:-20}; RFROM=${RFROM:-500}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
SAVES="12000 13000 14000 15000 16000 17000 18000 19000 20000 21000 22000 23000 24000 25000 26000 27000 28000 29000"
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; VS=$AN/blender/val_select_v2.py; CK=$AN/blender/train_ckpt_v4.py; EXP=$AN/viewer-v1/export_viewer_v1.py; MSV=$AN/viewer-v2/msv
DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; TAB=$AN/rd-reg-blender-v1.txt; TR=$AN/rdgrowth/train_rd_v4.py; MR=$AN/rdgrowth/mesh_reg_v1.py; MT=$AN/rdgrowth/test_mesh_reg_v1.py; MS=$AN/diag/mesh_stats_v2.py
REF=${REF:-/scratch/zr2245/$SCENE-rd3b4-lam$LAM-mu1.0-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt}
echo "scene $SCENE | REG cfg $CFG: w_sliver $WS (a0 $A0 from $RFROM) w_lap $WL w_normal $WN filter aspect>$FA & longest>$FK*median | blender black bg val-selected | flag '$FLAG' | lam $LAM | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $MR $MT $MS $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $VS $CK $EXP $PKG/stage2_run.sh $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/transforms_train.json $DATA/transforms_val.json $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "VERSION train_rd_v4" $TR || { echo "FAILED: $TR is not train_rd_v4"; exit 1; }
grep -q "from mesh_reg_v1 import" $TR || { echo "FAILED: $TR does not import mesh_reg_v1"; exit 1; }
for a in w_sliver w_lap w_normal filter_aspect filter_k sliver_a0 reg_from; do grep -q "'--$a'" $TR || { echo "FAILED: $TR has no --$a"; exit 1; }; done
grep -q "update_indoor(ops)" $TR || { echo "FAILED: $TR does not call update_indoor(ops)"; exit 1; }
grep -q "'--saves'" $CK || { echo "FAILED: $CK is not a train_ckpt wrapper (no --saves)"; exit 1; }
echo "=== mesh_reg_v1 CPU self-test (fails here, before any GPU time, if the module is broken) ==="
python $MT 2>&1 | tail -1 | tee $AN/rd4reg-$SCENE-$CFG-selftest.txt | grep -q "MESH_REG_TEST OK" || { echo "FAILED: mesh_reg_v1 self-test"; python $MT 2>&1 | tail -15; exit 1; }
mkdir -p $MSV
L=$LAM
G=/scratch/zr2245/$SCENE-rd4reg-$CFG-lam$L; GB=$G-best; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
echo "=== $SCENE cfg $CFG lam $L $FLAG: train_rd_v4 + prune (saves every 1k from 12k) ($(date)) ==="
if [ ! -f $CB ]; then
  python $CK --saves $SAVES --script $TR -- --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
    --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF --test_iterations 7000 11000 14000 20000 25000 30000 $FLAG \
    --w_sliver $WS --sliver_a0 $A0 --w_lap $WL --w_normal $WN --filter_aspect $FA --filter_k $FK --reg_from $RFROM --reg_log_every 500 > $G/train.log 2>&1 \
    || { echo "FAILED: TRAIN $SCENE cfg $CFG"; grep -n -m 8 -i 'error\|Traceback\|assumption broken' $G/train.log; tail -12 $G/train.log; exit 1; }
  [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE cfg $CFG"; tail -8 $G/train.log; exit 1; }
else echo "exists: $CB"; fi
grep -q "^\[REG\] mesh_reg_v1" $G/train.log || { echo "FAILED: train.log has no '[REG] mesh_reg_v1' banner (v4 regularisers not active)"; exit 1; }
grep -q "^\[REG jumpfilter delaunay" $G/train.log || { echo "FAILED: no jump-filter event after the Delaunay rebuild in train.log"; exit 1; }
echo "--- growth (first 2 / last 2 densify lines) ---"; grep -h "^\[RD [0-9]*\] V" $G/train.log | head -2; grep -h "^\[RD [0-9]*\] V" $G/train.log | tail -2
echo "--- reg terms (first 3 / last 3; ratio = |grad_reg| / |grad_main| on the vertices) ---"; grep -h "^\[REG [0-9]*\]" $G/train.log | head -3; grep -h "^\[REG [0-9]*\]" $G/train.log | tail -3
echo "--- jump filter + mesh reports ---"; grep -h "^\[REG jumpfilter\|^\[REG mesh" $G/train.log
grep -h "^\[RD prune" $G/train.log; grep -h "^\[ckpt\] all\|FAILED" $G/train.log | tail -2
echo "--- test curve ---"; grep -h "Evaluating test" $G/train.log | sed 's/\[ITER \([0-9]*\)\] Evaluating test: L1 [0-9.]* PSNR \([0-9.]*\) SSIM \([0-9.]*\) LPIPS \([0-9.]*\).*/\1 \2 \3 \4/'
echo "=== $SCENE cfg $CFG: val_select ($(date)) ==="
python $VS --lab $LAB --dataset $DATA --model $G/original --best $GB/original --delete_unselected > $G/val_select.log 2>&1 || { echo "FAILED: VAL_SELECT $SCENE cfg $CFG"; grep -n -m 8 -i 'error\|Traceback\|FAILED' $G/val_select.log; tail -15 $G/val_select.log; exit 1; }
grep -h "BEST\|VAL_SELECT_RESULT" $G/val_select.log
BCK=$GB/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; [ -f $BCK ] || { echo "FAILED: $BCK missing"; exit 1; }
echo "=== $SCENE cfg $CFG: mesh statistics on the selected checkpoint (+ v3 reference if present) ($(date)) ==="
python $MS --ckpt $BCK --label "$SCENE reg-$CFG lam$L" --out $GB/mesh_stats.json > $GB/mesh_stats.log 2>&1 || { echo "FAILED: MESH_STATS"; tail -8 $GB/mesh_stats.log; exit 1; }
grep -h "^MESH_STATS" $GB/mesh_stats.log | tee -a $TAB
if [ -f $REF ]; then python $MS --ckpt $REF --label "$SCENE v3-ref lam$L" > $G/mesh_stats_ref.log 2>&1 && grep -h "^MESH_STATS" $G/mesh_stats_ref.log | tee -a $TAB; else echo "(no v3 reference checkpoint at $REF)"; fi
echo "=== $SCENE cfg $CFG mu $MU: level-1.0 protocol on $GB ($(date)) ==="
G=$GB DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $GB/level.log 2>&1 \
  || { echo "FAILED: LEVEL-1.0 $SCENE cfg $CFG"; tail -8 $GB/level.log; exit 1; }
NA=$GB/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE cfg $CFG"; tail -8 $GB/level.log; exit 1; }
python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']; v=json.load(open('$G/original/val_select.json'))
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
jf=[e for e in ev if e['event']=='jumpfilter']; me=[e for e in ev if e['event']=='mesh']; rg=[e for e in ev if e['event']=='reg']
last=[r for r in v['rows'] if r['iteration']==30000]
m0=[e for e in me if e['tag']=='delaunay']; m1=[e for e in me if e['tag']=='final']
fm=lambda e: (f"sliver {100*e['sliver_frac']:.2f}% lap {e['lap']:.4f}" + (f" normal {e['normal']:.4f} dih>60 {100*e['dihedral_gt60_frac']:.2f}%" if 'normal' in e else '')) if e else 'n/a'
print('$SCENE lam $L REG-$CFG ws$WS wl$WL wn$WN fa$FA fk$FK RD-GROWTH+PRUNE-mu$MU-BLENDER-VALSEL | best iter', v['best_iteration'], 'val', round(v['best_val_psnr'],3), '(30k val', round(last[0]['val_psnr'],3) if last else 'n/a', ') | vertices', b['vertices'], 'faces', b['faces'],
      '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3), 'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | V@delaunay', dl[0]['vertices'] if dl else None,
      '| jump dropped', [(e['tag'], e['iteration'], e['dropped']) for e in jf], '| mesh@delaunay', fm(m0[0] if m0 else None), '| mesh@final', fm(m1[-1] if m1 else None),
      '| grad ratio reg/main first/last', (round(rg[0]['grad_ratio'],3) if rg and rg[0]['grad_ratio'] is not None else None, round(rg[-1]['grad_ratio'],3) if rg and rg[-1]['grad_ratio'] is not None else None),
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after']) for e in pr], '| blackbg $FLAG')
PY
echo "=== $SCENE cfg $CFG: viewer .msv export ($(date)) ==="
A=$NA/deliver/delivery/asset.mslb; [ -f $A ] || A=$(find $NA/deliver -name asset.mslb 2>/dev/null | head -1)
[ -n "$A" ] && [ -f "$A" ] || { echo "FAILED: no asset.mslb under $NA/deliver"; exit 1; }
CAM=$G/original/cameras.json; [ -f $CAM ] || CAM=""
bash $PKG/stage2_run.sh $EXP --asset $A ${CAM:+--cameras $CAM} --out $MSV/$SCENE-reg-$CFG-lam$L.msv --label "$SCENE reg-$CFG lam$L" --native-cache $AN/viewer-v2/native-cache > $G/export_msv.log 2>&1 \
  || { echo "FAILED: msv export"; tail -8 $G/export_msv.log; exit 1; }
grep -h "^VIEWER" $G/export_msv.log
echo "===== $TAB ====="; cat $TAB
echo "rd-reg-blender-v1 $SCENE cfg $CFG exit 0  $(date)"
