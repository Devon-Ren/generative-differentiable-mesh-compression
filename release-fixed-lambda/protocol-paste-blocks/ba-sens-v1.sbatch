#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=basens
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/basens-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/basens-%x-%j.err
# ba-sens-v1.sbatch (2026-11-05, VERSION v1): price-constant sensitivity = rd-prune-v3c.sbatch with BPV from --export (B_a = BPV - 2 B_f: BPV 8 / 13 = B_a 4 / 9 vs the default 10 / 6),
#   INDOOR explicit (0 outdoor params, 1 --indoor), output dir /scratch/zr2245/$SCENE-rd3<i>-lam$L-mu$MU-bpv$BPV (never the default dirs), table $AN/ba-sens-v1.txt.
#   Reading: a price error c is identical to lambda -> c*lambda, so the bpv8 / bpv13 points should fall ON the scene's lambda hull, not off it.
# rd-prune-v3c.sbatch (2026-11-01, VERSION v3c): = rd-prune-v3b.sbatch + INDOOR=1 passes --indoor to train_rd_v3.py (upstream update_indoor:
# feature_lr 0.004, add_percentage 1.27, iteration_mesh 12000, start_pruning 3000, prune_size 1300, depth priors off, ...), output dir
# /scratch/zr2245/$SCENE-rd3i-lam$L-mu$MU (outdoor-parameter runs in $SCENE-rd3-... are kept), table rd-prune3i-$SCENE.txt, smoke stage removed,
# XYZ must be explicit (qN).  Same lambda rule, same prune schedule (12000 14000 16000 18000: after the Delaunay rebuild at 11000), same codec.
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; IMAGES=${IMAGES:-images_2}; LAMS=${LAMS:?set LAMS="0.00062 0.0185" via --export}; INDOOR=${INDOOR:?set INDOOR=0 (outdoor params) or 1 (--indoor) via --export}
XYZ=${XYZ:?set XYZ=qN via --export (never auto)}; DC=${DC:-q7}; K=${K:-4096}; R=${R:-256}; BPV=${BPV:-10.0}; BPF=${BPF:-2.0}; MU=${MU:-1.0}; MAXF=${MAXF:-0.15}; PRUNE=${PRUNE:-"12000 14000 16000 18000"}
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
FLAG=""; [ "$INDOOR" = "1" ] && FLAG="--indoor"
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}; TAB=$AN/ba-sens-v1.txt; SUF=""; [ "$INDOOR" = "1" ] && SUF=i; TR=$AN/rdgrowth/train_rd_v3.py
echo "scene $SCENE | IMAGES $IMAGES | lams $LAMS | indoor flag '$FLAG' | B/v $BPV B/face $BPF | prune at $PRUNE mu $MU max_frac $MAXF | $XYZ/$DC K$K R$R | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $TR $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/rd_prune.py $PKG/scripts/run_garden.sh $AN/coarse_finetune.py $DATA/$IMAGES/. $LAB/upstream/train.py /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
grep -q "'--indoor'" $TR || { echo "FAILED: $TR has no --indoor flag"; exit 1; }
nverts () { python -c "import torch; sd=torch.load('$1',map_location='cpu',weights_only=False); print(len(sd['triangles_points']))"; }
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-rd3$SUF-lam$L-mu$MU-bpv$BPV; CB=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; mkdir -p $G
  echo "=== $SCENE lam $L mu $MU $FLAG: train+prune ($(date)) ==="
  if [ ! -f $CB ]; then
    python $TR --lab $LAB -s $DATA -i $IMAGES -r 1 --eval --data_device cpu -m $G/original --iterations 30000 --lam $L --bytes_per_vertex $BPV --bytes_per_face $BPF --dd_mode error --dd_norm mpx \
      --prune_iters $PRUNE --prune_mu $MU --prune_max_frac $MAXF $FLAG > $G/train.log 2>&1 \
      || { echo "FAILED: TRAIN $SCENE lam $L"; tail -12 $G/train.log; exit 1; }
    [ -f $CB ] || { echo "FAILED: no checkpoint for $SCENE lam $L"; tail -8 $G/train.log; exit 1; }
  else echo "exists: $CB"; fi
  VB=$(nverts $CB); echo "lam $L mu $MU -> trained vertices $VB ($(date))"; grep -h "^\[RD prune" $G/train.log
  echo "=== $SCENE lam $L mu $MU: level-1.0 protocol ==="
  G=$G DATA=$DATA IMAGES=$IMAGES FRACS="1.0" K=$K R=$R XYZ=$XYZ DC=$DC bash $PKG/scripts/run_garden.sh > $G/level.log 2>&1 \
    || { echo "FAILED: LEVEL-1.0 $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  NA=$G/axisB/v1.0; [ -f $NA/deliver/summary.json ] || { echo "FAILED: no delivery for $SCENE lam $L"; tail -8 $G/level.log; exit 1; }
  python - <<PY | tee -a $TAB
import json
b=json.load(open('$NA/report.json')); f=json.load(open('$NA/finetune/summary.json')); d=json.load(open('$NA/deliver/summary.json')); t=d['delivery_test']; h=f['history']
ev=[json.loads(l) for l in open('$G/original/rd_log.jsonl')]; dl=[e for e in ev if e['event']=='delaunay']; pr=[e for e in ev if e['event']=='prune']
print('$SCENE lam $L bpv $BPV bpf $BPF RD-GROWTH+PRUNE-mu$MU indoor=$INDOOR | vertices', b['vertices'], 'faces', b['faces'], '| val0', round(f['validation_initial']['psnr'],3), 'val', round(f['validation_best']['psnr'],3),
      'worst', round(h[-1]['worst_view_psnr'],2), '| test', round(f['test_best']['psnr'],3), '| delivered', round(d['file_MB'],3), 'MB test',
      round(t['psnr'],3), round(t['ssim'],4), round(t['lpips'],4), '| K$K R$R $XYZ/$DC | grow: V@delaunay', dl[0]['vertices'] if dl else None,
      '| prune rounds', [(e['iteration'], e['vertices_before'], e['vertices_after'], 'pairs', e['n_pairs'], 'cap', e['max_pairs']) for e in pr], '| $FLAG')
PY
done
echo "===== $TAB ====="; cat $TAB
echo "ba-sens-v1 $SCENE bpv $BPV exit 0  $(date)"
