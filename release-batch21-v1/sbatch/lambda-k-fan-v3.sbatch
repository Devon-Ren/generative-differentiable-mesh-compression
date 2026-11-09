#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=kfan3
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zr2245/analysis/kfan3-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/kfan3-%x-%j.err
# lambda-k-fan-v3.sbatch (2026-11-09, VERSION v3): = lambda-k-fan-v2 (the joint lambda x dictionary-size fan + DC q8, no training) generalised to the
# other datasets: XYZ via --export (overrides xyz_of), SUFFIX appended to the run dir (Blender hull points live in $SCENE-rd3b4-lam$L-mu1.0-best3),
# DATA / IMAGES via --export (Deep Blending: datasets/db/$SCENE, images; NeRF-Synthetic: datasets/nerf_synthetic/$SCENE, train, transforms_*.json),
# the dataset check accepting either a COLMAP image folder or Blender transforms.  Same rows, same table $AN/lambda-k-fan-v1.txt.  A lambda whose
# delivery is missing is skipped with FAILED: MISSING (counted, not fatal).  Restartable.
#   sbatch --export=ALL,SCENE=drjohnson,TAG=rd3i,XYZ=q15,IMAGES=images,DATA=/scratch/zr2245/datasets/db/drjohnson --job-name=drjohnson-b21-kfan .../lambda-k-fan-v3.sbatch
#   sbatch --export=ALL,SCENE=lego,TAG=rd3b4,SUFFIX=-best3,XYZ=q14,IMAGES=train,DATA=/scratch/zr2245/datasets/nerf_synthetic/lego --job-name=lego-b21-kfan .../lambda-k-fan-v3.sbatch
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; TAG=${TAG:?set TAG=rd3i (indoor / Deep Blending) or rd3 (outdoor) or rd3b4 (Blender) via --export}; IMAGES=${IMAGES:-images_2}; SUFFIX=${SUFFIX:-}
LAMS=${LAMS:-"0.00062 0.00185 0.0062 0.0185"}; KRS=${KRS:-"4096 1|1024 256|4096 1024|16384 1024"}
AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; TAB=$AN/lambda-k-fan-v1.txt
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}
xyz_of () { case $1 in room) echo q14;; garden|bonsai|counter|flowers|drjohnson|playroom) echo q15;; kitchen|bicycle|stump|treehill) echo q16;; truck|train) echo q18;; chair|drums|ficus|hotdog|lego|materials|mic|ship) echo q14;; *) echo "FAILED: no XYZ recorded for $1" >&2; return 1;; esac; }
XYZ=${XYZ:-$(xyz_of $SCENE)} || exit 1
case $XYZ in q1[3-8]) ;; *) echo "FAILED: XYZ must be q13..q18, got $XYZ"; exit 1;; esac
echo "lambda-k-fan-v3 | scene $SCENE tag $TAG suffix '$SUFFIX' | DATA $DATA/$IMAGES | lams $LAMS | K x R $KRS | XYZ $XYZ | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $PKG/codec_variants.py $PKG/deliver_coarse.sh $PKG/stage2_run.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
[ -e $DATA/$IMAGES/. ] || [ -f $DATA/transforms_train.json ] || { echo "FAILED: neither $DATA/$IMAGES nor $DATA/transforms_train.json exists"; exit 1; }
nf=0
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-$TAG-lam$L-mu1.0$SUFFIX; d=$G/axisB/v1.0/deliver; MESH=$G/axisB/v1.0/finetune/best.pt; C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  miss=0; for f in $d/shared-final.pt $d/config.json $d/summary.json $MESH $C; do [ -e $f ] || { echo "FAILED: MISSING $f"; miss=1; }; done
  [ $miss = 0 ] || { nf=$((nf+1)); continue; }
  python -c "import json; c=json.load(open('$d/config.json')); k=c.get('k',4096); r=c.get('r',256); assert (k,r)==(4096,256), ('baseline delivery is not K4096 R256', k, r); assert c.get('xyz')=='$XYZ', ('delivered xyz', c.get('xyz'), 'differs from', '$XYZ')" || { echo "FAILED: $SCENE lam $L baseline delivery is not K4096 R256 / $XYZ"; nf=$((nf+1)); continue; }
  echo "=== $SCENE lam $L: baseline + (a) DC q8 on $d (XYZ $XYZ) ($(date +%H:%M)) ==="
  for V in "const/ctx q7" "const/ctx q8"; do set -- $V; W=$1; DC=$2
    if python -c "import json,sys; s=json.load(open('$d/codec-variants/sweep.json')); sys.exit(0 if any((r['weight']+'/'+r['index_codec'],r['xyz'],r['dc'])==('$W','$XYZ','$DC') and r.get('test') for r in s['rows']) else 1)" 2>/dev/null; then echo "exists: $W $XYZ/$DC"; continue; fi
    bash $PKG/stage2_run.sh codec_variants.py --deliver-dir $d --xyz $XYZ --dc $DC --variants $W > $d/codec-variants-fan-${W/\//-}-$DC.log 2>&1 || { echo "FAILED: codec_variants $SCENE lam $L $W $XYZ/$DC"; tail -5 $d/codec-variants-fan-${W/\//-}-$DC.log; nf=$((nf+1)); }
  done
  echo "=== $SCENE lam $L: (d) dictionary re-fits on $MESH ($(date +%H:%M)) ==="
  OUT=$G/kfan-v1; mkdir -p $OUT
  echo "$KRS" | tr '|' '\n' | while read K R; do name=$OUT/k${K}_r${R}
    if [ -f $name/summary.json ]; then echo "exists: k${K}_r${R}"; continue; fi
    rm -rf $name; echo "--- K $K R $R $(date +%H:%M:%S)"
    bash $PKG/deliver_coarse.sh --coarse $MESH --output $name --source $C --dataset $DATA --lab $LAB --images $IMAGES --xyz $XYZ --dc q7 \
      --historical-config none --k $K --r $R --dictionary-rounds 4 > $name.log 2>&1 || { echo "FAILED: deliver_coarse $SCENE lam $L k${K}_r${R}"; tail -8 $name.log; }
    [ -f $name/summary.json ] || echo "FAILED: no summary for $SCENE lam $L k${K}_r${R}"
  done
  python - "$SCENE" "$L" "$d" "$XYZ" "$OUT" "$KRS" "$TAG$SUFFIX" <<'PY' | tee -a $TAB
import json, sys, pathlib
S, L, d, xyz, out, krs, tag = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4], pathlib.Path(sys.argv[5]), sys.argv[6], sys.argv[7]
s = json.load(open(d / 'codec-variants' / 'sweep.json')); rows = {}
for r in s['rows']:
    if r['xyz'] == xyz and r.get('test'): rows[(r['weight'] + '/' + r['index_codec'], r['dc'])] = r
base = rows.get(('const/ctx', 'q7'))
if base is None: print(f"{S} | lam {L} | baseline MISSING | {tag}"); sys.exit(0)
bp = base['test']['psnr']; bm = base['file_MB']
def line(name, mb, t): print(f"{S} | lam {L} | {name:22s} | {mb:7.3f} MB | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f} | dPSNR {t['psnr'] - bp:+.3f} | dMB {mb - bm:+.3f} | {tag}")
line('K4096 R256 q7 (base)', bm, base['test'])
r = rows.get(('const/ctx', 'q8')); line('K4096 R256 q8', r['file_MB'], r['test']) if r else print(f"{S} | lam {L} | K4096 R256 q8: MISSING | {tag}")
for kr in krs.split('|'):
    K, R = kr.split(); p = out / f'k{K}_r{R}' / 'summary.json'
    if p.exists(): j = json.load(open(p)); line(f'K{K} R{R} q7', j['file_MB'], j['delivery_test'])
    else: print(f"{S} | lam {L} | K{K} R{R} q7: MISSING | {tag}")
PY
done
echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB
echo "lambda-k-fan-v3 $SCENE exit 0 ($nf missing / failed)  $(date)"
