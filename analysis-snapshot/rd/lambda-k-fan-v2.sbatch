#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=lkfan2
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/lkfan-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/lkfan-%x-%j.err
# lambda-k-fan-v2.sbatch (2026-11-05, VERSION v2): = v1 + DATA override via --export (T&T: DATA=/scratch/zr2245/datasets/tandt/$SCENE, IMAGES=images)
#   + the delivered asset's XYZ (config.json) must equal xyz_of SCENE, else FAIL before any GPU work.  Same table file as v1 ($AN/lambda-k-fan-v1.txt).
# lambda-k-fan-v1.sbatch (2026-11-05, VERSION v1): Yao Q1/Q2 -- the joint lambda x dictionary-size RD fan, WITHOUT training.
# For one SCENE and every lambda asset $G = /scratch/zr2245/$SCENE-$TAG-lam$L-mu1.0 (level-1.0 protocol already run: axisB/v1.0/{finetune/best.pt, deliver/}):
#   (a) DC q8 instead of q7 on the delivered K4096 R256 asset      codec_variants const/ctx --dc q8   (seconds + test render)
#   (d) dictionary K x R in KRS instead of K4096 R256                deliver_coarse.sh on the same finetuned mesh (k-means 4 rounds + 400-step fit, 3-15 min each)
# Rows -> $AN/lambda-k-fan-v1.txt: scene | lam | lever | MB | test PSNR SSIM LPIPS | dPSNR / dMB vs that lambda's baseline (const/ctx q7 K4096R256).
# Restartable: existing sweep rows / summary.json are skipped.  48G: the dictionary fit holds a 2-3 M-vertex model.
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:?set SCENE via --export}; TAG=${TAG:?set TAG=rd3i (indoor rerun) or rd3 (outdoor) via --export}; IMAGES=${IMAGES:-images_2}
LAMS=${LAMS:-"0.00062 0.00185 0.0062 0.0185"}; KRS=${KRS:-"4096 1|1024 256|4096 1024|16384 1024"}
AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; TAB=$AN/lambda-k-fan-v1.txt
DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}
xyz_of () { case $1 in room) echo q14;; garden|bonsai|counter|flowers) echo q15;; kitchen|bicycle|stump|treehill) echo q16;; truck|train) echo q18;; *) echo "FAILED: no XYZ recorded for $1" >&2; return 1;; esac; }
XYZ=$(xyz_of $SCENE) || exit 1
echo "lambda-k-fan-v2 | scene $SCENE tag $TAG | DATA $DATA/$IMAGES | lams $LAMS | K x R $KRS | XYZ $XYZ | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $PKG/codec_variants.py $PKG/deliver_coarse.sh $PKG/stage2_run.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $DATA/$IMAGES/.; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
nf=0
for L in $LAMS; do
  G=/scratch/zr2245/$SCENE-$TAG-lam$L-mu1.0; d=$G/axisB/v1.0/deliver; MESH=$G/axisB/v1.0/finetune/best.pt; C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  miss=0; for f in $d/shared-final.pt $d/config.json $d/summary.json $MESH $C; do [ -e $f ] || { echo "FAILED: MISSING $f"; miss=1; }; done
  [ $miss = 0 ] || { nf=$((nf+1)); continue; }
  python -c "import json; c=json.load(open('$d/config.json')); k=c.get('k',4096); r=c.get('r',256); assert (k,r)==(4096,256), ('baseline delivery is not K4096 R256', k, r); assert c.get('xyz')=='$XYZ', ('delivered xyz', c.get('xyz'), 'differs from xyz_of', '$XYZ')" || { echo "FAILED: $SCENE lam $L baseline is not K4096 R256 at XYZ $XYZ (see assert above)"; nf=$((nf+1)); continue; }
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
  python - "$SCENE" "$L" "$d" "$XYZ" "$OUT" "$KRS" <<'PY' | tee -a $TAB
import json, sys, pathlib
S, L, d, xyz, out, krs = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3]), sys.argv[4], pathlib.Path(sys.argv[5]), sys.argv[6]
s = json.load(open(d / 'codec-variants' / 'sweep.json')); rows = {}
for r in s['rows']:
    if r['xyz'] == xyz and r.get('test'): rows[(r['weight'] + '/' + r['index_codec'], r['dc'])] = r
base = rows.get(('const/ctx', 'q7'))
if base is None: print(f"{S} | lam {L} | baseline MISSING"); sys.exit(0)
bp = base['test']['psnr']; bm = base['file_MB']
def line(name, mb, t): print(f"{S} | lam {L} | {name:22s} | {mb:7.3f} MB | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f} | dPSNR {t['psnr'] - bp:+.3f} | dMB {mb - bm:+.3f}")
line('K4096 R256 q7 (base)', bm, base['test'])
r = rows.get(('const/ctx', 'q8')); line('K4096 R256 q8', r['file_MB'], r['test']) if r else print(f"{S} | lam {L} | K4096 R256 q8: MISSING")
for kr in krs.split('|'):
    K, R = kr.split(); p = out / f'k{K}_r{R}' / 'summary.json'
    if p.exists(): j = json.load(open(p)); line(f'K{K} R{R} q7', j['file_MB'], j['delivery_test'])
    else: print(f"{S} | lam {L} | K{K} R{R} q7: MISSING")
PY
done
echo "===== $TAB ($SCENE rows) ====="; grep "^$SCENE " $TAB
[ $nf = 0 ] && echo "lambda-k-fan-v2 $SCENE exit 0  $(date)" || { echo "lambda-k-fan-v2 $SCENE FINISHED WITH $nf FAILED items  $(date)"; exit 1; }
