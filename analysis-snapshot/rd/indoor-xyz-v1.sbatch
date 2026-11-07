#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=indoor-xyz
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/zr2245/analysis/indoor-xyz-%j.out
#SBATCH --error=/scratch/zr2245/analysis/indoor-xyz-%j.err
# indoor-xyz-v1.sbatch (2026-11-05, VERSION v1): XYZ precision re-export on the --indoor level-1.0 room delivery.
# indoor-diag-v1-stages showed room quant+coding = -0.35 dB (val) of which DC q8 explains ~0.08 -> the rest is the XYZ q14 grid.
# Re-exports const/ctx q15 / q16 with DC q7 into the same deliver dir (seconds + one test render each, no training) and prints
# rows next to the existing q14 baseline: scene | xyz | MB | test PSNR SSIM LPIPS | dPSNR | dMB. Restartable (existing sweep rows skipped).
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENE=${SCENE:-room}; BASE=${BASE:-q14}; XYZS=${XYZS:-"q15 q16"}; DC=q7; W=const/ctx
AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; TAB=$AN/indoor-xyz-v1.txt
d=/scratch/zr2245/$SCENE-indoor/axisB/v1.0/deliver
echo "indoor-xyz-v1 | scene $SCENE base $BASE xyz $XYZS dc $DC | job $SLURM_JOB_ID on $(hostname) $(date)"
for f in $PKG/codec_variants.py $PKG/stage2_run.sh $d/shared-final.pt $d/codec-variants/sweep.json; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
nf=0
for X in $XYZS; do
  if python -c "import json,sys; s=json.load(open('$d/codec-variants/sweep.json')); sys.exit(0 if any((r['weight']+'/'+r['index_codec'],r['xyz'],r['dc'])==('$W','$X','$DC') and r.get('test') for r in s['rows']) else 1)" 2>/dev/null; then echo "exists: $W $X/$DC"; continue; fi
  echo "--- $SCENE $W $X/$DC $(date +%H:%M:%S)"
  bash $PKG/stage2_run.sh codec_variants.py --deliver-dir $d --xyz $X --dc $DC --variants $W > $d/codec-variants-xyz-$X-$DC.log 2>&1 || { echo "FAILED: codec_variants $SCENE $X/$DC"; tail -5 $d/codec-variants-xyz-$X-$DC.log; nf=$((nf+1)); }
done
python - "$SCENE" "$d" "$BASE" "$XYZS" "$DC" "$W" <<'PY' | tee -a $TAB
import json, sys, pathlib
S, d, base, xyzs, dc, w = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4].split(), sys.argv[5], sys.argv[6]
rows = {(r['xyz'], r['dc']): r for r in json.load(open(d / 'codec-variants' / 'sweep.json'))['rows'] if r['weight'] + '/' + r['index_codec'] == w and r.get('test')}
b = rows.get((base, dc))
if b is None: print(f"{S} | baseline {base}/{dc} MISSING"); sys.exit(0)
def line(x, r): t = r['test']; print(f"{S} | xyz {x} dc {dc} | {r['file_MB']:7.3f} MB | test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f} | dPSNR {t['psnr'] - b['test']['psnr']:+.3f} | dMB {r['file_MB'] - b['file_MB']:+.3f}")
line(base + ' (base)', b)
for x in xyzs:
    r = rows.get((x, dc)); line(x, r) if r else print(f"{S} | xyz {x} dc {dc}: MISSING")
PY
[ $nf = 0 ] && echo "indoor-xyz-v1 $SCENE exit 0  $(date)" || { echo "indoor-xyz-v1 $SCENE FINISHED WITH $nf FAILED items  $(date)"; exit 1; }
