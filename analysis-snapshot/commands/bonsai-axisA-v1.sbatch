#!/bin/bash
#SBATCH --job-name=bonsai-axisA
#SBATCH --account=torch_pr_38_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zr2245/bonsai-vis-collapse/axisA-bonsai.%j.log
# bonsai-axisA-v1.sbatch (2026-10-16), third scene, three checks on the bonsai v0.8 mesh (pipeline default q14/q7, stage2-v2 const/ctx):
#   (1) precision rule: geom_precision.py (extent, median edge, step ratio at q13..q16)  -> expected rule b = ceil(log2(extent / (0.1 * median edge)))
#   (2) precision re-export without training: codec_variants const/ctx at q13 / q14 / q15 x DC q7 on axisB v0.8 (+ q13/q15 on v1.0, v0.7) -> $B/precision-bonsai.txt
#   (3) K x R backbone (k-means rounds 4, 400-step fit): K1024 R1, K4096 R1 / R1024 / R4096, K16384 R4096; K4096 R256 = axisB deliver  -> $B/axisA-bonsai.txt
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
B=/scratch/zr2245/bonsai-vis-collapse; A=/scratch/zr2245/analysis; PKG=$A/vis_collapse-v12
DATA=/scratch/zr2245/datasets/m360/bonsai; IMAGES=images_2; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
C=$B/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; MESH=$B/axisB/v0.8/finetune/best.pt
for f in $MESH $C $PKG/deliver_coarse.sh $PKG/stage2_run.sh $A/geom_precision.py $B/axisB/v0.8/deliver/shared-final.pt; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=== (1) precision rule ==="
python $A/geom_precision.py $B/axisB/v0.8/deliver $B/axisB/v1.0/deliver 2>&1 | tee $B/geom-precision.txt
echo "=== (2) precision re-export ==="
: > $B/precision-bonsai.txt
for L in 0.8 1.0 0.7; do d=$B/axisB/v$L/deliver
  for XYZ in q13 q14 q15; do
    [ "$L" != 0.8 ] && [ "$XYZ" = q14 ] && continue      # q14 at 1.0 / 0.7 is the pipeline delivery itself
    bash $PKG/stage2_run.sh codec_variants.py --deliver-dir $d --xyz $XYZ --dc q7 --variants const/ctx > $d/codec-variants-$XYZ-q7.log 2>&1 || { echo "FAILED re-export v$L $XYZ"; tail -3 $d/codec-variants-$XYZ-q7.log; continue; }
  done
  python - "$d" "$L" >> $B/precision-bonsai.txt <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); L = sys.argv[2]; s = json.load(open(d / 'codec-variants' / 'sweep.json'))
for row in s['rows']:
    if (row['weight'], row['index_codec'], row['dc']) != ('const', 'ctx', 'q7'): continue
    t = row['test'] or {}; fl = (row.get('quantization_geometry') or {}).get('flipped_faces', '?')
    print(f"bonsai v{L} | vertices {s['vertices']} | {row['xyz']}/q7 {row['file_MB']:.3f} MB test {t.get('psnr', float('nan')):.3f} {t.get('ssim', float('nan')):.4f} {t.get('lpips', float('nan')):.4f} | flips {fl}")
PY
done
cat $B/precision-bonsai.txt
echo "=== (3) K x R backbone at v0.8 q14/q7 ==="
OUT=$B/axisA-v0.8; mkdir -p $OUT
for KR in "1024 1" "4096 1" "4096 1024" "4096 4096" "16384 4096"; do set -- $KR; K=$1; R=$2; name=$OUT/k${K}_r${R}
  if [ -f $name/summary.json ]; then echo "exists: k${K}_r${R}"; continue; fi
  rm -rf $name; echo "=== K $K R $R === $(date +%H:%M:%S)"
  bash $PKG/deliver_coarse.sh --coarse $MESH --output $name --source $C --dataset $DATA --lab $LAB --images $IMAGES --xyz q14 --dc q7 \
    --historical-config none --k $K --r $R --dictionary-rounds 4 > $name.log 2>&1 || { echo "FAILED k${K}_r${R}"; tail -8 $name.log; }
done
python - $OUT $B/axisB/v0.8/deliver <<'PY' | tee $B/axisA-bonsai.txt
import json, sys, pathlib
print('level,K,R,MB,test_psnr,test_ssim,test_lpips,vertices,source')
def row(d, K, R, src):
    if not (d / 'summary.json').exists(): print(f'0.8,{K},{R},MISSING'); return
    s = json.load(open(d / 'summary.json')); t = s['delivery_test']
    print(f"0.8,{K},{R},{s['file_MB']:.3f},{t['psnr']:.3f},{t['ssim']:.4f},{t['lpips']:.4f},{s['stages'][-1]['vertices']},{src}")
row(pathlib.Path(sys.argv[2]), 4096, 256, 'axisB-deliver')
for K, R in ((1024, 1), (4096, 1), (4096, 1024), (4096, 4096), (16384, 4096)): row(pathlib.Path(sys.argv[1]) / f'k{K}_r{R}', K, R, 'sweep')
PY
echo "bonsai-axisA exit 0"
