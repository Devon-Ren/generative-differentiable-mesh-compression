#!/bin/bash
#SBATCH --job-name=garden-axisA
#SBATCH --account=torch_pr_38_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/zr2245/garden-vis-collapse/axisA-garden.%j.log
# garden-axisA-v1.sbatch (2026-10-16): K x R backbone on the garden v0.8 mesh (q15/q7, stage2-v2 const/ctx, k-means rounds 4 = same recipe as Room).
# Points: K1024 R1, K4096 R1, K4096 R4096, K16384 R4096.  Already measured and reused in the table: K4096 R256 (garden default,
# 27.339 / 24.371) and K4096 R1024 (crossbank own, 28.056 / 24.393).  Table: $G/axisA-garden.txt
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
G=/scratch/zr2245/garden-vis-collapse; A=/scratch/zr2245/analysis; PKG=$A/vis_collapse-v12
DATA=/scratch/zr2245/datasets/m360/garden; IMAGES=images_4; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; MESH=$G/axisB/v0.8/finetune/best.pt
for f in $MESH $C $PKG/deliver_coarse.sh; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
nvidia-smi --query-gpu=name,memory.total --format=csv
OUT=$G/axisA-v0.8; mkdir -p $OUT
for KR in "1024 1" "4096 1" "4096 4096" "16384 4096"; do set -- $KR; K=$1; R=$2; name=$OUT/k${K}_r${R}
  if [ -f $name/summary.json ]; then echo "exists: k${K}_r${R}"; continue; fi
  rm -rf $name; echo "=== K $K R $R === $(date +%H:%M:%S)"
  bash $PKG/deliver_coarse.sh --coarse $MESH --output $name --source $C --dataset $DATA --lab $LAB --images $IMAGES --xyz q15 --dc q7 \
    --historical-config none --k $K --r $R --dictionary-rounds 4 > $name.log 2>&1 || { echo "FAILED k${K}_r${R}"; tail -8 $name.log; }
done
python - $OUT <<'PY' | tee $G/axisA-garden.txt
import json, sys, pathlib
print('level,K,R,MB,test_psnr,test_ssim,test_lpips,vertices,source')
print('0.8,4096,256,27.339,24.371,0.7288,0.2981,2537590,precision-sweep')
print('0.8,4096,1024,28.056,24.393,0.7302,0.2970,2537590,crossbank-own')
for K, R in ((1024, 1), (4096, 1), (4096, 4096), (16384, 4096)):
    d = pathlib.Path(sys.argv[1]) / f'k{K}_r{R}'
    if not (d / 'summary.json').exists(): print(f'0.8,{K},{R},MISSING'); continue
    s = json.load(open(d / 'summary.json')); t = s['delivery_test']
    print(f"0.8,{K},{R},{s['file_MB']:.3f},{t['psnr']:.3f},{t['ssim']:.4f},{t['lpips']:.4f},{s['stages'][-1]['vertices']},sweep")
PY
echo "garden-axisA exit 0"
