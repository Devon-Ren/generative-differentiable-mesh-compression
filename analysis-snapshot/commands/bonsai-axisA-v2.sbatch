#!/bin/bash
#SBATCH --job-name=bonsai-axisA2
#SBATCH --account=torch_pr_38_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/bonsai-vis-collapse/axisA2-bonsai.%j.log
# bonsai-axisA-v2.sbatch (2026-10-17): K x R backbone on the bonsai 1.0 and 0.7 meshes (q14/q7 const/ctx, k-means rounds 4),
# so the greedy allocator can be checked on a scene where the vertex lever is NOT free.  0.8 level comes from bonsai-axisA-v1.
# Points per level: K1024 R1, K4096 R1 / R1024 / R4096, K16384 R4096 (K4096 R256 = the axisB deliver).  Table: $B/axisA-bonsai-levels.txt
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
B=/scratch/zr2245/bonsai-vis-collapse; A=/scratch/zr2245/analysis; PKG=$A/vis_collapse-v12
DATA=/scratch/zr2245/datasets/m360/bonsai; IMAGES=images_2; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
C=$B/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
for f in $C $PKG/deliver_coarse.sh $B/axisB/v1.0/finetune/best.pt $B/axisB/v0.7/finetune/best.pt; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
nvidia-smi --query-gpu=name,memory.total --format=csv
for L in 1.0 0.7; do MESH=$B/axisB/v$L/finetune/best.pt; OUT=$B/axisA-v$L; mkdir -p $OUT
  for KR in "1024 1" "4096 1" "4096 1024" "4096 4096" "16384 4096"; do set -- $KR; K=$1; R=$2; name=$OUT/k${K}_r${R}
    if [ -f $name/summary.json ]; then echo "exists: v$L k${K}_r${R}"; continue; fi
    rm -rf $name; echo "=== v$L K $K R $R === $(date +%H:%M:%S)"
    bash $PKG/deliver_coarse.sh --coarse $MESH --output $name --source $C --dataset $DATA --lab $LAB --images $IMAGES --xyz q14 --dc q7 \
      --historical-config none --k $K --r $R --dictionary-rounds 4 > $name.log 2>&1 || { echo "FAILED v$L k${K}_r${R}"; tail -8 $name.log; }
  done
done
python - $B <<'PY' | tee $B/axisA-bonsai-levels.txt
import json, sys, pathlib
B = pathlib.Path(sys.argv[1]); print('level,K,R,MB,test_psnr,test_ssim,test_lpips,vertices,source')
def row(L, d, K, R, src):
    if not (d / 'summary.json').exists(): print(f'{L},{K},{R},MISSING'); return
    s = json.load(open(d / 'summary.json')); t = s['delivery_test']
    print(f"{L},{K},{R},{s['file_MB']:.3f},{t['psnr']:.3f},{t['ssim']:.4f},{t['lpips']:.4f},{s['stages'][-1]['vertices']},{src}")
for L in ('1.0', '0.8', '0.7'):
    row(L, B / 'axisB' / f'v{L}' / 'deliver', 4096, 256, 'axisB-deliver')
    for K, R in ((1024, 1), (4096, 1), (4096, 1024), (4096, 4096), (16384, 4096)): row(L, B / f'axisA-v{L}' / f'k{K}_r{R}', K, R, 'sweep')
PY
echo "bonsai-axisA2 exit 0"
