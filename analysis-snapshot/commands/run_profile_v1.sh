#!/bin/bash
# run_profile_v1.sh (2026-10-31): where does one training iteration spend its time on the L40S?
# Usage (on the GPU node):  bash $AN/commands/run_profile_v1.sh [scene] [lam]      default: room 0.00185
# Writes $AN/profile-v1/<scene>-lam<lam>.{log,json}; prints the PROFILE lines and a build-arch check.
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; M360=/scratch/zr2245/datasets/m360
S=${1:-room}; LAM=${2:-0.00185}; OUT=$AN/profile-v1; mkdir -p $OUT
MODEL=/scratch/zr2245/$S-rd3-lam$LAM-mu1.0/original
images_of () { case $1 in room|kitchen|counter|bonsai) echo images_2;; *) echo images_4;; esac; }
CK=$MODEL/point_cloud/iteration_30000/point_cloud_state_dict.pt
if [ ! -f $CK ]; then echo "FAILED: no checkpoint $CK"; exit 0; fi
if ! nvidia-smi -L > /dev/null 2>&1; then echo "FAILED: no GPU on this node"; exit 0; fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
echo "=== build check: which SM architectures are compiled into the CUDA extensions (L40S = sm_89) ==="
for M in diff_triangle_rasterization._C fused_ssim_cuda; do
  SO=$(python -c "import importlib; m=importlib.import_module('$M'); print(m.__file__)" 2>/dev/null)
  if [ -z "$SO" ]; then echo "$M: import failed"; continue; fi
  if command -v cuobjdump > /dev/null; then
    echo "$M: $(cuobjdump --list-elf $SO 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ') ptx: $(cuobjdump --list-ptx $SO 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
  else
    echo "$M: $(strings $SO | grep -o 'sm_[0-9][0-9]' | sort -u | tr '\n' ' ') (strings fallback; cuobjdump not on PATH)"
  fi
done
echo "=== profile $S lam $LAM ($(date +%H:%M:%S)) ==="
L=$OUT/$S-lam$LAM.log; J=$OUT/$S-lam$LAM.json
if python $AN/diag/profile_train_v1.py --lab $LAB --model $MODEL --dataset $M360/$S --images $(images_of $S) --out $J > $L 2>&1; then
  grep "^PROFILE" $L
  echo "kernel tables: grep -A 18 'top CUDA kernels' $L"
else
  echo "FAILED: profile_train_v1 ($S lam $LAM)"; tail -25 $L
fi
echo "run_profile_v1 done ($(date +%H:%M:%S)); log $L"
