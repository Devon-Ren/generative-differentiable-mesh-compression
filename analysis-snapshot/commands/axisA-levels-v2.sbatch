#!/bin/bash
#SBATCH --job-name=axisA
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/zr2245/room-vis-collapse/axisA-levels.%j.log
# axisA-levels-v2.sbatch (2026-10-16): levels 0.7 + 0.6, reduced grid = K1024 x {1,256,1024,4096}, K4096 x {1,256,1024,4096}, K16384 x {4096}
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
export WAIT_FOR=
LEVELS="0.7 0.6" KS="1024 4096" RS="1 256 1024 4096" bash /scratch/zr2245/analysis/axisA-levels-v1.sh; rc1=$?
LEVELS="0.7 0.6" KS="16384" RS="4096" bash /scratch/zr2245/analysis/axisA-levels-v1.sh; rc2=$?
echo "axisA-levels exit $rc1 $rc2"; grep -c '^level' /scratch/zr2245/room-vis-collapse/axisA-levels.txt
