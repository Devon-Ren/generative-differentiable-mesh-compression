#!/bin/bash
#SBATCH --job-name=axisA
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/zr2245/room-vis-collapse/axisA-levels.%j.log
# one axis-B level per job: sbatch --export=ALL,LEVELS=0.9,WAIT_FOR= axisA-levels-v1.sbatch
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
echo "levels: $LEVELS"; nvidia-smi --query-gpu=name,memory.total --format=csv
bash /scratch/zr2245/analysis/axisA-levels-v1.sh
rc=$?; echo "axisA-levels exit $rc"; exit $rc
