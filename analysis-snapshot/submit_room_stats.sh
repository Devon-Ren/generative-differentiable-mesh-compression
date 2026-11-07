#!/usr/bin/env bash
# CPU-only redundancy statistics for Room (no GPU requested). Read-only on all inputs.
# Usage:  sbatch submit_room_stats.sh            # crop Delaunay (fast, upward-biased hit rate)
#         sbatch --mem=128G --time=04:00:00 submit_room_stats.sh full
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --job-name=room-redundancy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=room-redundancy-%j.out
#SBATCH --error=room-redundancy-%j.err
set -eo pipefail
MODE=${1:-crop}
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
python -c 'import numpy, scipy, torch, zstandard' || { echo "missing numpy/scipy/torch/zstandard"; exit 1; }

RUN=/scratch/zr2245/room-fixed-sharing-0N4OSU/run
OUT=/scratch/zr2245/room-redundancy-stats
mkdir -p "$OUT"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
python redundancy_stats.py \
  --ckpt "$RUN/original/point_cloud/iteration_30000/point_cloud_state_dict.pt" \
  --shared "$RUN/shared/residual_k4096_r1024_f1/asset.mslb" \
  --delaunay "$MODE" \
  --out "$OUT/room-redundancy-$MODE.json"
echo "RESULT: $OUT/room-redundancy-$MODE.json"
