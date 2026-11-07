#!/bin/bash
#SBATCH --job-name=hull-v2
#SBATCH --account=torch_pr_38_tandon_advanced
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zr2245/room-vis-collapse/hull-v2.slurm.%j.log
# hull-v2-v1.sbatch (2026-10-16): re-export every measured Room RD point (axis B v1.0..v0.5, geo-only, axis A 12 K x R dirs)
# in the FINAL stage2-v2 format (const vertex_weight + ctx indices, q14/q7) and measure decoded test. No training.
# Output: /scratch/zr2245/room-vis-collapse/hull-v2.txt  (send that file back).
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
S=/scratch/zr2245/analysis/hull-v2-reexport-v1.sh
[ -f "$S" ] || { echo "MISSING $S (upload analysis/commands/hull-v2-reexport-v1.sh first)"; exit 2; }
[ -f /scratch/zr2245/analysis/vis_collapse-v12/codec_variants.py ] || { echo "MISSING vis_collapse-v12/codec_variants.py"; exit 2; }
nvidia-smi --query-gpu=name,memory.total --format=csv
bash "$S"
rc=$?; echo "hull-v2 exit $rc"; grep -c '^' /scratch/zr2245/room-vis-collapse/hull-v2.txt 2>/dev/null; exit $rc
