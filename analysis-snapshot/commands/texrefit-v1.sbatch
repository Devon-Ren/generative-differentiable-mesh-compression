#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=texrefit1
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zr2245/analysis/texrefit1-%j.out
#SBATCH --error=/scratch/zr2245/analysis/texrefit1-%j.err
# texrefit-v1.sbatch (2026-10-30): probe P2 on room + kitchen (lam 6.2e-3 and 1.85e-3), ~1 GPU-h.  Same runner as the
# interactive path: bash /scratch/zr2245/analysis/commands/run_texrefit_v1.sh
echo "job $SLURM_JOB_ID on $(hostname) $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash /scratch/zr2245/analysis/commands/run_texrefit_v1.sh
