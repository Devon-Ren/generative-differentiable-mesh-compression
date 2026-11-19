#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=relsmoke
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/zr2245/analysis/relsmoke-%j.out
# smoke-test-release-v1.sbatch (2026-11-19): runs the code supplement (ByteMesh-code.zip) end to end on garden at 10 MB, from a fresh unzip.
# NOT part of the zip.  Expected, from the paper run of the same setting: file 9.997 MB, test PSNR 23.67 dB (seed variation: 0.03 dB, 0.4 % in size).
# Usage: sbatch /scratch/zr2245/analysis/smoke-test-release-v1.sbatch      (ByteMesh-code.zip must be in /scratch/zr2245/analysis)
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
ENV_PATH=/scratch/zr2245/conda-envs/pmsh-cu126; export CUDA_HOME=$ENV_PATH; export PATH=$ENV_PATH/bin:$PATH
export LD_LIBRARY_PATH="$ENV_PATH/targets/x86_64-linux/lib:$ENV_PATH/lib:$ENV_PATH/lib64:${LD_LIBRARY_PATH:-}"; export OMP_NUM_THREADS=8
AN=/scratch/zr2245/analysis; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; DATA=/scratch/zr2245/datasets/m360/garden
T=/scratch/zr2245/release-smoke-$(date +%m%d-%H%M%S); mkdir -p $T && unzip -q $AN/ByteMesh-code.zip -d $T || { echo "FAILED: unzip"; exit 1; }
cd $T/ByteMesh-code && echo "release at $T/ByteMesh-code" && bash build.sh > $T/build.log 2>&1 || { echo "FAILED: build"; tail -20 $T/build.log; exit 1; }
bash run_pipeline.sh $LAB $DATA images_4 10 q15 $T/garden-10mb 0 > $T/run.log 2>&1 || { echo "FAILED: pipeline"; tail -30 $T/run.log; exit 1; }
tail -3 $T/run.log
python - "$T/garden-10mb" <<'PY'
import json, sys, glob
d = json.load(open(sys.argv[1] + '/mesh/deliver/summary.json')); t = d['delivery_test']
print('RELSMOKE garden target 10 MB | file %.3f MB | test PSNR %.3f SSIM %.4f LPIPS %.4f | paper run: 9.997 MB, 23.672 dB' % (d['file_MB'], t['psnr'], t['ssim'], t['lpips']))
PY
echo "relsmoke exit 0  $(date)"
