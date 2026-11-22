# f3v1 environment (sourced by the run scripts; no 'set -e' anywhere)
export MM=/scratch/zr2245/memory-medium
export F3=$MM/f3v1
export DATA=$MM/data
export OUT=$MM/out
export HF_HOME=/scratch/zr2245/hf-cache
export DA3_ENV=/scratch/zr2245/conda-envs/da3
mkdir -p $OUT
main_env() {
  source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
  conda activate /scratch/zr2245/conda-envs/pmsh-cu126
  source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
}
da3_env() {
  source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
  conda activate $DA3_ENV
}
fail() { echo "FAILED: $1"; [ -n "$2" ] && [ -f "$2" ] && { echo "---- last lines of $2 ----"; tail -n 30 "$2"; }; }
