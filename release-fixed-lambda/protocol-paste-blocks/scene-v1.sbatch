#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=scene-vis
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/zr2245/analysis/scene-%j.out
#SBATCH --error=/scratch/zr2245/analysis/scene-%j.err
# scene-v1.sbatch (2026-10-16): garden-v3.sbatch made scene-generic.  SCENE picks the MipNeRF360 folder and the output dir
# (G=/scratch/zr2245/$SCENE-vis-collapse); IMAGES defaults to images_2 (indoor, as Room) -- pass images_4 for outdoor scenes.
# Pipeline = vis_collapse-v12/scripts/run_garden.sh (train 30k -> footprint -> 1.0 / 0.8 / 0.7 collapse + finetune + stage2-v2 delivery q14/q7).
# Restartable.  Memory 64G because coarse_vis.py needs >= 48 GB RAM; general account (long job).  Submit:
#     sbatch --export=ALL,SCENE=bonsai,IMAGES=images_2 --job-name=bonsai-vis /scratch/zr2245/analysis/scene-v1.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
PKG=/scratch/zr2245/analysis/vis_collapse-v12
SCENE=${SCENE:?set SCENE=bonsai (etc.) via --export}
export G=${G:-/scratch/zr2245/$SCENE-vis-collapse}
export DATA=${DATA:-/scratch/zr2245/datasets/m360/$SCENE}
export IMAGES=${IMAGES:-images_2}
echo "scene $SCENE | G $G | DATA $DATA | IMAGES $IMAGES"
mkdir -p $G; echo "job $SLURM_JOB_ID on $(hostname) start $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f $PKG/scripts/run_garden.sh ] || { echo "vis_collapse-v12 not installed at $PKG"; exit 1; }
[ -f /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION ] || { echo "stage2-v2 not installed (run stage2-v2-patch-v1/apply_stage2_v2.sh first)"; exit 1; }
ls $PKG/native/_native_qem_vis*.so >/dev/null 2>&1 || { echo "native module missing in $PKG/native (cp from vis_collapse-v11/native)"; exit 1; }
[ -d $DATA/$IMAGES ] || { echo "dataset not found: $DATA/$IMAGES"; exit 1; }
[ -f $DATA/sparse/0/cameras.bin ] || { echo "COLMAP sparse model not found: $DATA/sparse/0/cameras.bin"; exit 1; }
NORMALS=$DATA/${IMAGES/images/normals}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
n_img=$(ls $DATA/$IMAGES | grep -ci '\.jpe\?g$'); n_nrm=$(ls $NORMALS 2>/dev/null | grep -ci '\.png$')
echo "images: $n_img in $DATA/$IMAGES | normals: $n_nrm in $NORMALS"
if [ "$n_nrm" -lt "$n_img" ]; then
  echo "=== stage -1: Metric3D normals -> $NORMALS (same protocol as Room's normals_2) ==="
  (cd $LAB/upstream && python extract_normals.py -s $DATA -i $IMAGES > $G.normals.log 2>&1) || { echo "NORMALS FAILED: real cause = first error below (full log: $G.normals.log)"; grep -n -m 6 -i 'error\|No module' $G.normals.log; exit 1; }
  n_nrm=$(ls $NORMALS | grep -ci '\.png$'); [ "$n_nrm" -ge "$n_img" ] || { echo "normals incomplete: $n_nrm / $n_img"; exit 1; }
fi
head -1 $PKG/VERSION; cat /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION
bash $PKG/scripts/run_garden.sh 2>&1 | tee -a $G.log   # script name is historical; everything is driven by G / DATA / IMAGES
rc=${PIPESTATUS[0]}
echo "run_garden.sh exit $rc  end $(date)"
echo "===== $G/garden.txt ($SCENE) ====="; cat $G/garden.txt 2>/dev/null || echo "(no table: check $G/*.log and $G/axisB/*.log)"
exit $rc
