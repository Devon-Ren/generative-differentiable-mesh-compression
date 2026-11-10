#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=genc22
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zr2245/analysis/genc22-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/genc22-%x-%j.err
# generic-codecs-b22.sbatch (2026-11-10, VERSION genc-b22 = genc-b21 + RUN/DATA/IMAGES/QP exported for the 9 Mip-360 / T&T scenes the Draco runner does not know (its own table has room and garden only; the o3dgc runner knows all 11), same originals as the o3dgc rows ($SCENE-vis-collapse); b21 text follows): the generic-codec baselines (Draco / zstd / texture route: generic-v1/run_generic_v1.sh,
# and MPEG-4 SC3DMC-TFAN: o3dgc-v1/run_o3dgc_v3.sh) on the ORIGINAL mesh of every scene in SCENES, one after another, in one job.  So far only
# room and garden have rows (generic-codecs.txt, o3dgc rows).  Both runners are restartable and non-fatal per row; a scene whose runner fails
# prints FAILED and the job moves to the next scene.  Deep Blending scenes use the --indoor originals ($SCENE-indoor) through the runners'
# generic branch (RUN / DATA / IMAGES / QP exported).  TEX="4096" (one texture resolution: the xatlas bake is the slow part).
#   sbatch --export=ALL,SCENES="kitchen counter bonsai drjohnson playroom" --job-name=b21-genc-a .../generic-codecs-b21.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
SCENES=${SCENES:?set SCENES="kitchen counter ..." via --export}; export TEX=${TEX:-"4096"}
AN=/scratch/zr2245/analysis; G=$AN/generic-v1; O=$AN/o3dgc-v1; D=/scratch/zr2245/datasets
for f in $G/run_generic_v1.sh $G/generic_codec_v1.py $G/eval_ckpt_v1.py $O/run_o3dgc_v3.sh $O/o3dgc_codec_v2.py; do [ -f $f ] || { echo "FAILED: MISSING $f (generic-codecs-v1 / o3dgc-v3 blocks were never installed?)"; exit 1; }; done
echo "generic-codecs-b21 | scenes $SCENES | TEX '$TEX' | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nf=0
for S in $SCENES; do
  unset RUN DATA IMAGES QP
  case $S in
    drjohnson|playroom) export RUN=/scratch/zr2245/$S-indoor/original DATA=$D/db/$S IMAGES=images QP=15;;
    room|garden) ;;   # known to both runners' case tables
    kitchen) export RUN=/scratch/zr2245/$S-vis-collapse/original DATA=$D/m360/$S IMAGES=images_2 QP=16;;
    counter|bonsai) export RUN=/scratch/zr2245/$S-vis-collapse/original DATA=$D/m360/$S IMAGES=images_2 QP=15;;
    flowers) export RUN=/scratch/zr2245/$S-vis-collapse/original DATA=$D/m360/$S IMAGES=images_4 QP=15;;
    bicycle|stump|treehill) export RUN=/scratch/zr2245/$S-vis-collapse/original DATA=$D/m360/$S IMAGES=images_4 QP=16;;
    truck|train) export RUN=/scratch/zr2245/$S-vis-collapse/original DATA=$D/tandt/$S IMAGES=images QP=18;;
    *) echo "FAILED: no settings for scene $S"; nf=$((nf+1)); continue;;
  esac
  if [ -n "$RUN" ] && [ ! -f $RUN/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then echo "FAILED: $S original missing at $RUN"; nf=$((nf+1)); continue; fi
  echo "=== $S: generic-v1 (zstd / Draco / texture) $(date) ==="
  bash $G/run_generic_v1.sh $S > $G/$S-b22.log 2>&1 || { echo "FAILED: run_generic_v1 $S"; tail -12 $G/$S-b22.log; nf=$((nf+1)); }
  grep -h "^$S \|FAILED\|WARNING" $G/$S-b22.log | tail -14
  echo "=== $S: o3dgc-v3 (MPEG SC3DMC / TFAN) $(date) ==="
  EVAL=1 bash $O/run_o3dgc_v3.sh $S > $O/$S-b21.log 2>&1 || { echo "FAILED: run_o3dgc_v3 $S"; tail -12 $O/$S-b21.log; nf=$((nf+1)); }
  grep -ih "ROW\|FAILED\|done" $O/$S-b21.log | tail -8
done
echo "===== generic-codecs.txt (these scenes) ====="; for S in $SCENES; do grep -h "^$S " $AN/generic-codecs.txt 2>/dev/null; done
echo "generic-codecs-b22 exit 0 ($nf failures)  $(date)"
