#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=p2b-lego
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/zr2245/analysis/p2b-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/p2b-%x-%j.err
# p2b-lego-v2.sbatch (2026-11-05, VERSION v2 = v1 with the GL-free surf_extract_v2; open3d needs libEGL on the nodes): P2b on lego -- geometry/colour decoupling, no training of the soup.
#   step 1  surf_extract_v2: trained soup (lam 6.2e-4, val-selected) -> median depth (surf_depth, alpha >= 0.5) at the 100 train views -> TSDF (torch) -> marching cubes (skimage)
#           -> largest components -> quadric decimation (fast_simplification) to T in TARGETS -> Taubin -> mesh-T<T>.npz (+ raw)   SURF lines
#   step 2  texture_refit_v2 on the SOUP itself (opaque render reference): vsh3_init / vsh3 / tb4                      TEXREFIT lines
#   step 3  texture_refit_v2 on every mesh-T: vsh3_init vdc vsh3 tb2 tb4 tb8 tb4sh1, and vsh3 after 1 and 2 midpoint subdivisions
# Rows -> $AN/p2b-lego-v2.txt.  Restartable (json "complete": true / surf summary.json are skipped).  ~1-1.5 GPU-h.
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis
SCENE=${SCENE:-lego}; DATA=${DATA:-/scratch/zr2245/datasets/nerf_synthetic/$SCENE}; IMAGES=${IMAGES:-train}
MODEL=${MODEL:-/scratch/zr2245/$SCENE-rd3b4-lam0.00062-mu1.0-best/original}; ITER=${ITER:-30000}
TARGETS=${TARGETS:-"50000 100000 200000"}; RES=${RES:-512}; TAUBIN=${TAUBIN:-5}; STEPS=${STEPS:-3000}
VAR_SOUP=${VAR_SOUP:-"vsh3_init vsh3 tb4"}; VAR_MESH=${VAR_MESH:-"vsh3_init vdc vsh3 tb2 tb4 tb8 tb4sh1"}; SUBS=${SUBS:-"1 2"}
OUT=$AN/p2b-v2/$SCENE; TAB=$AN/p2b-$SCENE-v2.txt; SE=$AN/diag/surf_extract_v2.py; TR=$AN/diag/texture_refit_v2.py; mkdir -p $OUT
echo "p2b-v2 | scene $SCENE | model $MODEL iter $ITER | targets $TARGETS res $RES taubin $TAUBIN | steps $STEPS | job $SLURM_JOB_ID on $(hostname) $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for f in $SE $TR $MODEL/point_cloud/iteration_$ITER/point_cloud_state_dict.pt $DATA/transforms_train.json $DATA/transforms_test.json $LAB/upstream/train.py; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
python -c "import skimage, fast_simplification, scipy; print('skimage', skimage.__version__, 'scipy', scipy.__version__, 'fast_simplification OK')" || { echo "FAILED: scikit-image / fast_simplification not importable (pip install scikit-image fast-simplification on a node with network)"; exit 1; }
python -c "import nvdiffrast; print('nvdiffrast', nvdiffrast.__version__)" || { echo "FAILED: nvdiffrast not importable (see commands/texrefit-v1.md §1)"; exit 1; }
python $SE --selftest | grep -q "selftest PASS" || { echo "FAILED: surf_extract_v2 selftest"; python $SE --selftest; exit 1; }
python $TR --selftest | grep -q "selftest PASS" || { echo "FAILED: texture_refit_v2 selftest"; python $TR --selftest; exit 1; }
run_refit () { # label json mesh_args variants
  local label=$1 j=$2 margs=$3; shift 3
  if [ -s $j ] && grep -q '"complete": true' $j; then echo "exists: $label"; grep -h "^TEXREFIT" ${j%.json}.log; return 0; fi
  python $TR --lab $LAB --model $MODEL --iteration $ITER --dataset $DATA --images $IMAGES --label "$label" --out $j --steps $STEPS $margs --variants "$@" > ${j%.json}.log 2>&1 \
    || { echo "FAILED: refit $label"; grep -n -m 5 "FAILED\|Error\|Traceback" ${j%.json}.log; tail -8 ${j%.json}.log; return 1; }
  grep -h "^TEXREFIT" ${j%.json}.log | tee -a $TAB
}
echo "=== step 1: surface extraction ($(date +%H:%M)) ==="
if [ -f $OUT/surf/summary.json ]; then echo "exists: $OUT/surf"; grep -h "^SURF" $OUT/surf.log; else
  python $SE --lab $LAB --dataset $DATA --model $MODEL --iteration $ITER --images $IMAGES --out $OUT/surf --targets $TARGETS --res $RES --taubin $TAUBIN > $OUT/surf.log 2>&1 \
    || { echo "FAILED: surf_extract"; grep -n -m 5 "FAILED\|Error\|Traceback" $OUT/surf.log; tail -12 $OUT/surf.log; exit 1; }
  grep -h "camera convention\|first view\|TSDF integration" $OUT/surf.log; grep -h "^SURF" $OUT/surf.log | tee -a $TAB
fi
for T in $TARGETS; do [ -s $OUT/surf/mesh-T$T.npz ] || { echo "FAILED: $OUT/surf/mesh-T$T.npz missing/empty"; exit 1; }; done
echo "=== step 2: soup reference under the opaque rasterizer ($(date +%H:%M)) ==="
run_refit "$SCENE soup" $OUT/soup.json "" $VAR_SOUP || exit 1
echo "=== step 3: external meshes ($(date +%H:%M)) ==="
nf=0
for T in $TARGETS; do
  run_refit "$SCENE T$T" $OUT/T$T.json "--mesh $OUT/surf/mesh-T$T.npz" $VAR_MESH || nf=$((nf+1))
  for S in $SUBS; do run_refit "$SCENE T$T sub$S" $OUT/T$T-sub$S.json "--mesh $OUT/surf/mesh-T$T.npz --subdivide $S" vsh3 || nf=$((nf+1)); done
done
echo "===== $TAB ====="; cat $TAB
[ $nf = 0 ] && echo "p2b-lego-v2 exit 0  $(date)" || { echo "p2b-lego-v2 FINISHED WITH $nf FAILED refits  $(date)"; exit 1; }
