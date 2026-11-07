#!/usr/bin/env bash
# run_texrefit_v1.sh (2026-10-30): probe P2 (diag/texture_refit_v1.py) on the frozen v3 meshes lam 6.2e-3 and lam 1.85e-3
# of each scene in SCENES (default "room kitchen"); per-vertex plane labels from P1b are passed when present.
# Restartable: a checkpoint whose json says "complete" is skipped.  Rows -> $AN/texrefit-v1.txt (TEXREFIT ...).
# Run on a GPU node:  bash run_texrefit_v1.sh      (env: SCENES, STEPS, VARIANTS, LAMS)
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; OUT=$AN/texrefit-v1; TAB=$AN/texrefit-v1.txt
PY=$AN/diag/texture_refit_v1.py; M360=/scratch/zr2245/datasets/m360; mkdir -p $OUT
SCENES=${SCENES:-"room kitchen"}; LAMS=${LAMS:-"0.0062 0.00185"}; STEPS=${STEPS:-3000}
VARIANTS=${VARIANTS:-"vsh3_init vsh3 vdc frgb tb2 tb4 tb4sh1"}
images_of () { case $1 in room|kitchen|counter|bonsai) echo images_2;; *) echo images_4;; esac; }
n_ok=0; n_fail=0; n_skip=0
for S in $SCENES; do
  for LAM in $LAMS; do
    T=v3-lam$LAM; MODEL=/scratch/zr2245/$S-rd3-lam$LAM-mu1.0/original
    J=$OUT/$S-$T.json; L=$OUT/$S-$T.log; PL=$AN/planarity-v1/$S-$T-plane-labels.npy
    if [ ! -f $MODEL/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then echo "missing: $S $T ($MODEL)"; n_skip=$((n_skip+1)); continue; fi
    if [ -s $J ] && grep -q '"complete": true' $J; then echo "exists: $S $T"; n_skip=$((n_skip+1)); continue; fi
    PLARG=""; [ -f $PL ] && PLARG="--plane_labels $PL" || echo "no plane labels for $S $T (planar split skipped)"
    echo "=== $S $T ($(date +%H:%M:%S)) ==="
    if python $PY --lab $LAB --model $MODEL --dataset $M360/$S --images $(images_of $S) --label "$S $T" --out $J --steps $STEPS --variants $VARIANTS $PLARG > $L 2>&1; then
      grep -h "^TEXREFIT" $L | tee -a $TAB; n_ok=$((n_ok+1))
    else echo "FAILED: $S $T"; tail -n 15 $L; n_fail=$((n_fail+1)); fi
  done
done
echo "===== $TAB ====="; cat $TAB 2>/dev/null
echo "run_texrefit_v1 done: ok $n_ok failed $n_fail skipped $n_skip  $(date)"
