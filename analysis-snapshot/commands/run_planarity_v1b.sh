#!/usr/bin/env bash
# run_planarity_v1b.sh (2026-10-30): P1 (face level, planarity_v1.py) where still missing + P1b (vertex level,
# planarity_points_v1.py) on every checkpoint: v3 lambda 6.2e-3, v3 lambda 1.85e-3, dense original.
# Fixes the v1 runner bug (one `local` line expanded $T before assigning it, so every tag wrote <scene>-.json and the
# later tags were skipped as "exists"): the 11 files <scene>-.json / .log are renamed to <scene>-v3-lam0.0062.*.
# Restartable.  Face rows -> $AN/planarity-v1.txt (PLANARITY ...), point rows -> $AN/planarity-v1b.txt (PLANES ...).
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
AN=/scratch/zr2245/analysis; OUT=$AN/planarity-v1; TABF=$AN/planarity-v1.txt; TABP=$AN/planarity-v1b.txt
PYF=$AN/diag/planarity_v1.py; PYP=$AN/diag/planarity_points_v1.py; mkdir -p $OUT
SCENES=${SCENES:-"room kitchen counter bonsai garden stump bicycle flowers treehill truck train"}
for S in $SCENES; do
  for ext in json log; do [ -f $OUT/$S-.$ext ] && [ ! -f $OUT/$S-v3-lam0.0062.$ext ] && mv $OUT/$S-.$ext $OUT/$S-v3-lam0.0062.$ext && echo "renamed $S-.$ext -> $S-v3-lam0.0062.$ext"; done
done
n_ok=0; n_fail=0; n_skip=0
run_one () {
  local S; local T; local CK; S=$1; T=$2; CK=$3
  local JF=$OUT/$S-$T.json; local LF=$OUT/$S-$T.log; local JP=$OUT/$S-$T-points.json; local LP=$OUT/$S-$T-points.log
  if [ ! -f "$CK" ]; then echo "missing: $S $T ($CK)"; n_skip=$((n_skip+1)); return; fi
  if [ -s $JF ]; then echo "exists (faces): $S $T"; else
    echo "=== $S $T faces ($(date +%H:%M:%S)) ==="
    if python $PYF --state "$CK" --label "$S $T" --out $JF > $LF 2>&1; then grep -h "^PLANARITY" $LF | tee -a $TABF; else echo "FAILED: $S $T faces"; tail -5 $LF; n_fail=$((n_fail+1)); fi
  fi
  if [ -s $JP ]; then echo "exists (points): $S $T"; n_skip=$((n_skip+1)); return; fi
  echo "=== $S $T points ($(date +%H:%M:%S)) ==="
  if python $PYP --state "$CK" --label "$S $T" --out $JP --save_labels $OUT/$S-$T-plane-labels.npy > $LP 2>&1; then
    grep -h "^PLANES" $LP | tee -a $TABP; n_ok=$((n_ok+1))
  else echo "FAILED: $S $T points"; tail -5 $LP; n_fail=$((n_fail+1)); fi
}
dense_ckpt () {  # first existing dense checkpoint of a scene
  local S=$1; local c
  for c in /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt \
           /scratch/zr2245/$S-fixed-sharing-0N4OSU/run/original/point_cloud/iteration_30000/point_cloud_state_dict.pt \
           /scratch/zr2245/$S-rd-error-lam0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; do
    [ -f "$c" ] && { echo "$c"; return; }
  done
  echo "/scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt"
}
for S in $SCENES; do
  run_one $S v3-lam0.0062  /scratch/zr2245/$S-rd3-lam0.0062-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  run_one $S v3-lam0.00185 /scratch/zr2245/$S-rd3-lam0.00185-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
done
for S in $SCENES; do run_one $S dense "$(dense_ckpt $S)"; done
echo "===== $TABP ====="; cat $TABP
echo "run_planarity_v1b done: ok $n_ok failed $n_fail skipped $n_skip  $(date)"
