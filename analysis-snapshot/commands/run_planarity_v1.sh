#!/usr/bin/env bash
# run_planarity_v1.sh (2026-10-30): CPU planarity probe on every existing checkpoint (lambda runs first, dense originals last).
# Restartable: a checkpoint whose json exists is skipped.  Rows -> $AN/planarity-v1.txt, json + log -> $AN/planarity-v1/.
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
AN=/scratch/zr2245/analysis; OUT=$AN/planarity-v1; TAB=$AN/planarity-v1.txt; PY=$AN/diag/planarity_v1.py; mkdir -p $OUT
SCENES=${SCENES:-"room kitchen counter bonsai garden stump bicycle flowers treehill truck train"}
n_ok=0; n_fail=0; n_skip=0
run_one () {  # scene tag checkpoint
  local S=$1 T=$2 CK=$3 J=$OUT/$S-$T.json L=$OUT/$S-$T.log
  if [ ! -f $CK ]; then echo "missing: $S $T ($CK)"; n_skip=$((n_skip+1)); return; fi
  if [ -s $J ]; then echo "exists: $S $T"; n_skip=$((n_skip+1)); return; fi
  echo "=== $S $T ($(date +%H:%M:%S)) ==="
  if python $PY --state $CK --label "$S $T" --out $J > $L 2>&1; then
    grep -h "^PLANARITY" $L | tee -a $TAB; n_ok=$((n_ok+1))
  else
    echo "FAILED: $S $T"; tail -5 $L; n_fail=$((n_fail+1))
  fi
}
for S in $SCENES; do
  run_one $S v3-lam0.0062  /scratch/zr2245/$S-rd3-lam0.0062-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  run_one $S v3-lam0.00185 /scratch/zr2245/$S-rd3-lam0.00185-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
done
for S in $SCENES; do
  run_one $S dense /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
done
echo "===== $TAB ====="; cat $TAB
echo "run_planarity_v1 done: ok $n_ok failed $n_fail skipped $n_skip  $(date)"
