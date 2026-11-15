#!/usr/bin/env bash
# dc-predict-v1.sh (2026-11-15): bytes saved by neighbour prediction on the DC colour stream, on the same 11 meshes as the codec-time run
# (MeshSplatting trained with a vertex cap near 1.08 million, fine-tuned).  CPU only, about 2 minutes per scene.
# Reads <cap run>/axisB/v1.0/deliver/shared-final.pt.  Rows -> $AN/dc-predict.txt.  Run directly in a CPU session:  bash dc-predict-v1.sh
[ -f /scratch/zr2245/anaconda3/etc/profile.d/conda.sh ] && { source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; }
[ -f /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate ] && source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
AN=${AN:-/scratch/zr2245/analysis}; ROOT=${ROOT:-/scratch/zr2245}; PKG=${PKG:-$AN/vis_collapse-v12}; OUT=$AN/dc-predict; TAB=$AN/dc-predict.txt
SCENES=${SCENES:-"garden room truck kitchen counter bonsai bicycle stump flowers treehill train"}; mkdir -p $OUT
mid () { case $1 in room) echo 1078000;; kitchen) echo 1086000;; counter) echo 1083000;; bonsai|garden) echo 1073000;; bicycle) echo 1080000;; stump) echo 1069000;; flowers) echo 1071000;; treehill) echo 1079000;; truck) echo 1078000;; train) echo 1085000;; esac; }
qp () { case $1 in room) echo 14;; kitchen|bicycle|stump|treehill) echo 16;; counter|bonsai|garden|flowers) echo 15;; truck|train) echo 18;; esac; }
for f in $AN/dc_predict_v1.py $PKG/stage2_run.sh; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
python -c "import zstandard" 2>/dev/null || pip install zstandard > $OUT/pip.log 2>&1 || { echo "FAILED: zstandard not installable"; exit 1; }
python $AN/dc_predict_v1.py --selftest || { echo "FAILED: selftest"; exit 1; }
echo "dc-predict-v1 | host $(hostname) | $(date)"
nf=0
for S in $SCENES; do B=$(mid $S); Q=$(qp $S); CG=$ROOT/$S-budget-$B/axisB/v1.0
  if [ -f $CG/deliver/shared-final.pt ]; then
    bash $PKG/stage2_run.sh $AN/dc_predict_v1.py --shared $CG/deliver/shared-final.pt --xyz q$Q --dc q7 --out $OUT/$S.json --label "$S cap$B" > $OUT/$S.log 2>&1 && grep -h "^DCPRED" $OUT/$S.log | tee -a $TAB || { echo "FAILED: $S"; tail -6 $OUT/$S.log; nf=$((nf+1)); }
  else echo "note: $S: no $CG/deliver/shared-final.pt, skipped"; nf=$((nf+1)); fi
done
echo "dc-predict-v1 exit 0 ($nf failures)  $(date)"
