#!/usr/bin/env bash
# submit-batch48-v1.sh (2026-11-17): the baseline "vector quantization + zstd" on all 21 scenes (same meshes as the time table: MeshSplatting
# with the middle cap).  One CPU job, about 15 minutes.   Usage: bash submit-batch48-v1.sh        Collect: bash submit-batch48-v1.sh collect
AN=/scratch/zr2245/analysis; ACC=torch_pr_38_general
if [ "${1:-}" = collect ]; then cat $AN/vq-zstd-b48.txt 2>/dev/null; echo "== job log tail"; tail -3 $(ls -t $AN/b48-vqzstd-*.out 2>/dev/null | head -1) 2>/dev/null; exit 0; fi
for f in $AN/vq_zstd_v1.py $AN/b46-scenes.sh $AN/vis_collapse-v12/stage2_run.sh; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
cat > $AN/b48-vqzstd.sh <<'JOB'
#!/usr/bin/env bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8; AN=/scratch/zr2245/analysis; source $AN/b46-scenes.sh; PKG=$AN/vis_collapse-v12; OUT=$AN/vq-zstd-b48; TAB=$AN/vq-zstd-b48.txt; mkdir -p $OUT; : > $TAB
python -c "import zstandard" 2>/dev/null || pip install zstandard > $OUT/pip.log 2>&1
python $AN/vq_zstd_v1.py --selftest || { echo "FAILED: selftest"; exit 1; }
for S in $MAIN $DBS $OBJS; do CG=$(run $S)/axisB/v1.0
  [ -f $CG/deliver/shared-final.pt ] || { echo "note: $S: no $CG/deliver/shared-final.pt, skipped" | tee -a $TAB; continue; }
  bash $PKG/stage2_run.sh $AN/vq_zstd_v1.py --shared $CG/deliver/shared-final.pt --xyz q$(qp $S) --dc q7 --out $OUT/$S.json --label "$S" > $OUT/$S.log 2>&1 && grep -h -E "^(VQZSTD|note)" $OUT/$S.log | tee -a $TAB || { echo "FAILED: $S" | tee -a $TAB; tail -6 $OUT/$S.log; }
done
echo "b48-vqzstd exit 0  $(date)"
JOB
j=$(sbatch --parsable --account=$ACC --cpus-per-task=8 --mem=48G --time=01:00:00 --job-name=b48-vqzstd --output=$AN/b48-vqzstd-%j.out $AN/b48-vqzstd.sh) && echo "submitted $j  b48-vqzstd (CPU)" || echo "FAILED: sbatch"
