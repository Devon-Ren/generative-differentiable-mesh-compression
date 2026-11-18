#!/usr/bin/env bash
# submit-batch49-v1.sh (2026-11-18): design experiments for the appendix.
#   cpu : one CPU job (about 30 min)
#         (1) VQ + Draco and VQ + TFAN: the same vector quantization as BM-Codec followed by an existing mesh codec, all 21 scenes
#         (2) the position bit depth given by the stated rule (grid step <= 0.1 x median edge) against the one used, all 21 scenes
#         (3) share of the training time spent in the size probes, from the logs of the ByteMesh runs
#   p1  : Phase 1 only (growth to the target, no collapse rounds) on garden, room, bicycle, kitchen, truck at 5, 10 and 20 MB:
#         15 GPU jobs of about 1 hour.  Same controller and settings as ByteMesh, but growth aims at the target (alpha 1.0) and Phase 2 is off.
# Needs in $AN: vq_meshcodec_v1.py, vq_zstd_v1.py, qbits_rule_v1.py, probe_time_v1.py, b46-scenes.sh, ablation-b24-r10.sbatch, o3dgc-v1/o3dgc_driver
# Usage: bash submit-batch49-v1.sh cpu | p1 | all        Collect: bash submit-batch49-v1.sh collect
AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245; ACC=torch_pr_38_general; W=${1:?cpu | p1 | all | collect}
if [ $W = collect ]; then
  for f in vq-mesh-b49.txt qbits-b49.txt probetime-b49.txt; do echo "== $f"; cat $AN/$f 2>/dev/null; done
  echo "== p1only"; grep -h -E "^(garden|room|bicycle|kitchen|truck) abl24-p1only \| target" $AN/ablation-b24.txt | cut -c1-330 | sort
  echo "== unfinished"; for f in $(ls -t $AN/b49-*.out $AN/abl24-*-b49-*.out 2>/dev/null); do tail -1 $f | grep -q "exit 0" || echo "$f : $(tail -1 $f | cut -c1-140)"; done; exit 0
fi
if [ $W = cpu ] || [ $W = all ]; then
  for f in vq_meshcodec_v1.py vq_zstd_v1.py qbits_rule_v1.py probe_time_v1.py b46-scenes.sh; do [ -e $AN/$f ] || { echo "FAILED: missing $AN/$f"; exit 1; }; done
  cat > $AN/b49-cpu.sh <<'JOB'
#!/usr/bin/env bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8; AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245; source $AN/b46-scenes.sh; PKG=$AN/vis_collapse-v12; OUT=$AN/vq-mesh-b49; mkdir -p $OUT
DRV=$AN/o3dgc-v1/o3dgc_driver; [ -x $DRV ] || DRV=$AN/tfan-time/o3dgc_driver_timed; [ -x $DRV ] && DOPT="--driver $DRV" || { DOPT=""; echo "note: no TFAN driver, that column stays empty"; }
python -c "import DracoPy, zstandard" 2>/dev/null || pip install DracoPy zstandard > $OUT/pip.log 2>&1
python $AN/vq_meshcodec_v1.py --selftest $DOPT || { echo "FAILED: selftest"; exit 1; }
: > $AN/vq-mesh-b49.txt; : > $AN/qbits-b49.txt
for S in $MAIN $DBS $OBJS; do CG=$(run $S)/axisB/v1.0
  [ -f $CG/deliver/shared-final.pt ] || { echo "note: $S: no $CG/deliver/shared-final.pt, skipped" | tee -a $AN/vq-mesh-b49.txt; continue; }
  bash $PKG/stage2_run.sh $AN/vq_meshcodec_v1.py --shared $CG/deliver/shared-final.pt --xyz q$(qp $S) --dc q7 $DOPT --out $OUT/$S.json --label "$S" > $OUT/$S.log 2>&1 && grep -h -E "^(VQMESH|note)" $OUT/$S.log | tee -a $AN/vq-mesh-b49.txt || { echo "FAILED: vqmesh $S" | tee -a $AN/vq-mesh-b49.txt; tail -6 $OUT/$S.log; }
  [ -f $CG/finetune/best.pt ] && python $AN/qbits_rule_v1.py --ckpt $CG/finetune/best.pt --used $(qp $S) --label $S 2>&1 | grep -E "^QBITS|Error" | tee -a $AN/qbits-b49.txt
done
python $AN/probe_time_v1.py $(ls -d $ROOT/*-abl24-hold-t*mb 2>/dev/null) | tee $AN/probetime-b49.txt
echo "b49-cpu exit 0  $(date)"
JOB
  j=$(sbatch --parsable --account=$ACC --cpus-per-task=8 --mem=48G --time=01:30:00 --job-name=b49-cpu --output=$AN/b49-cpu-%j.out $AN/b49-cpu.sh) && echo "submitted $j  b49-cpu (CPU)" || echo "FAILED: sbatch cpu"
fi
if [ $W = p1 ] || [ $W = all ]; then
  chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; return 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; return 1; }; }
  chk $AN/rdgrowth/train_rd_adapt_v9b_abl7.py 1083 1eeb64ab5032e903f69ad1a21466d99f && chk $AN/ablation-b24-r10.sbatch 221 5f4e05f1e2db138ca5341214f0c12ad7 || exit 1
  # r12 = r10 + the mode p1only: the controller of ByteMesh with growth aiming at the target (alpha 1.0) and no collapse rounds
  python3 - "$AN/ablation-b24-r10.sbatch" "$AN/ablation-b24-r12.sbatch" <<'PY' || exit 1
import sys
s = open(sys.argv[1]).read(); L = [l for l in s.split('\n') if l.startswith('  hold) ')]
if len(L) != 1: sys.exit('FAILED: the hold line was not found exactly once')
new = '  p1only) XARGS=""; OVER=1.0; PRUNE=""; EXTRA="" ;;   # batch49: Phase 1 only (growth to the target, no collapse rounds)'
open(sys.argv[2], 'w').write(s.replace(L[0], L[0] + '\n' + new, 1)); print('written', sys.argv[2])
PY
  bash -n $AN/ablation-b24-r12.sbatch || { echo "CHECK FAILED: syntax of r12"; exit 1; }
  [ "$(diff $AN/ablation-b24-r10.sbatch $AN/ablation-b24-r12.sbatch | grep -c '^[<>]')" = 1 ] || { echo "CHECK FAILED: r12 differs from r10 by more than one line"; exit 1; }
  cfg () { case $1 in garden) echo "INDOOR=0,IMAGES=images_4,XYZ=q15";; room) echo "INDOOR=1,IMAGES=images_2,XYZ=q14";; bicycle) echo "INDOOR=0,IMAGES=images_4,XYZ=q16";; kitchen) echo "INDOOR=1,IMAGES=images_2,XYZ=q16";;
    truck) echo "INDOOR=0,IMAGES=images,XYZ=q18,DATA=$ROOT/datasets/tandt/truck";; esac; }
  n=0; for T in 5 10 20; do for S in garden room bicycle kitchen truck; do
    [ -e $ROOT/$S-abl24-p1only-t${T}mb ] && { echo "exists, skipped: $S p1only $T MB"; continue; }
    j=$(sbatch --parsable --account=$ACC --cpus-per-task=6 --time=02:30:00 --export=ALL,SCENE=$S,$(cfg $S),SMOKE=0,SELFCHECK=0,ABL=p1only,TARGETS="$T" --job-name=$S-p1only$T-b49 --output=$AN/abl24-%x-%j.out $AN/ablation-b24-r12.sbatch) && { n=$((n+1)); echo "submitted $j  $S p1only $T MB"; } || echo "FAILED: sbatch $S $T"
  done; done; echo "$n GPU jobs submitted"
fi
squeue -u $USER -h -o "%j %T" | grep b49 | sort | uniq -c
