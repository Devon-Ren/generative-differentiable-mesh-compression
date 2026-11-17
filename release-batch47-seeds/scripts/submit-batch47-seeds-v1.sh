#!/usr/bin/env bash
# submit-batch47-seeds-v1.sh (2026-11-17): run-to-run variation of ByteMesh (the hold controller of the paper, growth aim 1.35).
# Seeds 1 and 2 on garden, room, bicycle, kitchen and truck at a target of 10 MB; the paper runs (batch31, no reseeding = seed 0 of the
# upstream code) are the third run of each scene.  10 GPU jobs of about 1 hour.
# Usage: bash submit-batch47-seeds-v1.sh            Collect: bash submit-batch47-seeds-v1.sh collect
AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245; ACC=torch_pr_38_general; SCENES="garden room bicycle kitchen truck"; SEEDS="1 2"
if [ "${1:-}" = collect ]; then
  grep -h -E "^(garden|room|bicycle|kitchen|truck) abl24-(hold|holds[12]) \| target 10 MB" $AN/ablation-b24.txt | cut -c1-330 | sort
  echo "== unfinished"; for f in $(ls -t $AN/abl24-*-b47-*.out 2>/dev/null); do tail -1 $f | grep -q "exit 0" || echo "$f : $(tail -1 $f | cut -c1-140)"; done; exit 0
fi
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; return 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; return 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl7.py 1083 1eeb64ab5032e903f69ad1a21466d99f && chk $AN/ablation-b24-r10.sbatch 221 5f4e05f1e2db138ca5341214f0c12ad7 || exit 1
# r11 = r10 + the mode holds<N>: identical to "hold" (the paper's controller) with --seed N; the run directory and the result row carry the mode name
python3 - "$AN/ablation-b24-r10.sbatch" "$AN/ablation-b24-r11.sbatch" <<'PY' || exit 1
import sys
s = open(sys.argv[1]).read(); L = [l for l in s.split('\n') if l.startswith('  hold) ')]
if len(L) != 1 or 'OVER=1.35' not in L[0] or '--trim_hold' not in L[0]: sys.exit('FAILED: the hold line was not found exactly once')
new = L[0].replace('  hold) ', '  holds[1-9]) ', 1).replace('--allow_any_prune_iter"', '--allow_any_prune_iter --seed ${ABL#holds}"', 1)
if new.count('--seed ${ABL#holds}') != 1: sys.exit('FAILED: the seed argument could not be added')
open(sys.argv[2], 'w').write(s.replace(L[0], L[0] + '\n' + new, 1)); print('written', sys.argv[2])
PY
bash -n $AN/ablation-b24-r11.sbatch || { echo "CHECK FAILED: syntax of r11"; exit 1; }
[ "$(diff $AN/ablation-b24-r10.sbatch $AN/ablation-b24-r11.sbatch | grep -c '^[<>]')" = 1 ] || { echo "CHECK FAILED: r11 differs from r10 by more than the one added line"; exit 1; }
grep -n "holds\[1-9\])" $AN/ablation-b24-r11.sbatch | cut -c1-200
cfg () { case $1 in garden) echo "INDOOR=0,IMAGES=images_4,XYZ=q15";; room) echo "INDOOR=1,IMAGES=images_2,XYZ=q14";; bicycle) echo "INDOOR=0,IMAGES=images_4,XYZ=q16";; kitchen) echo "INDOOR=1,IMAGES=images_2,XYZ=q16";;
  truck) echo "INDOOR=0,IMAGES=images,XYZ=q18,DATA=$ROOT/datasets/tandt/truck";; esac; }
n=0; for K in $SEEDS; do for S in $SCENES; do V=holds$K
  [ -e $ROOT/$S-abl24-$V-t10mb ] && { echo "exists, skipped: $S $V"; continue; }
  j=$(sbatch --parsable --account=$ACC --cpus-per-task=6 --time=02:30:00 --export=ALL,SCENE=$S,$(cfg $S),SMOKE=0,SELFCHECK=0,ABL=$V,TARGETS="10" --job-name=$S-$V-b47 --output=$AN/abl24-%x-%j.out $AN/ablation-b24-r11.sbatch) && { n=$((n+1)); echo "submitted $j  $S-$V-b47"; } || echo "FAILED: sbatch $S $V"
done; done
echo "$n GPU jobs submitted"; squeue -u $USER -h -o "%j %T" | grep b47 | sort | uniq -c
