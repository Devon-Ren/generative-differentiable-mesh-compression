#!/usr/bin/env bash
# submit-batch34-v1.sh (2026-11-13): the SAME vertex cap on all 13 captured scenes: how far apart are the files?
#   MeshSplatting with --max_points = 300,000 / 600,000 / 900,000, fine-tuning and our codec as in every other cap run.  39 GPU jobs.
#   The three caps are fixed here, before any of these runs.  (All 13 scenes already have a run at a cap of 1,069,000-1,086,000.)
# Usage: bash submit-batch34-v1.sh [CAPS]   (default "300000 600000 900000")
AN=/scratch/zr2245/analysis; CAPS=${1:-"300000 600000 900000"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/capjob-v1.sh 17 5a42c14029c6bbb3db8abcf1e921f5ae
chk $AN/budget-sweep-v2.sbatch 82 247b78f2bad723ecd7cae26e236076d9
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
echo "CHECK OK (caps: $CAPS)"; source $AN/capjob-v1.sh; n=0
for B in $CAPS; do
  case $B in 300000|600000|900000) ;; *) echo "FAILED: cap $B is not one of 300000 600000 900000"; continue;; esac
  for S in room kitchen counter bonsai garden bicycle stump flowers treehill truck train drjohnson playroom; do
    j=$(capjob $S $B $S-cap$B-b34) && { n=$((n+1)); echo "submitted $j  $S cap $B"; } || echo "FAILED: cap $S $B"
  done
done
echo "$n GPU jobs submitted"; squeue -u $USER -h -o "%T" | sort | uniq -c
