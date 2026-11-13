#!/usr/bin/env bash
# submit-batch35-v1.sh (2026-11-13): the hold controller with MeshSplatting's importance score in place of one of our two selection rules.
#   holdis : the triangles to split are drawn by upstream importance sampling (how many are split: unchanged plan)
#   holdic : upstream's importance score replaces the residual in the cost of an edge collapse (byte target and fold-over test unchanged)
#   Scenes garden (outdoor) and room (indoor), target 10 MB.  4 GPU jobs.  Reference: the hold runs of batch31 (garden 9.997 MB, room 9.992 MB).
# Usage: bash submit-batch35-v1.sh [VARIANTS]   (default "holdis holdic")
AN=/scratch/zr2245/analysis; VAR=${1:-"holdis holdic"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl8.py 1092 0f6ca00cec9da3f1cf3da0f6d9f59bc5
chk $AN/ablation-b24-r9.sbatch 220 86899591acad64fe4783baab438a0093
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b_abl8.py && bash -n $AN/ablation-b24-r9.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
echo "CHECK OK (variants: $VAR)"; n=0
cfg () { case $1 in room) echo "INDOOR=1,IMAGES=images_2,XYZ=q14";; garden) echo "INDOOR=0,IMAGES=images_4,XYZ=q15";; esac; }
for V in $VAR; do
  case $V in holdis|holdic) ;; *) echo "FAILED: unknown variant $V"; continue;; esac
  for S in garden room; do
    [ -e /scratch/zr2245/$S-abl24-$V-t10mb ] && { echo "exists, skipped: $S $V"; continue; }
    j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=02:30:00 --export=ALL,SCENE=$S,$(cfg $S),SMOKE=0,SELFCHECK=0,ABL=$V,TARGETS="10" --job-name=$S-$V-b35 $AN/ablation-b24-r9.sbatch) && { n=$((n+1)); echo "submitted $j  $S-$V-b35"; } || echo "FAILED: sbatch $S $V"
  done
done
echo "$n GPU jobs submitted"; squeue -u $USER -h -o "%T" | sort | uniq -c
