#!/usr/bin/env bash
# submit-batch41-v1.sh (2026-11-15): growth-aim ablation of the hold controller.  garden and room, target 10 MB, aims 1.0 / 1.2 / 1.5
# (ABL holda100 / holda120 / holda150; the hold runs of batch31 are the aim 1.35 reference: garden 9.997 MB 23.672 dB, room 9.992 MB 27.991 dB).
# 6 GPU jobs, about 50 to 60 minutes each.   Usage: bash submit-batch41-v1.sh [VARIANTS]   (default "holda100 holda120 holda150")
AN=/scratch/zr2245/analysis; VAR=${1:-"holda100 holda120 holda150"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl7.py 1083 1eeb64ab5032e903f69ad1a21466d99f
chk $AN/ablation-b24-r10.sbatch 221 5f4e05f1e2db138ca5341214f0c12ad7
bash -n $AN/ablation-b24-r10.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
echo "CHECK OK (variants: $VAR)"; n=0
cfg () { case $1 in room) echo "INDOOR=1,IMAGES=images_2,XYZ=q14";; garden) echo "INDOOR=0,IMAGES=images_4,XYZ=q15";; esac; }
for V in $VAR; do
  case $V in holda100|holda120|holda150) ;; *) echo "FAILED: unknown variant $V"; continue;; esac
  for S in garden room; do
    [ -e /scratch/zr2245/$S-abl24-$V-t10mb ] && { echo "exists, skipped: $S $V"; continue; }
    j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=02:30:00 --export=ALL,SCENE=$S,$(cfg $S),SMOKE=0,SELFCHECK=0,ABL=$V,TARGETS="10" --job-name=$S-$V-b41 $AN/ablation-b24-r10.sbatch) && { n=$((n+1)); echo "submitted $j  $S-$V-b41"; } || echo "FAILED: sbatch $S $V"
  done
done
echo "$n GPU jobs submitted"; squeue -u $USER -h -o "%T" | sort | uniq -c
