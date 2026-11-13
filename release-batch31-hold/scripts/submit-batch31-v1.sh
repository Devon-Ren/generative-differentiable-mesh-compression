#!/usr/bin/env bash
# submit-batch31-v1.sh (2026-11-12): the prediction-free controller ("hold": measure every 500 iterations, remove what is above the target now).
#   H   : all 21 scenes at 10 MB (captured) / 2.5 MB (objects) -> abl24-hold.  garden and lego start first; the other 19 only after those two ended without error.
#   HX  : the other targets of the main table: 20 and 5 MB (captured), 4 and 1.5 MB (objects); two runs per job; after the same two pilots; lower priority.
# Usage: bash submit-batch31-v1.sh [GROUPS]   (default "H HX")
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; GR=${1:-"H HX"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl7.py 1083 1eeb64ab5032e903f69ad1a21466d99f
chk $AN/ablation-b24-r8.sbatch 218 82259b73557bfcdb760648f42fad08cc
chk $AN/ablation-b24-r8-blender.sbatch 220 d204fb12d9bc3b5c634ab9870b1f38b4
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b_abl7.py && bash -n $AN/ablation-b24-r8.sbatch && bash -n $AN/ablation-b24-r8-blender.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
echo "CHECK OK (groups: $GR)"; n=0
cfg () { case $1 in room) echo "INDOOR=1,IMAGES=images_2,XYZ=q14";; kitchen) echo "INDOOR=1,IMAGES=images_2,XYZ=q16";; counter|bonsai) echo "INDOOR=1,IMAGES=images_2,XYZ=q15";;
  garden|flowers) echo "INDOOR=0,IMAGES=images_4,XYZ=q15";; bicycle|stump|treehill) echo "INDOOR=0,IMAGES=images_4,XYZ=q16";; truck|train) echo "INDOOR=0,IMAGES=images,XYZ=q18,DATA=$D/tandt/$1";;
  drjohnson|playroom) echo "INDOOR=1,IMAGES=images,XYZ=q15,DATA=$D/db/$1";; esac; }
ctl () { # $1 scene  $2 targets ("10" or "20 5")  $3 time limit  $4 tag for the job name  $5.. extra sbatch options; sets LASTJ
  local S=$1 TG=$2 TL=$3 TAG=$4; shift 4; local j
  case $S in lego|chair|drums|ficus|hotdog|materials|mic|ship) j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=$TL "$@" --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,ABL=hold,TARGETS="$TG" --job-name=$S-$TAG-b31 $AN/ablation-b24-r8-blender.sbatch);;
    *) j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=$TL "$@" --export=ALL,SCENE=$S,$(cfg $S),SMOKE=0,SELFCHECK=0,ABL=hold,TARGETS="$TG" --job-name=$S-$TAG-b31 $AN/ablation-b24-r8.sbatch);; esac
  LASTJ=$j; [ -n "$j" ] && { n=$((n+1)); echo "submitted $j  $S-$TAG-b31 (targets $TG)"; } || echo "FAILED: sbatch $S $TAG"; }
MIP="room kitchen counter bonsai garden bicycle stump flowers treehill truck train drjohnson playroom"; OBJ="lego chair drums ficus hotdog materials mic ship"
pilot () { # $1 job name -> job id from the queue, or 'done' when its result row exists, or ''
  local id=$(squeue -u $USER -h -n $1 -o %i | head -1); [ -n "$id" ] && { echo $id; return; }
  case $1 in garden-hold-b31) grep -q "^garden abl24-hold | target 10 MB" $AN/ablation-b24.txt 2>/dev/null && echo done;; lego-hold-b31) grep -q "^lego abl24-hold | target 2.5 MB" $AN/ablation-b24-blender.txt 2>/dev/null && echo done;; esac; }
JG=$(pilot garden-hold-b31); JL=$(pilot lego-hold-b31)
for G in $GR; do case $G in
H)  [ -z "$JG" ] && { ctl garden 10 02:30:00 hold; JG=$LASTJ; }; [ -z "$JL" ] && { ctl lego 2.5 02:30:00 hold; JL=$LASTJ; }
    DG=""; [ "$JG" != done ] && DG="--dependency=afterok:$JG"; DL=""; [ "$JL" != done ] && DL="--dependency=afterok:$JL"
    for S in $MIP; do [ $S = garden ] && continue; [ -e /scratch/zr2245/$S-abl24-hold-t10mb ] && { echo "exists, skipped: $S hold 10"; continue; }; ctl $S 10 02:30:00 hold $DG; done
    for S in $OBJ; do [ $S = lego ] && continue; [ -e /scratch/zr2245/$S-abl24b-hold-t2.5mb ] && { echo "exists, skipped: $S hold 2.5"; continue; }; ctl $S 2.5 02:30:00 hold $DL; done;;
HX) [ -n "$JG" ] && [ -n "$JL" ] || { echo "NOT SUBMITTED: HX needs the two pilots (run group H first)"; continue; }
    DG=""; [ "$JG" != done ] && DG="--dependency=afterok:$JG"; DL=""; [ "$JL" != done ] && DL="--dependency=afterok:$JL"
    for S in $MIP; do ctl $S "20 5" 04:30:00 holdx --nice=200 $DG; done
    for S in $OBJ; do ctl $S "4 1.5" 04:00:00 holdx --nice=200 $DL; done;;
*) echo "FAILED: unknown group $G";;
esac; done
echo "$n GPU jobs submitted"; squeue -u $USER -h -o "%T" | sort | uniq -c
