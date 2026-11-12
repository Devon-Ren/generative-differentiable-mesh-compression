#!/usr/bin/env bash
# submit-batch30-v1.sh (2026-11-12): the overnight batch.  Everything at ONE target per dataset: 10 MB (13 captured scenes), 2.5 MB (8 objects).
#   FIN  : the final design of the controller (make_batch30_files_v1.py: growth aim 1.35 for every scene, predictive rounds with a 5% margin,
#          measured rounds at 24,000 / 25,500 / 27,000) on all 21 scenes -> abl24-final.  garden and lego start first; the other 19 start only
#          after those two have ended without error.
#   CAP  : MeshSplatting with a vertex cap from the leave-one-out size prior, on every scene not yet run (15) + the retry rule (one more training
#          on a miss > 10%) + the two retries still open (kitchen 900000, lego third training 130000)
#   R35  : fallback, third version without the late rounds, growth aim 1.35 (14 runs; 7 exist from the pilot)  -> abl24-ratio35
#   R    : fallback, third version, default growth aim (1.25 captured, 1.1 objects)                           -> abl24-ratio
# Queue order: FIN, then CAP, then R35, then R (nice values).  R35 and R start only after a pilot job of the third version ended without error.
# Usage: bash submit-batch30-v1.sh [GROUPS]   (default "FIN CAP R35 R")
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; GR=${1:-"FIN CAP R35 R"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/budget-sweep-v2.sbatch 82 247b78f2bad723ecd7cae26e236076d9
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
chk $AN/budget-sweep-blender-v1.sbatch 105 d720b8ccba85622f90f25ee0b52b77ee
chk $AN/rdgrowth/train_rd_adapt_v9b_abl5.py 1078 3f1f876e372659b9dd8ad0a07610497d
chk $AN/rdgrowth/train_rd_adapt_v9b_abl6.py 1081 37d3ca515f0df262bdee538c8a0fe162
chk $AN/ablation-b24-r7.sbatch 217 e9c48dba072b122a671efffc27996380
chk $AN/ablation-b24-r7-blender.sbatch 219 6d731771709ee222175e858c2a56355d
chk $AN/ablation-b24-r6.sbatch 216 4625b51e1cb8f7086c228fda3ec69fc7
chk $AN/ablation-b24-r6-blender.sbatch 218 c7e8a016bcabf0391204c07936ceab15
chk $AN/capjob-v1.sh 17 5a42c14029c6bbb3db8abcf1e921f5ae
chk $AN/cap-pass2-v3.sbatch 32 d96578bca24ef217eae0929e34d46175
bash -n $AN/capjob-v1.sh && bash -n $AN/cap-pass2-v3.sbatch && python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b_abl6.py && bash -n $AN/ablation-b24-r7.sbatch && bash -n $AN/ablation-b24-r7-blender.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
for S in room kitchen counter bonsai drjohnson playroom; do [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no --indoor original for $S"; exit 1; }; done
for S in garden bicycle stump flowers treehill truck train; do [ -f /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no original for $S"; exit 1; }; done
for S in lego chair drums ficus hotdog materials mic ship; do [ -f /scratch/zr2245/$S-blender4-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no $S-blender4-best original"; exit 1; }; done
echo "CHECK OK (groups: $GR)"; source $AN/capjob-v1.sh; n=0
cfg () { case $1 in room) echo "INDOOR=1,IMAGES=images_2,XYZ=q14";; kitchen) echo "INDOOR=1,IMAGES=images_2,XYZ=q16";; counter|bonsai) echo "INDOOR=1,IMAGES=images_2,XYZ=q15";;
  garden|flowers) echo "INDOOR=0,IMAGES=images_4,XYZ=q15";; bicycle|stump|treehill) echo "INDOOR=0,IMAGES=images_4,XYZ=q16";; truck|train) echo "INDOOR=0,IMAGES=images,XYZ=q18,DATA=$D/tandt/$1";;
  drjohnson|playroom) echo "INDOOR=1,IMAGES=images,XYZ=q15,DATA=$D/db/$1";; esac; }
ctl () { # $1 scene  $2 variant  $3 sbatch generation (r6|r7)  $4.. extra sbatch options; sets LASTJ
  local S=$1 V=$2 G=$3; shift 3; local j
  case $S in lego|chair|drums|ficus|hotdog|materials|mic|ship) j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=02:30:00 "$@" --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,ABL=$V,TARGETS=2.5 --job-name=$S-$V-b30 $AN/ablation-b24-$G-blender.sbatch);;
    *) j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --time=02:30:00 "$@" --export=ALL,SCENE=$S,$(cfg $S),SMOKE=0,SELFCHECK=0,ABL=$V,TARGETS=10 --job-name=$S-$V-b30 $AN/ablation-b24-$G.sbatch);; esac
  LASTJ=$j; [ -n "$j" ] && { n=$((n+1)); echo "submitted $j  $S-$V-b30"; } || echo "FAILED: sbatch $S $V"; }
dep () { # $1 pilot job name  $2 result table  $3 row prefix -> prints the dependency option ('' when the pilot already finished); fails when the pilot is unknown
  local id=$(squeue -u $USER -h -n $1 -o %i | head -1)
  if [ -n "$id" ]; then echo "--dependency=afterok:$id"; return 0; fi
  grep -q "^$3 " $2 2>/dev/null && return 0; return 1; }
MIP="room kitchen counter bonsai garden bicycle stump flowers treehill truck train drjohnson playroom"; OBJ="lego chair drums ficus hotdog materials mic ship"
for G in $GR; do case $G in
CAP) for X in room:1078000:2.15 counter:1083000:2.11 bonsai:1073000:2.17 bicycle:1080000:2.14 stump:1069000:2.15 flowers:1071000:2.05 treehill:1079000:2.09 train:1085000:2.12 drjohnson:1081000:2.11 drums:176000:0.89 ficus:171000:1.02 hotdog:177000:0.9 materials:176000:0.92 mic:174000:0.95 ship:177000:0.87; do
       S=${X%%:*}; R=${X#*:}; B=${R%%:*}; C=${R#*:}; case $S in drums|ficus|hotdog|materials|mic|ship) T=2.5;; *) T=10;; esac
       j=$(NICE=100 capjob $S $B $S-cap1-b30) && { n=$((n+1)); echo "submitted $j  $S-cap1-b30 (cap $B)"; sbatch --parsable --dependency=afterok:$j --export=ALL,S=$S,CAP1=$B,T=$T,C=$C --job-name=$S-pass2v3 $AN/cap-pass2-v3.sbatch >/dev/null; } || echo "FAILED: cap $S"; done
     j=$(NICE=100 capjob kitchen 900000 kitchen-cap2-b30) && { n=$((n+1)); echo "submitted $j  kitchen-cap2-b30 (cap 900000)"; }
     j=$(NICE=100 capjob lego 130000 lego-cap3-b30) && { n=$((n+1)); echo "submitted $j  lego-cap3-b30 (cap 130000)"; };;
FIN) ctl garden final r7; JG=$LASTJ; ctl lego final r7; JL=$LASTJ
     [ -n "$JG" ] && for S in $MIP; do [ $S = garden ] && continue; ctl $S final r7 --dependency=afterok:$JG; done
     [ -n "$JL" ] && for S in $OBJ; do [ $S = lego ] && continue; ctl $S final r7 --dependency=afterok:$JL; done;;
R|R35) V=ratio; NI=600; [ $G = R35 ] && { V=ratio35; NI=300; }
     if DC=$(dep garden-ratio35-b29 $AN/ablation-b24.txt "garden abl24-ratio35"); then for S in $MIP; do [ -e /scratch/zr2245/$S-abl24-$V-t10mb ] && { echo "exists, skipped: $S $V"; continue; }; ctl $S $V r6 --nice=$NI $DC; done
     else echo "NOT SUBMITTED: $V on captured scenes (pilot job garden-ratio35-b29 is neither in the queue nor finished with a result row)"; fi
     if DB=$(dep lego-ratio25-b29 $AN/ablation-b24-blender.txt "lego abl24-ratio25"); then for S in $OBJ; do [ -e /scratch/zr2245/$S-abl24b-$V-t2.5mb ] && { echo "exists, skipped: $S $V"; continue; }; ctl $S $V r6 --nice=$NI $DB; done
     else echo "NOT SUBMITTED: $V on objects (pilot job lego-ratio25-b29 is neither in the queue nor finished with a result row)"; fi;;
*) echo "FAILED: unknown group $G";;
esac; done
echo "$n GPU jobs submitted"; squeue -u $USER -h -o "%T" | sort | uniq -c
