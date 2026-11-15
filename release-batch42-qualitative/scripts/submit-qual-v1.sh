#!/usr/bin/env bash
# submit-qual-v1.sh (2026-11-15): renders for the qualitative figure.  garden, room, bicycle, truck; per scene the same test views of
#   FP32  : the MeshSplatting model with released settings
#   TFAN  : MeshSplatting with a vertex cap of 600,000, SH quantised to 4 bits (the mesh behind the TFAN point near 10 MB, batch36)
#   SIMP  : the FP32 model simplified after training to about 10 MB, our codec (decoded file)
#   OURS  : the hold run at a target of 10 MB (decoded file)
# One GPU job per scene (render-b23.sbatch, unchanged), about 15 minutes each.  Output: /scratch/zr2245/renders-qual/<scene>/view-*.png, psnr.txt
# Usage: bash submit-qual-v1.sh        then, when the jobs are done:  bash pack-qual-v1.sh
AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245; D=$ROOT/datasets; OUTR=$ROOT/renders-qual; VIEWS=${VIEWS:-"0 3 6 9 12 15 18 21"}; SCENES=${SCENES:-"garden room bicycle truck"}
for f in $AN/render/decode_asset_v2.py $AN/render/render_ckpt_v1.py $AN/render-b23.sbatch; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
bash -n $AN/render-b23.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
cfg () { case $1 in
  garden)  echo "$ROOT/garden-vis-collapse $D/m360/garden images_4 919000";;
  room)    echo "$ROOT/room-indoor $D/m360/room images_2 1020000";;
  bicycle) echo "$ROOT/bicycle-vis-collapse $D/m360/bicycle images_4 1201000";;
  truck)   echo "$ROOT/truck-vis-collapse $D/tandt/truck images 929000";; esac; }
n=0; mkdir -p $OUTR
for S in $SCENES; do read ORIG DATA IMAGES B <<< "$(cfg $S)"
  FP=$ORIG/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; TF=$AN/generic-rd/$S/cap600000/sh4/point_cloud_state_dict.pt
  SI=$ORIG/axisB/g0ctrl-$B/deliver; OU=$ROOT/$S-abl24-hold-t10mb/axisB/v1.0/deliver; miss=0
  for p in $FP $TF; do [ -f $p ] || { echo "MISSING ($S): $p"; miss=1; }; done
  for p in $SI $OU; do [ -d $p ] || { echo "MISSING ($S): $p"; miss=1; }; done
  [ $miss = 0 ] || { echo "skipped: $S (see MISSING lines)"; continue; }
  [ -e $OUTR/$S/psnr.txt ] && { echo "exists, skipped: $OUTR/$S"; continue; }
  j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 --job-name=$S-qual --export=ALL,SCENE=$S,DATA=$DATA,IMAGES=$IMAGES,MODEL=$ORIG/original,VIEWS="$VIEWS",OUT=$OUTR/$S,ASSETS="FP32=$FP|TFAN=$TF|SIMP=$SI|OURS=$OU" $AN/render-b23.sbatch) && { n=$((n+1)); echo "submitted $j  $S-qual"; } || echo "FAILED: sbatch $S"
done
echo "$n GPU jobs submitted"; squeue -u $USER -h -o "%j %T" | grep qual
