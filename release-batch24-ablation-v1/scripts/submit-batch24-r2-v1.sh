#!/usr/bin/env bash
# submit-batch24-r2-v1.sh (2026-11-11): round 2 on the remaining reservation hours (4 L40S until 11/12 09:35 EST), 27 jobs, ~65 GPU-h.
#  (a) third fixed price per scene for the reference curve: Mip/T&T lam 0.00062 (covers 20 MB), NeRF-Synthetic lam 0.00185 (covers 4 MB)
#      -> bounds the interpolation and gives the 20 MB / 4 MB quality columns (reviewer W4 / Q1)
#  (b) run-to-run variance of the full method: room, garden, truck at 10 MB and lego at 2.5 MB, seeds 1 and 2 (seed 0 = the paper's run) (Q6)
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl2.py 938 75e92ffc397018fe472fc97801d0b9f9
chk $AN/ablation-b24-r2.sbatch 203 2adf57a024da31f1eb41781b174cb49b
chk $AN/ablation-b24-r2-blender.sbatch 205 b4315505e93cc9137141d81eb8a37f60
bash -n $AN/ablation-b24-r2.sbatch && bash -n $AN/ablation-b24-r2-blender.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
grep -q "VERSION abl24-r2" $AN/ablation-b24-r2.sbatch && grep -q "VERSION abl24b-r2" $AN/ablation-b24-r2-blender.sbatch || { echo "CHECK FAILED: sbatch versions"; exit 1; }
for S in room garden truck; do [ -d /scratch/zr2245/$S-abl24-repeat1-t10mb ] && echo "note: $S repeat1 dir exists (restartable)"; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "abl24|g0r24"); [ "$busy" = 0 ] || echo "note: $busy batch24 jobs still in the queue"
scontrol show reservation $RES | grep -o "EndTime=[^ ]*\|NodeCnt=[^ ]*"
echo "CHECK OK"; k=0
sb () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 "$@") && { echo "submitted $j  ${@: -1}  (${*:3:1})" >&2; echo $j; } || { echo "FAILED: sbatch $*" >&2; echo ""; }; }
acc () { [ -n "$1" ] && k=$((k+1)); }
cfg () { case $1 in
  room)     echo "1 images_2 q14 ''";;     kitchen)  echo "1 images_2 q16 ''";;   counter)  echo "1 images_2 q15 ''";;  bonsai)   echo "1 images_2 q15 ''";;
  garden)   echo "0 images_4 q15 ''";;     bicycle)  echo "0 images_4 q16 ''";;   stump)    echo "0 images_4 q16 ''";;  flowers)  echo "0 images_4 q15 ''";;
  treehill) echo "0 images_4 q16 ''";;     truck)    echo "0 images q18 ',DATA=$D/tandt/truck'";;  train)    echo "0 images q18 ',DATA=$D/tandt/train'";; esac; }
# (a) Mip/T&T third price, 11 jobs (the longest: 20-36 MB meshes), submitted first
for S in room kitchen counter bonsai garden bicycle stump flowers treehill truck train; do
  eval set -- $(cfg $S); B="SCENE=$S,INDOOR=$1,IMAGES=$2,XYZ=$3,SMOKE=0,SELFCHECK=0,SMOKE2=0$4"
  j=$(sb --time=04:30:00 --export=ALL,$B,ABL=constlam,LAMS="0.00062" --job-name=$S-clam3-abl24 $AN/ablation-b24-r2.sbatch); acc $j
done
# (b) repeats of the full method, seeds 1 and 2
for S in room garden truck; do eval set -- $(cfg $S); B="SCENE=$S,INDOOR=$1,IMAGES=$2,XYZ=$3,SMOKE=0,SELFCHECK=0$4"
  for SEED in 1 2; do j=$(sb --time=03:30:00 --export=ALL,$B,ABL=repeat,TARGETS=10,SEED=$SEED --job-name=$S-rep$SEED-abl24 $AN/ablation-b24-r2.sbatch); acc $j; done; done
for SEED in 1 2; do j=$(sb --time=02:30:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,ABL=repeat,TARGETS=2.5,SEED=$SEED --job-name=lego-rep$SEED-abl24b $AN/ablation-b24-r2-blender.sbatch); acc $j; done
# (a) NeRF-Synthetic third price, 8 jobs
for S in lego chair drums ficus hotdog materials mic ship; do
  j=$(sb --time=03:00:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,SMOKE2=0,ABL=constlam,LAMS="0.00185" --job-name=$S-clam3-abl24b $AN/ablation-b24-r2-blender.sbatch); acc $j
done
echo "submitted $k jobs (expected 27: 11 + 6 + 2 + 8), all on the reservation"
squeue -u $USER -o "%.10i %.22j %.2t %.10M %R" | head -40
