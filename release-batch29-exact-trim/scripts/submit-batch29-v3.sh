#!/usr/bin/env bash
# submit-batch29-v3.sh (2026-11-12): pilot of the ratio-form size prediction (12 jobs, public partition, about 8 GPU-h).
#   ratio / ratio25 / ratio35 : fold-over-safe byte-targeted trimming + ratio-form prediction, rounds act from 16,000; growth aim default / 1.25 / 1.35
# Second version, for reference: kitchen -1.3%, garden +1.2%, room -1.5%, treehill -0.5% (aim 1.35); drjohnson -13.0%, hotdog -17.3% (aim 1.25).
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl5.py 1078 3f1f876e372659b9dd8ad0a07610497d
chk $AN/ablation-b24-r6.sbatch 216 4625b51e1cb8f7086c228fda3ec69fc7
chk $AN/ablation-b24-r6-blender.sbatch 218 c7e8a016bcabf0391204c07936ceab15
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b_abl5.py && bash -n $AN/ablation-b24-r6.sbatch && bash -n $AN/ablation-b24-r6-blender.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
echo "CHECK OK"
sb () { j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 "$@") && echo "submitted $j  (${*:3:1})" || echo "FAILED: sbatch $*"; }
C="SMOKE=0,SELFCHECK=0"
for S in drjohnson playroom; do sb --time=02:30:00 --export=ALL,SCENE=$S,INDOOR=1,IMAGES=images,XYZ=q15,$C,DATA=$D/db/$S,ABL=ratio35,TARGETS=10 --job-name=$S-ratio35-b29 $AN/ablation-b24-r6.sbatch; done
for S in hotdog lego; do for A in ratio35 ratio25; do sb --time=02:30:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,$C,ABL=$A,TARGETS=2.5 --job-name=$S-$A-b29 $AN/ablation-b24-r6-blender.sbatch; done; done
for A in ratio35 ratio; do sb --time=02:30:00 --export=ALL,SCENE=kitchen,INDOOR=1,IMAGES=images_2,XYZ=q16,$C,ABL=$A,TARGETS=10 --job-name=kitchen-$A-b29 $AN/ablation-b24-r6.sbatch; done
sb --time=02:30:00 --export=ALL,SCENE=mic,XYZ=q14,INDOOR=1,$C,ABL=ratio35,TARGETS=2.5 --job-name=mic-ratio35-b29 $AN/ablation-b24-r6-blender.sbatch
sb --time=02:30:00 --export=ALL,SCENE=room,INDOOR=1,IMAGES=images_2,XYZ=q14,$C,ABL=ratio35,TARGETS=5 --job-name=room-ratio35-b29 $AN/ablation-b24-r6.sbatch
sb --time=02:30:00 --export=ALL,SCENE=treehill,INDOOR=0,IMAGES=images_4,XYZ=q16,$C,ABL=ratio35,TARGETS=20 --job-name=treehill-ratio35-b29 $AN/ablation-b24-r6.sbatch
sb --time=02:00:00 --export=ALL,SCENE=garden,INDOOR=0,IMAGES=images_4,XYZ=q15,$C,ABL=ratio35,TARGETS=10 --job-name=garden-ratio35-b29 $AN/ablation-b24-r6.sbatch
squeue -u $USER -o "%.10i %.26j %.2t %.10M %R" | grep -E "b29|JOBID"
