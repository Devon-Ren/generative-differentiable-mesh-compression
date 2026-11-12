#!/usr/bin/env bash
# submit-batch29-v2.sh (2026-11-12): pilot of the byte-targeted, fold-over-safe trimming (10 jobs, public partition, about 8 GPU-h).
#   feas    : bytes wanted per round = excess / rounds left            feasq : excess / min(rounds left, 3)
#   feas25 / feasq25 : same with growth aim 1.25 (objects)             feasq35 : feasq with growth aim 1.35
# Reference (current method): kitchen 10 MB +13.5%, drjohnson 10 +12.0%, playroom 10 +8.3%, room 5 +6.1%, garden 10 +2.0%, treehill 20 -6.6%,
#   hotdog 2.5 +9.2% (aim 1.25: +20.6%), lego 2.5 -0.2% (aim 1.25: +11.4%).
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl4.py 1053 c36e97b1e56f17e51291bec76833c3fe
chk $AN/ablation-b24-r5.sbatch 213 0947c52ac846f35b14141dcd0efa9db8
chk $AN/ablation-b24-r5-blender.sbatch 215 792b10a43264e763ea2882bbba29a488
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b_abl4.py && bash -n $AN/ablation-b24-r5.sbatch && bash -n $AN/ablation-b24-r5-blender.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
grep -q "^def _face_normals" $AN/rdgrowth/rd_prune.py && grep -q "^def edge_table" $AN/rdgrowth/rd_prune.py || { echo "CHECK FAILED: $AN/rdgrowth/rd_prune.py lacks _face_normals / edge_table"; exit 1; }
echo "CHECK OK"
sb () { j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 "$@") && echo "submitted $j  (${*:3:1})" || echo "FAILED: sbatch $*"; }
C="SMOKE=0,SELFCHECK=0"
for A in feasq feas feasq35; do sb --time=02:30:00 --export=ALL,SCENE=kitchen,INDOOR=1,IMAGES=images_2,XYZ=q16,$C,ABL=$A,TARGETS=10 --job-name=kitchen-$A-b29 $AN/ablation-b24-r5.sbatch; done
for A in feasq25 feas25; do sb --time=02:30:00 --export=ALL,SCENE=hotdog,XYZ=q14,INDOOR=1,$C,ABL=$A,TARGETS=2.5 --job-name=hotdog-$A-b29 $AN/ablation-b24-r5-blender.sbatch; done
sb --time=02:30:00 --export=ALL,SCENE=drjohnson,INDOOR=1,IMAGES=images,XYZ=q15,$C,DATA=$D/db/drjohnson,ABL=feasq,TARGETS=10 --job-name=drjohnson-feasq-b29 $AN/ablation-b24-r5.sbatch
sb --time=02:30:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,$C,ABL=feasq25,TARGETS=2.5 --job-name=lego-feasq25-b29 $AN/ablation-b24-r5-blender.sbatch
sb --time=02:30:00 --export=ALL,SCENE=treehill,INDOOR=0,IMAGES=images_4,XYZ=q16,$C,ABL=feasq35,TARGETS=20 --job-name=treehill-feasq35-b29 $AN/ablation-b24-r5.sbatch
sb --time=02:30:00 --export=ALL,SCENE=room,INDOOR=1,IMAGES=images_2,XYZ=q14,$C,ABL=feasq,TARGETS=5 --job-name=room-feasq-b29 $AN/ablation-b24-r5.sbatch
sb --time=02:00:00 --export=ALL,SCENE=garden,INDOOR=0,IMAGES=images_4,XYZ=q15,$C,ABL=feasq,TARGETS=10 --job-name=garden-feasq-b29 $AN/ablation-b24-r5.sbatch
squeue -u $USER -o "%.10i %.26j %.2t %.10M %R" | grep -E "b29|JOBID"
