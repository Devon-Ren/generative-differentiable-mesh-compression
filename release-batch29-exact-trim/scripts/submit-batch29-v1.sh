#!/usr/bin/env bash
# submit-batch29-v1.sh (2026-11-12): pilot of exact trimming (12 jobs, 13 runs, public partition, about 10 GPU-h).
#   exact   : collapse rounds deliver the planned pairs exactly, default growth aim (1.25 captured, 1.1 objects)
#   exact25 : same, growth aim 1.25 (objects)          exact35 : same, growth aim 1.35          exactfl : front-loaded variant, default aim
# Reference (current method): kitchen 10 MB +13.5%, garden 10 +2.0%, treehill 20 -6.6%, room 5 +6.1%, playroom 10 +8.3%, drjohnson 10 +12.0%,
#   lego 2.5 -0.2% (aim 1.25: +11.4%), hotdog 2.5 +9.2% (aim 1.25: +20.6%), mic 2.5 -6.6% (aim 1.25: -6.2%), lego 1.5 +1.1%.
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl3.py 962 a30a25faa3c204d0551604ce78255122
chk $AN/ablation-b24-r4.sbatch 208 d16aa0d2f6fd254fa0b680c1f0c9043a
chk $AN/ablation-b24-r4-blender.sbatch 210 79f9ede7b9b0207563c471b486fd965c
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b_abl3.py && bash -n $AN/ablation-b24-r4.sbatch && bash -n $AN/ablation-b24-r4-blender.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
grep -q "n_pass_pairs=n_picked" $AN/rdgrowth/rd_prune.py && grep -q "n_flip_rejected=n_flip" $AN/rdgrowth/rd_prune.py || { echo "CHECK FAILED: $AN/rdgrowth/rd_prune.py does not report n_pass_pairs / n_flip_rejected"; exit 1; }
for G in kitchen-abl24-exact-t10mb kitchen-abl24-exact35-t10mb kitchen-abl24-exactfl-t10mb garden-abl24-exact35-t10mb treehill-abl24-exact35-t20mb room-abl24-exact35-t5mb playroom-abl24-exact-t10mb drjohnson-abl24-exact-t10mb lego-abl24b-exact25-t2.5mb hotdog-abl24b-exact25-t2.5mb mic-abl24b-exact25-t2.5mb hotdog-abl24b-exact-t2.5mb; do [ -e /scratch/zr2245/$G ] && echo "note: /scratch/zr2245/$G already exists (finished stages are skipped)"; done
echo "CHECK OK"
sb () { j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 "$@") && echo "submitted $j  (${*:3:1})" || echo "FAILED: sbatch $*"; }
C="SMOKE=0,SELFCHECK=0"
for A in exact exact35 exactfl; do sb --time=02:30:00 --export=ALL,SCENE=kitchen,INDOOR=1,IMAGES=images_2,XYZ=q16,$C,ABL=$A,TARGETS=10 --job-name=kitchen-$A-b29 $AN/ablation-b24-r4.sbatch; done
sb --time=02:00:00 --export=ALL,SCENE=garden,INDOOR=0,IMAGES=images_4,XYZ=q15,$C,ABL=exact35,TARGETS=10 --job-name=garden-exact35-b29 $AN/ablation-b24-r4.sbatch
sb --time=02:30:00 --export=ALL,SCENE=treehill,INDOOR=0,IMAGES=images_4,XYZ=q16,$C,ABL=exact35,TARGETS=20 --job-name=treehill-exact35-b29 $AN/ablation-b24-r4.sbatch
sb --time=02:30:00 --export=ALL,SCENE=room,INDOOR=1,IMAGES=images_2,XYZ=q14,$C,ABL=exact35,TARGETS=5 --job-name=room-exact35-b29 $AN/ablation-b24-r4.sbatch
for S in playroom drjohnson; do sb --time=02:30:00 --export=ALL,SCENE=$S,INDOOR=1,IMAGES=images,XYZ=q15,$C,DATA=$D/db/$S,ABL=exact,TARGETS=10 --job-name=$S-exact-b29 $AN/ablation-b24-r4.sbatch; done
sb --time=04:00:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,$C,ABL=exact25,TARGETS="2.5 1.5" --job-name=lego-exact25-b29 $AN/ablation-b24-r4-blender.sbatch
for S in hotdog mic; do sb --time=02:30:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,$C,ABL=exact25,TARGETS=2.5 --job-name=$S-exact25-b29 $AN/ablation-b24-r4-blender.sbatch; done
sb --time=02:30:00 --export=ALL,SCENE=hotdog,XYZ=q14,INDOOR=1,$C,ABL=exact,TARGETS=2.5 --job-name=hotdog-exact-b29 $AN/ablation-b24-r4-blender.sbatch
squeue -u $USER -o "%.10i %.26j %.2t %.10M %R" | grep -E "b29|JOBID"
