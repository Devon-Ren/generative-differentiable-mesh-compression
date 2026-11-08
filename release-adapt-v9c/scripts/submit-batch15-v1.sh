AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9c.py 945 c0d8e4b9af8a371def922ce32223b576
chk $AN/adapt-v9c-blender.sbatch 180 7958c9ab708d016d5761af02263c7007
chk $AN/adapt-v9c.sbatch 172 8e2e55198bf01c6c0a13d159f960b3a0
chk $AN/gate_batch15_v1.py 61 51fc06ca19cbbcb0805b9223a70bb4d5
chk $AN/rdgrowth/train_rd_adapt_v9b.py 932 e379f9906f639d5d69acf127543ce690
chk $AN/adapt-v9b-blender.sbatch 173 c6d80dcb8d3af50754079363ffb96dfa
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
grep -q "version='adapt-v9c'" $AN/rdgrowth/train_rd_adapt_v9c.py && grep -q "max(0.0, slope) \* (horizon_iter - iteration)" $AN/rdgrowth/train_rd_adapt_v9c.py && grep -q "if k_asked > 0 else lam_used" $AN/rdgrowth/train_rd_adapt_v9c.py || { echo "CHECK FAILED: train_rd_adapt_v9c.py is not adapt-v9c"; exit 1; }
grep -q "VERSION v9cb-r1" $AN/adapt-v9c-blender.sbatch && grep -q "VERSION v9c-r1" $AN/adapt-v9c.sbatch || { echo "CHECK FAILED: sbatch files are not v9c-r1 / v9cb-r1"; exit 1; }
grep -q '12500 250 19750' $AN/adapt-v9c.sbatch && grep -q '12500 250 19750' $AN/adapt-v9c-blender.sbatch && grep -q 'OVER=${OVER:-1.25}' $AN/adapt-v9c.sbatch && grep -q 'OVER=${OVER:-1.1}' $AN/adapt-v9c-blender.sbatch || { echo "CHECK FAILED: sbatch defaults (PRUNE every 250, OVER 1.25 / 1.1) missing"; exit 1; }
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9c.py $AN/gate_batch15_v1.py || { echo "CHECK FAILED: python files do not compile"; exit 1; }
bash -n $AN/adapt-v9c-blender.sbatch && bash -n $AN/adapt-v9c.sbatch && bash -n $AN/gate-batch15.sbatch && bash -n $AN/submit-batch15-sweep-v1.sh || { echo "CHECK FAILED: bash syntax"; exit 1; }
for S in lego mic ship; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in room kitchen counter; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for d in /scratch/zr2245/room-adapt9c-t5mb /scratch/zr2245/counter-adapt9c-t5mb /scratch/zr2245/kitchen-adapt9c-t10mb /scratch/zr2245/counter-adapt9c-t10mb /scratch/zr2245/kitchen-adapt9c-t20mb /scratch/zr2245/lego-adapt9cb-t2.5mb /scratch/zr2245/mic-adapt9xb-t2.5mb /scratch/zr2245/ship-adapt9xb-t2.5mb; do [ -e $d ] && echo "note: $d exists (restart: finished stages are skipped)"; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt9|gate1"); [ "$busy" = 0 ] || echo "note: $busy adapt/gate jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0; IDS=""
m () { EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5"; [ "$6" = smoke ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9c $AN/adapt-v9c.sbatch) && { echo "submitted $1 $5 MB $j"; k=$((k+1)); IDS="$IDS:$j"; } || echo "FAILED: sbatch $1 $5"; }
b () { EX="SCENE=$1,XYZ=q14,TARGETS=$2"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9cb $AN/adapt-v9c-blender.sbatch) && { echo "submitted $1 $2 MB $j"; k=$((k+1)); IDS="$IDS:$j"; } || echo "FAILED: sbatch $1 $2"; }
m kitchen 1 images_2 q16 20 nosmoke; m kitchen 1 images_2 q16 10 smoke; m counter 1 images_2 q15 10 nosmoke; m room 1 images_2 q14 5 smoke; m counter 1 images_2 q15 5 nosmoke; b lego 2.5
echo "submitted $k of 6 v9c test jobs (kitchen 20 / kitchen 10 / counter 10 / room 5 first; counter 5 + lego 2.5 as GPUs free up; ~1.7 h wall)"
[ "$k" = 6 ] || { echo "FAILED: only $k test jobs submitted; NOT submitting the gate (run section 4 by hand)"; exit 1; }
g=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --dependency=afterany${IDS} $AN/gate-batch15.sbatch) && echo "gate job $g (after ${IDS#:}; on PASS it submits the 43-job v9c sweep; log $AN/gate-batch15.log)" || echo "FAILED: gate sbatch not accepted -> run section 4 by hand when the 6 jobs are done"
# v9b table completion: mic / ship 2.5 MB with v9b, SMOKE=0 (the smoke stage is what failed); not part of the gate
for S in mic ship; do j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,SCENE=$S,XYZ=q14,TARGETS=2.5,SMOKE=0,SELFCHECK=0 --job-name=$S-2.5-adapt9xb $AN/adapt-v9b-blender.sbatch) && echo "submitted v9b $S 2.5 MB $j (completes adapt-v9b-blender.txt)" || echo "FAILED: sbatch v9b $S 2.5"; done
