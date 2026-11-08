AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9d.py 970 e950d66563e5eca3bb8aa18ee77eabe2
chk $AN/adapt-v9d-blender.sbatch 189 2b733fad8db12cbbb7094eaa78452ebc
chk $AN/adapt-v9d.sbatch 181 60582922b3d9cca3f089e50f2ead53a8
chk $AN/gate_batch16_v1.py 61 f998c05eb39e39400756af2d30baa375
chk $AN/rdgrowth/rd_prune_v2.py 241 b500c019c4393d4ba8cfd87da100e1d3
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
grep -q "version='adapt-v9d'" $AN/rdgrowth/train_rd_adapt_v9d.py && grep -q "from rd_prune_v2 import" $AN/rdgrowth/train_rd_adapt_v9d.py && grep -q "kept_v=kept_v, kept_f=kept_f" $AN/rdgrowth/train_rd_adapt_v9d.py && grep -q "max(0.0, slope) \* (horizon_iter - iteration)" $AN/rdgrowth/train_rd_adapt_v9d.py || { echo "CHECK FAILED: train_rd_adapt_v9d.py is not adapt-v9d"; exit 1; }
grep -q "PRUNE_VERSION = 'rd_prune_v2'" $AN/rdgrowth/rd_prune_v2.py && grep -q "kept_v=None, kept_f=None" $AN/rdgrowth/rd_prune_v2.py || { echo "CHECK FAILED: rd_prune_v2.py missing / wrong"; exit 1; }
grep -q "VERSION v9db-r1" $AN/adapt-v9d-blender.sbatch && grep -q "VERSION v9d-r1" $AN/adapt-v9d.sbatch || { echo "CHECK FAILED: sbatch files are not v9d-r1 / v9db-r1"; exit 1; }
grep -q '12500 250 19750' $AN/adapt-v9d.sbatch && grep -q '12500 250 19750' $AN/adapt-v9d-blender.sbatch && grep -q 'OVER=${OVER:-1.25}' $AN/adapt-v9d.sbatch && grep -q 'OVER=${OVER:-1.1}' $AN/adapt-v9d-blender.sbatch || { echo "CHECK FAILED: sbatch defaults (PRUNE every 250, OVER 1.25 / 1.1) missing"; exit 1; }
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9d.py $AN/rdgrowth/rd_prune_v2.py $AN/gate_batch16_v1.py || { echo "CHECK FAILED: python files do not compile"; exit 1; }
bash -n $AN/adapt-v9d-blender.sbatch && bash -n $AN/adapt-v9d.sbatch && bash -n $AN/gate-batch16.sbatch && bash -n $AN/submit-batch16-sweep-v1.sh || { echo "CHECK FAILED: bash syntax"; exit 1; }
for S in lego; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in room kitchen counter; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for d in /scratch/zr2245/room-adapt9d-t5mb /scratch/zr2245/counter-adapt9d-t5mb /scratch/zr2245/kitchen-adapt9d-t10mb /scratch/zr2245/counter-adapt9d-t10mb /scratch/zr2245/kitchen-adapt9d-t20mb /scratch/zr2245/lego-adapt9db-t2.5mb; do [ -e $d ] && echo "note: $d exists (restart: finished stages are skipped)"; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt9|gate1"); [ "$busy" = 0 ] || echo "note: $busy adapt/gate jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0; IDS=""
m () { EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5"; [ "$6" = smoke ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9d $AN/adapt-v9d.sbatch) && { echo "submitted $1 $5 MB $j"; k=$((k+1)); IDS="$IDS:$j"; } || echo "FAILED: sbatch $1 $5"; }
b () { EX="SCENE=$1,XYZ=q14,TARGETS=$2"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9db $AN/adapt-v9d-blender.sbatch) && { echo "submitted $1 $2 MB $j"; k=$((k+1)); IDS="$IDS:$j"; } || echo "FAILED: sbatch $1 $2"; }
m kitchen 1 images_2 q16 20 nosmoke; m kitchen 1 images_2 q16 10 smoke; m counter 1 images_2 q15 10 nosmoke; m room 1 images_2 q14 5 smoke; m counter 1 images_2 q15 5 nosmoke; b lego 2.5
echo "submitted $k of 6 v9d test jobs (kitchen 20 / kitchen 10 / counter 10 / room 5 first; counter 5 + lego 2.5 as GPUs free up; ~1.7 h wall)"
[ "$k" = 6 ] || { echo "FAILED: only $k test jobs submitted; NOT submitting the gate (run section 4 by hand)"; exit 1; }
g=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --dependency=afterany${IDS} $AN/gate-batch16.sbatch) && echo "gate job $g (after ${IDS#:}; on PASS it submits the 43-job v9d sweep; log $AN/gate-batch16.log)" || echo "FAILED: gate sbatch not accepted -> run section 4 by hand when the 6 jobs are done"
