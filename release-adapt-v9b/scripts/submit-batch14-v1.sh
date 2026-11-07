AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b.py 932 e379f9906f639d5d69acf127543ce690
chk $AN/adapt-v9b-blender.sbatch 173 c6d80dcb8d3af50754079363ffb96dfa
chk $AN/adapt-v9b.sbatch 165 c0f90652afe97b76d0d3148fe1243f15
chk $AN/gate_batch14_v1.py 54 82b266502b55a9adeacfd192e51c1320
chk $AN/blender/val_select_v3.py 254 129b0aecdac3c7fef7f76d7d996899e4
chk $AN/rdgrowth/probe_bytes_v1.py 66 b03b945dce50345ea404c4f6862b928f
[ -f $AN/rdgrowth/train_rd_adapt_v9.py ] && [ "$(md5sum < $AN/rdgrowth/train_rd_adapt_v9.py | cut -c1-32)" = a120351f64c9472e0353cef29d8ddb04 ] || echo "note: train_rd_adapt_v9.py is not the batch12 file (v9b does not import it; informational)"
grep -q "version='adapt-v9b'" $AN/rdgrowth/train_rd_adapt_v9b.py && grep -q "if k_asked > 0 else lam_used" $AN/rdgrowth/train_rd_adapt_v9b.py || { echo "CHECK FAILED: train_rd_adapt_v9b.py is not adapt-v9b"; exit 1; }
grep -q "max_split_up = max_split" $AN/rdgrowth/train_rd_adapt_v9b.py && grep -q "remaining = max(1, -(-(opt.densify_until_iter - iteration) // opt.densification_interval))" $AN/rdgrowth/train_rd_adapt_v9b.py || { echo "CHECK FAILED: train_rd_adapt_v9b.py lacks the v9 plan fixes"; exit 1; }
grep -q "VERSION v9xb-r1" $AN/adapt-v9b-blender.sbatch && grep -q "VERSION v9x-r1" $AN/adapt-v9b.sbatch || { echo "CHECK FAILED: sbatch files are not v9x-r1 / v9xb-r1"; exit 1; }
grep -q 'OVER=${OVER:-1.25}' $AN/adapt-v9b.sbatch && grep -q 'OVER=${OVER:-1.1}' $AN/adapt-v9b-blender.sbatch && grep -q '12500 500 19500' $AN/adapt-v9b.sbatch || { echo "CHECK FAILED: sbatch defaults (OVER 1.25 / 1.1, PRUNE from 12500) missing"; exit 1; }
python3 -m py_compile $AN/rdgrowth/train_rd_adapt_v9b.py $AN/gate_batch14_v1.py $AN/blender/val_select_v3.py $AN/rdgrowth/probe_bytes_v1.py || { echo "CHECK FAILED: python files do not compile"; exit 1; }
bash -n $AN/adapt-v9b-blender.sbatch && bash -n $AN/adapt-v9b.sbatch && bash -n $AN/gate-batch14.sbatch && bash -n $AN/submit-batch14-sweep-v1.sh || { echo "CHECK FAILED: bash syntax"; exit 1; }
for f in $AN/vis_collapse-v12/stage2_run.sh $AN/vis_collapse-v12/scripts/run_garden.sh /scratch/zr2245/shared-collapse-stage2-v2/STAGE2_VERSION $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/rd_prune.py $AN/rdgrowth/rd_growth_v2.py $AN/blender/train_ckpt_v2.py $AN/coarse_finetune.py /scratch/zr2245/meshsplatting-original-lossless-v1/upstream/train.py; do [ -e $f ] || { echo "CHECK FAILED: MISSING $f"; exit 1; }; done
for S in lego; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in room bicycle garden; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
[ -f $D/tandt/truck/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for truck"; exit 1; }
for d in /scratch/zr2245/room-adapt9x-t10mb /scratch/zr2245/bicycle-adapt9x-t10mb /scratch/zr2245/room-adapt9x-t20mb /scratch/zr2245/garden-adapt9x-t5mb /scratch/zr2245/truck-adapt9x-t10mb /scratch/zr2245/lego-adapt9xb-t2.5mb; do [ -e $d ] && echo "note: $d exists (restart: finished stages are skipped)"; done
[ -f $AN/adapt-v9b.txt ] && echo "note: $AN/adapt-v9b.txt already has $(grep -c '' $AN/adapt-v9b.txt) rows (the gate reads the LAST row per scene/target; fine on a restart)"
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt7|adapt8|adapt9|gate14"); [ "$busy" = 0 ] || echo "note: $busy adapt/gate jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show node gl030 | grep -o "CfgTRES=[^ ]*"
scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0; IDS=""
m () { EX="SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS=$5$6"; [ "$7" = smoke ] || EX="$EX,SMOKE=0,SELFCHECK=0"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=03:30:00 --export=ALL,$EX --job-name=$1-$5-adapt9x $AN/adapt-v9b.sbatch) && { echo "submitted $1 $5 MB $j"; k=$((k+1)); IDS="$IDS:$j"; } || echo "FAILED: sbatch $1 $5"; }
b () { EX="SCENE=$1,XYZ=q14,TARGETS=$2"
  j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 --time=01:45:00 --export=ALL,$EX --job-name=$1-$2-adapt9xb $AN/adapt-v9b-blender.sbatch) && { echo "submitted $1 $2 MB $j"; k=$((k+1)); IDS="$IDS:$j"; } || echo "FAILED: sbatch $1 $2"; }
m room 1 images_2 q14 20 "" nosmoke; m room 1 images_2 q14 10 "" smoke; m bicycle 0 images_4 q16 10 "" nosmoke; m garden 0 images_4 q15 5 "" smoke; m truck 0 images q18 10 ",DATA=$D/tandt/truck" nosmoke; b lego 2.5
echo "submitted $k of 6 test jobs (4 GPUs: room 20 / room 10 / bicycle 10 / garden 5 first; truck 10 + lego 2.5 start as GPUs free up; ~1.5 h wall)"
[ "$k" = 6 ] || { echo "FAILED: only $k test jobs submitted; NOT submitting the gate (run it by hand, see section 4)"; exit 1; }
g=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --dependency=afterany${IDS} $AN/gate-batch14.sbatch) && echo "gate job $g (runs after ${IDS#:}; if it passes it submits the 43-job sweep itself; log $AN/gate-batch14.log)" || echo "FAILED: gate sbatch not accepted (CPU job on the reservation refused?) -> run section 4 by hand when the 6 jobs are done"
