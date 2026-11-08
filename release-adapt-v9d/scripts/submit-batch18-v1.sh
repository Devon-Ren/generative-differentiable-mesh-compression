AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
for f in $AN/originals-v2.sbatch $AN/rd-prune-v3c.sbatch $AN/adapt-v9d.sbatch $AN/rdgrowth/train_rd_adapt_v9d.py $AN/rdgrowth/rd_prune_v2.py $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/probe_bytes_v1.py; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
grep -q "VERSION v2" $AN/originals-v2.sbatch && grep -q "VERSION v3c" $AN/rd-prune-v3c.sbatch && grep -q "VERSION v9d-r1" $AN/adapt-v9d.sbatch || { echo "CHECK FAILED: sbatch versions (originals v2 / rd-prune v3c / adapt v9d-r1)"; exit 1; }
[ "$(md5sum < $AN/originals-v2.sbatch | cut -c1-32)" = 5cf0d13a0291ade6138b01b3fe5d71fa ] || echo "note: originals-v2.sbatch md5 differs from the 11/01 paste (edited since?); VERSION v2 is present, continuing"
[ "$(md5sum < $AN/rd-prune-v3c.sbatch | cut -c1-32)" = 43ec7a3736344ad720e19f4186f0eb30 ] || echo "note: rd-prune-v3c.sbatch md5 differs from the 11/01 paste (edited since?); VERSION v3c is present, continuing"
grep -q "version='adapt-v9d'" $AN/rdgrowth/train_rd_adapt_v9d.py && grep -q "PRUNE_VERSION = 'rd_prune_v2'" $AN/rdgrowth/rd_prune_v2.py || { echo "CHECK FAILED: v9d trainer / rd_prune_v2 wrong"; exit 1; }
bash -n $AN/originals-v2.sbatch && bash -n $AN/rd-prune-v3c.sbatch && bash -n $AN/adapt-v9d.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
for S in drjohnson playroom; do [ -f $D/db/$S/sparse/0/cameras.bin ] && [ -d $D/db/$S/images ] || { echo "CHECK FAILED: Deep Blending scene $S missing at $D/db/$S (run section 0)"; exit 1; }; echo "$S: $(ls $D/db/$S/images | wc -l) images, normals: $(ls $D/db/$S/normals 2>/dev/null | wc -l) (0 = the originals job extracts them first, ~10-15 min)"; done
for d in /scratch/zr2245/drjohnson-indoor /scratch/zr2245/playroom-indoor /scratch/zr2245/drjohnson-adapt9d-t5mb /scratch/zr2245/playroom-adapt9d-t5mb; do [ -e $d ] && echo "note: $d exists (restart: finished stages are skipped)"; done
busy=$(squeue -u $USER -h -o "%j" | grep -cE "adapt9|origi|rd3i|gate1"); [ "$busy" = 0 ] || echo "note: $busy related jobs still in the queue"
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0
sb () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 "$@") && { echo "submitted $j  ${@: -1}  (${*:3:1})"; k=$((k+1)); } || echo "FAILED: sbatch $*"; }
for S in drjohnson playroom; do
  # (a) FP32 original with --indoor + level-1.0 delivery (the reference the ratio and dPSNR are computed against); XYZ q15 = the indoor middle (room q14, counter/bonsai q15, kitchen q16)
  sb --time=03:30:00 --export=ALL,SCENE=$S,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/$S --job-name=$S-origi $AN/originals-v2.sbatch
  # (b) adaptive v9d, 5 MB (smoke on; self-check will be skipped: no earlier delivery for this scene)
  sb --time=03:30:00 --export=ALL,SCENE=$S,INDOOR=1,IMAGES=images,XYZ=q15,TARGETS=5,DATA=$D/db/$S --job-name=$S-5-adapt9d $AN/adapt-v9d.sbatch
done
for S in drjohnson playroom; do for T in 10 20; do
  sb --time=03:30:00 --export=ALL,SCENE=$S,INDOOR=1,IMAGES=images,XYZ=q15,TARGETS=$T,DATA=$D/db/$S,SMOKE=0,SELFCHECK=0 --job-name=$S-$T-adapt9d $AN/adapt-v9d.sbatch
done; done
for S in drjohnson playroom; do
  # (c) two fixed-lambda points (the 32x and 73x lambdas) for the quality-on-curve check
  sb --time=03:30:00 --export=ALL,SCENE=$S,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/$S,LAMS="0.00185 0.0062" --job-name=$S-rd3i $AN/rd-prune-v3c.sbatch
done
echo "submitted $k of 10 (first wave: 2 originals + 2 adaptive 5 MB; then adaptive 10/20; then the fixed-lambda pairs; ~12 GPU-h, ~4.5 h wall on 4 GPUs)"
