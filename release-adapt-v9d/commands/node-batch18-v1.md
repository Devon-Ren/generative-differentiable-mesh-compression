# node-batch18-v1 (2026-11-08): one-shot on an unseen dataset, Deep Blending (drjohnson, playroom), nothing changed

**Why this dataset.** Yao's "a person" data (THuman2.0 / ZJU-MoCap) and ScanNet need a loader MeshSplatting does not have and have no compression baseline; Wenhao parked them for the next paper on 11/04. Deep Blending is COLMAP format like Mip-360, the third standard dataset of every 3DGS-compression paper (3DGS, HAC, HAC++ all report drjohnson / playroom), and we have never run it. It was in the HPC plan as B3. One shot, as is: the v9d controller with its sbatch defaults (aim 1.25, 30 rounds), the `--indoor` upstream preset (both scenes are interiors), XYZ q15 (the indoor middle: room q14, counter / bonsai q15, kitchen q16), full-resolution images (3DGS convention for this dataset).

**What runs** (10 jobs, ~12 GPU-h, ~4.5 h wall): per scene one FP32 original with level-1.0 delivery (the reference; its job extracts Metric3D normals first if missing), three adaptive runs at 5 / 10 / 20 MB, and one fixed-lambda job with the 32x and 73x lambdas (quality-on-curve check). Tables: `originals-indoor.txt`, `adapt-v9d.txt`, `rd-prune3i-$SCENE.txt`; run dirs `$SCENE-indoor`, `$SCENE-adapt9d-t$T mb`, `$SCENE-rd3i-lam$L-mu1.0`.

**What to expect, stated now.** Unseen indoor class: landing 1.00-1.12 (the indoor pattern), phase B active, quality within 0.1 dB of the two fixed-lambda points at equal bytes. Published references for the write-up: 3DGS 29.4 dB at ~660 MB on Deep Blending; HAC++ at a few MB (exact numbers to be copied from the HAC++ table).

## 0. Dataset check, download if missing (any node; tandt_db.zip is the 3DGS bundle that also holds T&T)

```bash
D=/scratch/zr2245/datasets; mkdir -p $D
for S in drjohnson playroom; do if [ -f $D/db/$S/sparse/0/cameras.bin ]; then echo "ok   db/$S  $(ls $D/db/$S/images | wc -l) images"; else echo "MISS db/$S"; fi; done
if [ ! -f $D/db/playroom/sparse/0/cameras.bin ]; then
  cd $D && wget -c --tries=5 --timeout=60 https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip && unzip -q -o tandt_db.zip 'db/*' && ls $D/db && echo "Deep Blending unpacked"
fi
for S in drjohnson playroom; do [ -f $D/db/$S/sparse/0/cameras.bin ] && echo "ok db/$S: $(ls $D/db/$S | tr '\n' ' ')" || echo "STILL MISSING db/$S"; done
```

## 1. Submit the 10 jobs (one paste; checks inside the script)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/submit-batch18-v1.sh <<'SUB18'
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
SUB18
bash $AN/submit-batch18-v1.sh
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
```

## 2. After 15-20 min: first lines of the four running jobs

```bash
AN=/scratch/zr2245/analysis
squeue -u $USER -o "%.10i %.22j %.3t %.8M %R"
for f in $AN/origi-*-origi-*.out $AN/adapt9d-*-adapt9d-*.out; do [ -f $f ] || continue; case $f in *drjohnson*|*playroom*) echo "=== $f"; grep -h "images:\|normals\|=== train\|phase A aims\|smoke\|note:\|FAILED" $f | tail -6 | cut -c1-300;; esac; done
```

## 3. When all ten are done (~4.5 h): one file, upload it

```bash
AN=/scratch/zr2245/analysis; OUT=$AN/batch18-results-$(date +%m%d-%H%M).txt
{
echo "===== sacct"; sacct -X -u $USER -S now-12hours -o JobID,JobName%24,State,Elapsed,ExitCode | grep -E "drjohnson|playroom|JobID"
echo "===== originals-indoor.txt (DB rows)"; grep -E "^(drjohnson|playroom) " $AN/originals-indoor.txt
echo "===== adapt-v9d.txt (DB rows)"; grep -E "^(drjohnson|playroom) " $AN/adapt-v9d.txt
for S in drjohnson playroom; do echo "===== rd-prune3i-$S.txt"; cat $AN/rd-prune3i-$S.txt 2>/dev/null; done
echo "===== kappa (DB rows)"; python3 $AN/rdgrowth/kappa_analysis_v1.py '/scratch/zr2245/drjohnson-adapt9d-t*mb' '/scratch/zr2245/playroom-adapt9d-t*mb' 2>&1
echo "===== train logs"; for d in /scratch/zr2245/drjohnson-adapt9d-t*mb /scratch/zr2245/playroom-adapt9d-t*mb; do [ -f $d/train.log ] || continue; echo "--- $d"; grep -h "^\[RD 9500\]\|phase A ended\|RD adapt 1[2-9][0-9]*\] probe\|^\[RD prune [0-9]*\] V\|no collapse this round\|^Training is done" $d/train.log | sed -E 's/\(lambda-prune held; [0-9]+ edges below lam [0-9.e+-]+\)//; s/ \| score q10.*//; s/\[0[0-9]\/11 [0-9:]+\]//' | cut -c1-330; done
echo "===== original train test line"; for S in drjohnson playroom; do echo "--- $S"; grep -h "Evaluating test" /scratch/zr2245/$S-indoor/train.log 2>/dev/null | tail -2; done
echo "===== FAILED lines"; grep -l "FAILED" $AN/origi-*.out $AN/adapt9d-*.out $AN/rd3i-*.out 2>/dev/null | while read f; do case $f in *drjohnson*|*playroom*) echo "--- $f"; grep -h "FAILED" $f | head -3; tail -6 $f | cut -c1-300;; esac; done
} > $OUT 2>&1
echo "wrote $OUT ($(grep -c '' $OUT) lines, $(du -h $OUT | cut -f1))"
```
