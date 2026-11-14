#!/usr/bin/env bash
# codec-time-v1.sh (2026-11-14): encode / decode time of our codec and of zstd, Draco, MPEG TFAN on the SAME mesh per scene: the MeshSplatting
# mesh trained with a vertex cap near 1.08 million (fine-tuned), 11 Mip-NeRF 360 / T&T scenes.  CPU only, 8 threads, about 20 minutes.
# Ours reads <cap run>/axisB/v1.0/deliver/shared-final.pt (what the encoder receives); the generic codecs read <cap run>/axisB/v1.0/finetune/best.pt.
# Rows -> $AN/codec-time.txt.   Run:  sbatch ... codec-time-v1.sh    or directly in an interactive CPU session:  bash codec-time-v1.sh
[ -f /scratch/zr2245/anaconda3/etc/profile.d/conda.sh ] && { source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; }
[ -f /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate ] && source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
AN=${AN:-/scratch/zr2245/analysis}; ROOT=${ROOT:-/scratch/zr2245}; PKG=${PKG:-$AN/vis_collapse-v12}; G=$AN/generic-v1; DRV=$AN/o3dgc-v1/o3dgc_driver; OUT=$AN/codec-time; TAB=$AN/codec-time.txt
SCENES=${SCENES:-"garden room truck kitchen counter bonsai bicycle stump flowers treehill train"}; mkdir -p $OUT
mid () { case $1 in room) echo 1078000;; kitchen) echo 1086000;; counter) echo 1083000;; bonsai|garden) echo 1073000;; bicycle) echo 1080000;; stump) echo 1069000;; flowers) echo 1071000;; treehill) echo 1079000;; truck) echo 1078000;; train) echo 1085000;; esac; }
qp () { case $1 in room) echo 14;; kitchen|bicycle|stump|treehill) echo 16;; counter|bonsai|garden|flowers) echo 15;; truck|train) echo 18;; esac; }
for f in $AN/codec_time_v1.py $AN/generic_time_v1.py $PKG/stage2_run.sh $G/generic_codec_v1.py; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
python -c "import DracoPy, zstandard" 2>/dev/null || pip install DracoPy zstandard > $OUT/pip.log 2>&1 || echo "note: DracoPy / zstandard not installable, those columns stay empty"
DOPT=""; [ -x $DRV ] && DOPT="--driver $DRV" || echo "note: no Open3DGC driver at $DRV, the TFAN column stays empty"
echo "codec-time-v1 | host $(hostname) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs) | threads $OMP_NUM_THREADS | $(date)"
nf=0
for S in $SCENES; do B=$(mid $S); Q=$(qp $S); CG=$ROOT/$S-budget-$B/axisB/v1.0
  if [ -f $CG/deliver/shared-final.pt ]; then
    bash $PKG/stage2_run.sh $AN/codec_time_v1.py --shared $CG/deliver/shared-final.pt --xyz q$Q --dc q7 --out $OUT/$S-ours.json --label "$S cap$B ours" > $OUT/$S-ours.log 2>&1 && grep -h "^TIME" $OUT/$S-ours.log | tee -a $TAB || { echo "FAILED: ours $S"; tail -6 $OUT/$S-ours.log; nf=$((nf+1)); }
  else echo "note: $S: no $CG/deliver/shared-final.pt, skipped"; nf=$((nf+1)); fi
  if [ -f $CG/finetune/best.pt ]; then
    python $AN/generic_time_v1.py --ckpt $CG/finetune/best.pt --qp $Q --bits 8 --out $OUT/$S-generic.json --generic $G $DOPT --label "$S cap$B generic" > $OUT/$S-generic.log 2>&1 && grep -h "^TIME" $OUT/$S-generic.log | tee -a $TAB || { echo "FAILED: generic $S"; tail -6 $OUT/$S-generic.log; nf=$((nf+1)); }
  else echo "note: $S: no $CG/finetune/best.pt, skipped"; nf=$((nf+1)); fi
done
echo "codec-time-v1 exit 0 ($nf failures)  $(date)"
