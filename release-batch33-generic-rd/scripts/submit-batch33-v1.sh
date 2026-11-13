#!/usr/bin/env bash
# submit-batch33-v1.sh (2026-11-13): rate-distortion points of the generic codecs on the 11 Mip-NeRF 360 / T&T scenes (11 GPU jobs, about 10 GPU-h;
# most of each job is CPU encoding).  Per scene: 4 meshes (FP32 model; simplified after training + fine-tuned at three vertex counts) x SH precision 8 / 6 / 4 bits.
AN=/scratch/zr2245/analysis
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/generic_rd_v1.py 46 17c568d6da5cfaf473cfe7067bf38b60
chk $AN/generic-rd-v1.sbatch 68 a9ad66a9e7bd9a740b82f5b7bff42681
python3 -m py_compile $AN/generic_rd_v1.py && bash -n $AN/generic-rd-v1.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
for f in $AN/generic-v1/generic_codec_v1.py $AN/generic-v1/eval_ckpt_v1.py; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
[ -x $AN/o3dgc-v1/o3dgc_driver ] || echo "note: no Open3DGC driver, the TFAN column will be empty"
miss=0; for S in room kitchen counter bonsai; do for B in $(case $S in room) echo 2198000 1020000 477000;; kitchen) echo 1895000 928000 409000;; counter) echo 2040000 978000 449000;; bonsai) echo 2090000 1005000 473000;; esac); do [ -f /scratch/zr2245/$S-indoor/axisB/g0ctrl-$B/finetune/best.pt ] || { echo "note: simplified mesh missing: $S $B"; miss=$((miss+1)); }; done; done
for S in garden truck; do [ -f /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no original for $S"; exit 1; }; done
echo "CHECK OK ($miss simplified meshes missing among the indoor scenes)"; n=0
for S in garden room truck kitchen counter bonsai bicycle stump flowers treehill train; do
  j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=8 --export=ALL,SCENE=$S --job-name=$S-genrd-b33 $AN/generic-rd-v1.sbatch) && { n=$((n+1)); echo "submitted $j  $S-genrd-b33"; } || echo "FAILED: sbatch $S"; done
echo "$n GPU jobs submitted"
