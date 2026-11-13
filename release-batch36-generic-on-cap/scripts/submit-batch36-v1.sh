#!/usr/bin/env bash
# submit-batch36-v1.sh (2026-11-13): zstd / Draco / MPEG TFAN on meshes trained with a vertex cap (600,000, about 1.08 million, 3,000,000),
# 11 Mip-NeRF 360 / T&T scenes, SH precision 8 / 6 / 4 bits.  No training: the cap runs exist.  11 GPU jobs (encoding on CPU, one render per point).
AN=/scratch/zr2245/analysis
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/generic_rd_v1.py 46 17c568d6da5cfaf473cfe7067bf38b60
chk $AN/generic-rd-v2.sbatch 70 e7e3bd69c56855efda4dbd86d7c4de7d
bash -n $AN/generic-rd-v2.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
for f in $AN/generic-v1/generic_codec_v1.py $AN/generic-v1/eval_ckpt_v1.py; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
[ -x $AN/o3dgc-v1/o3dgc_driver ] || echo "note: no Open3DGC driver, the TFAN column will be empty"
mid () { case $1 in room) echo 1078000;; kitchen) echo 1086000;; counter) echo 1083000;; bonsai|garden) echo 1073000;; bicycle) echo 1080000;; stump) echo 1069000;; flowers) echo 1071000;; treehill) echo 1079000;; truck) echo 1078000;; train) echo 1085000;; esac; }
SC="garden room truck kitchen counter bonsai bicycle stump flowers treehill train"; ft=0; raw=0; miss=0
for S in $SC; do for B in 600000 $(mid $S) 3000000; do G=/scratch/zr2245/$S-budget-$B
  if [ -f $G/axisB/v1.0/finetune/best.pt ]; then ft=$((ft+1)); elif [ -f $G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then raw=$((raw+1)); echo "note: $S cap $B has no fine-tuned mesh, the trained one will be used"; else miss=$((miss+1)); echo "note: $S cap $B: no mesh found under $G"; fi; done; done
echo "CHECK OK | meshes: $ft fine-tuned, $raw trained only, $miss missing (of 33)"; [ $miss -gt 6 ] && { echo "STOPPED: too many meshes missing, nothing submitted"; exit 1; }
n=0
for S in $SC; do
  j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=8 --export=ALL,SCENE=$S --job-name=$S-gencap-b36 $AN/generic-rd-v2.sbatch) && { n=$((n+1)); echo "submitted $j  $S-gencap-b36"; } || echo "FAILED: sbatch $S"; done
echo "$n GPU jobs submitted"
