#!/usr/bin/env bash
# submit-batch38-v1.sh (2026-11-14): zstd / Draco / MPEG TFAN on the 8 NeRF-Synthetic objects: the model trained at the default cap and three
# cap-trained meshes per object (existing cap runs), SH precision 8 / 6 / 4 bits.  No training.  8 GPU jobs of a few minutes.
AN=/scratch/zr2245/analysis
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/generic_rd_v1.py 46 17c568d6da5cfaf473cfe7067bf38b60
chk $AN/generic-rd-v3.sbatch 72 88b8342f718d39217f76889717b6a6e9
bash -n $AN/generic-rd-v3.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
caps () { case $1 in lego) echo "78124 183000 372025";; chair) echo "66755 178000 238956";; drums) echo "68000 176000 258000";; ficus) echo "35557 67614 171000";;
  hotdog) echo "47578 177000 222000";; materials) echo "68793 176000 259000";; mic) echo "46659 83900 174000";; ship) echo "74000 177000 294000";; esac; }
SC="lego chair drums ficus hotdog materials mic ship"; ft=0; raw=0; miss=0; full=0
for S in $SC; do [ -f /scratch/zr2245/$S-blender4-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && full=$((full+1)) || echo "note: $S: no default-cap model under /scratch/zr2245/$S-blender4-best"
  for B in $(caps $S); do G=/scratch/zr2245/$S-budgetb-$B-best
  if [ -f $G/axisB/v1.0/finetune/best.pt ]; then ft=$((ft+1)); elif [ -f $G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ]; then raw=$((raw+1)); echo "note: $S cap $B has no fine-tuned mesh, the trained one will be used"; else miss=$((miss+1)); echo "note: $S cap $B: no mesh found under $G"; fi; done; done
echo "CHECK OK | default-cap models $full of 8 | cap meshes: $ft fine-tuned, $raw trained only, $miss missing (of 24)"; [ $miss -gt 6 ] || [ $full -lt 8 ] && { echo "STOPPED: meshes missing, nothing submitted"; exit 1; }
n=0
for S in $SC; do
  j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=8 --time=00:30:00 --export=ALL,SCENE=$S,CAPS="full $(caps $S)" --job-name=$S-genobj-b38 $AN/generic-rd-v3.sbatch) && { n=$((n+1)); echo "submitted $j  $S-genobj-b38"; } || echo "FAILED: sbatch $S"; done
echo "$n GPU jobs submitted"
