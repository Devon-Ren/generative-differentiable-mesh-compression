#!/usr/bin/env bash
# submit-batch27-v1.sh (2026-11-12): the three intuitive size-control baselines on ONE scene per dataset (garden, truck, playroom at T = 10 MB;
# lego at T = 2.5 MB), public partition (fairshare), 12 GPU runs, about 10 GPU-h.
#   cap1  : MeshSplatting with a vertex cap chosen from a prior (file MB = a * cap + c fitted on the Mip-360 / T&T cap runs, the test scene left out)
#   cap2  : measure the cap1 file, rescale the cap, train and encode again (submitted automatically by cap-pass2 after cap1)
#   adapt : "update the cap as you go": the byte probe + growth plan set the number of splits per step, MeshSplatting's importance sampling
#           picks the triangles, no collapse (ablation-b24-r3 sbatch, ABL=adaptcap)
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/ablation-b24-r3.sbatch 204 949703fb20c0b3c5ba58a8d3e53eb21c
chk $AN/ablation-b24-r3-blender.sbatch 206 69614021a3aaf74cc7f999697ed8be54
chk $AN/rdgrowth/train_rd_adapt_v9b_abl2.py 938 75e92ffc397018fe472fc97801d0b9f9
chk $AN/budget-sweep-v2.sbatch 82 247b78f2bad723ecd7cae26e236076d9
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
chk $AN/budget-sweep-blender-v1.sbatch 105 d720b8ccba85622f90f25ee0b52b77ee
chk $AN/cap-pass2-v1.sbatch 26 63398b4a79453d26734ccfe0bcf2fb32
bash -n $AN/ablation-b24-r3.sbatch && bash -n $AN/ablation-b24-r3-blender.sbatch && bash -n $AN/cap-pass2-v1.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
[ -f /scratch/zr2245/playroom-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no playroom --indoor original (cap job needs it)"; exit 1; }
echo "CHECK OK"; k=0
sb () { j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 "$@") && { echo "submitted $j  ${@: -1}  (${*:3:1})" >&2; k=$((k+1)); echo $j; } || { echo "FAILED: sbatch $*" >&2; echo ""; }; }
p2 () { [ -n "$1" ] && sbatch --parsable --dependency=afterok:$1 --export=ALL,S=$2,CAP1=$3,T=$4 --job-name=$2-pass2 $AN/cap-pass2-v1.sbatch >/dev/null && echo "  pass-2 launcher queued for $2 (after $1)"; }
# cap1 (prior-chosen caps: garden 1105000, truck 1078000, playroom 1081000, lego 53000) + pass-2 launchers
j=$(sb --time=02:00:00 --export=ALL,SCENE=garden,IMAGES=images_4,XYZ=q15,INDOOR=0,BUDGETS=1105000 --job-name=garden-cap1-b27 $AN/budget-sweep-v2.sbatch); p2 "$j" garden 1105000 10
j=$(sb --time=02:00:00 --export=ALL,SCENE=truck,IMAGES=images,XYZ=q18,INDOOR=0,DATA=$D/tandt/truck,BUDGETS=1078000 --job-name=truck-cap1-b27 $AN/budget-sweep-v2.sbatch); p2 "$j" truck 1078000 10
j=$(sb --time=02:30:00 --export=ALL,SCENE=playroom,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/playroom,BUDGETS=1081000 --job-name=playroom-cap1-b27 $AN/budget-sweep-v3.sbatch); p2 "$j" playroom 1081000 10
j=$(sb --time=02:00:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,LAMS_AUTO=" ",BUDGETS=53000 --job-name=lego-cap1-b27 $AN/budget-sweep-blender-v1.sbatch); p2 "$j" lego 53000 2.5
# adaptive cap
sb --time=02:00:00 --export=ALL,SCENE=garden,INDOOR=0,IMAGES=images_4,XYZ=q15,SMOKE=0,SELFCHECK=0,ABL=adaptcap,TARGETS=10 --job-name=garden-adapt-b27 $AN/ablation-b24-r3.sbatch >/dev/null
sb --time=02:00:00 --export=ALL,SCENE=truck,INDOOR=0,IMAGES=images,XYZ=q18,SMOKE=0,SELFCHECK=0,DATA=$D/tandt/truck,ABL=adaptcap,TARGETS=10 --job-name=truck-adapt-b27 $AN/ablation-b24-r3.sbatch >/dev/null
sb --time=02:30:00 --export=ALL,SCENE=playroom,INDOOR=1,IMAGES=images,XYZ=q15,SMOKE=0,SELFCHECK=0,DATA=$D/db/playroom,ABL=adaptcap,TARGETS=10 --job-name=playroom-adapt-b27 $AN/ablation-b24-r3.sbatch >/dev/null
sb --time=02:00:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,ABL=adaptcap,TARGETS=2.5 --job-name=lego-adapt-b27 $AN/ablation-b24-r3-blender.sbatch >/dev/null
echo "8 GPU jobs submitted now (4 cap1 + 4 adaptcap); 4 cap2 jobs are submitted by the pass-2 launchers when each cap1 finishes"
squeue -u $USER -o "%.10i %.22j %.2t %.10M %R" | head -20
