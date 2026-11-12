#!/usr/bin/env bash
# submit-batch28-v1.sh (2026-11-12): how stable is a vertex cap, and fair cap baselines (public partition / fairshare; 11 GPU runs now, up to 3 retries).
#   A  cap stability   : caps within 0.1% of a cap that was already run (garden 1,105,000 and 1,073,000; playroom 1,081,000).  5 runs.
#   B  fair object cap : lego 183,000 and chair 178,000 at 2.5 MB (cap from the NeRF-Synthetic size prior, the object left out) + conditional retry.  2 runs.
#   C  indoor cap      : kitchen 1,086,000 at 10 MB (cap from the Mip-360 / T&T size prior, kitchen left out) + conditional retry.  1 run.
#   D  one growth aim  : BudgetMesh on lego, hotdog, mic at 2.5 MB with the captured-scene aim 1.25 instead of 1.1 (seed 0, rows tagged abl24-repeat-s0).  3 runs.
# Usage: bash submit-batch28-v1.sh [GROUPS]   (default "A B C D")
AN=/scratch/zr2245/analysis; D=/scratch/zr2245/datasets; GROUPS_=${1:-"A B C D"}
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/budget-sweep-v2.sbatch 82 247b78f2bad723ecd7cae26e236076d9
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
chk $AN/budget-sweep-blender-v1.sbatch 105 d720b8ccba85622f90f25ee0b52b77ee
chk $AN/ablation-b24-r3-blender.sbatch 206 69614021a3aaf74cc7f999697ed8be54
chk $AN/rdgrowth/train_rd_adapt_v9b_abl2.py 938 75e92ffc397018fe472fc97801d0b9f9
chk $AN/cap-pass2-v2.sbatch 36 57ceb08062ae905438f3a607963cc428
bash -n $AN/cap-pass2-v2.sbatch || { echo "CHECK FAILED: syntax"; exit 1; }
for S in kitchen playroom; do [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no --indoor original for $S"; exit 1; }; done
[ -f /scratch/zr2245/garden-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no garden original"; exit 1; }
for S in lego chair; do [ -f /scratch/zr2245/$S-blender4-best/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no $S-blender4-best original"; exit 1; }; done
for G in garden-budget-1104000 garden-budget-1106000 garden-budget-1074000 playroom-budget-1080000 playroom-budget-1082000 lego-budget-183000 chair-budget-178000 kitchen-budget-1086000 lego-abl24b-repeat0-t2.5mb hotdog-abl24b-repeat0-t2.5mb mic-abl24b-repeat0-t2.5mb; do [ -e /scratch/zr2245/$G ] && echo "note: /scratch/zr2245/$G already exists (finished stages are skipped)"; done
echo "CHECK OK (groups: $GROUPS_)"; k=0
sb () { j=$(sbatch --parsable --account=torch_pr_38_general --cpus-per-task=6 "$@") && { echo "submitted $j  ${@: -1}  (${*:3:1})" >&2; echo $j; } || { echo "FAILED: sbatch $*" >&2; echo ""; }; }
p2 () { [ -n "$1" ] && sbatch --parsable --dependency=afterok:$1 --export=ALL,S=$2,CAP1=$3,T=$4,C=$5 --job-name=$2-pass2v2 $AN/cap-pass2-v2.sbatch >/dev/null && echo "  retry launcher queued for $2 (after $1)"; }
for G in $GROUPS_; do case $G in
A) for B in 1104000 1106000 1074000; do sb --time=02:00:00 --export=ALL,SCENE=garden,IMAGES=images_4,XYZ=q15,INDOOR=0,BUDGETS=$B --job-name=garden-cap$B-b28 $AN/budget-sweep-v2.sbatch >/dev/null; done
   for B in 1080000 1082000; do sb --time=02:30:00 --export=ALL,SCENE=playroom,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/playroom,BUDGETS=$B --job-name=playroom-cap$B-b28 $AN/budget-sweep-v3.sbatch >/dev/null; done;;
B) j=$(sb --time=02:00:00 --export=ALL,SCENE=lego,XYZ=q14,INDOOR=1,LAMS_AUTO=" ",BUDGETS=183000 --job-name=lego-cap1-b28 $AN/budget-sweep-blender-v1.sbatch); p2 "$j" lego 183000 2.5 0.95
   j=$(sb --time=02:00:00 --export=ALL,SCENE=chair,XYZ=q14,INDOOR=1,LAMS_AUTO=" ",BUDGETS=178000 --job-name=chair-cap1-b28 $AN/budget-sweep-blender-v1.sbatch); p2 "$j" chair 178000 2.5 0.88;;
C) j=$(sb --time=03:00:00 --export=ALL,SCENE=kitchen,IMAGES=images_2,XYZ=q16,INDOOR=1,BUDGETS=1086000 --job-name=kitchen-cap1-b28 $AN/budget-sweep-v3.sbatch); p2 "$j" kitchen 1086000 10 2.08;;
D) for S in lego hotdog mic; do sb --time=02:30:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,SMOKE=0,SELFCHECK=0,ABL=repeat,TARGETS=2.5,SEED=0,OVER=1.25 --job-name=$S-aim125-b28 $AN/ablation-b24-r3-blender.sbatch >/dev/null; done;;
*) echo "FAILED: unknown group $G";;
esac; done
squeue -u $USER -o "%.10i %.26j %.2t %.10M %R" | grep -E "b28|pass2v2|JOBID"
