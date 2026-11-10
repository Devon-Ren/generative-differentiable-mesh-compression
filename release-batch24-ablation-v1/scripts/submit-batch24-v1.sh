#!/usr/bin/env bash
# submit-batch24-v1.sh (2026-11-10): the training-rule ablation on all 11 Mip-360 / T&T scenes + the vertex-cap rerun at the controller
# meshes' vertex counts.  Run on a login node AFTER make_batch24_files_v1.py has written $AN/rdgrowth/train_rd_adapt_v9b_abl.py and
# $AN/ablation-b24.sbatch (md5s checked below).  Everything goes to the yw523-l40s reservation (no fairshare cost).
# Per scene: nocollapse (10 MB), constants (10 + 20 MB), constlam (lam 0.00185 + 0.0062), matched (10 MB, after nocollapse: it replays
# that run's split counts), cap (budget-sweep-v3: --max_points at V_controller/0.9 for the 20/10/5 MB meshes).  55 jobs, ~165 GPU-h.
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3"; exit 1; }; }
chk $AN/rdgrowth/train_rd_adapt_v9b_abl.py 934 2ae0a46b27aa135a02c1b56194e52e6d
chk $AN/ablation-b24.sbatch 202 a823daddfa8374bffd8537e0ba896d75
chk $AN/budget-sweep-v3.sbatch 88 3c4ba68fe54bc67ddd7852f39d5365f5
bash -n $AN/ablation-b24.sbatch && bash -n $AN/budget-sweep-v3.sbatch || { echo "CHECK FAILED: sbatch syntax"; exit 1; }
grep -q "VERSION abl24-r1" $AN/ablation-b24.sbatch && grep -q "VERSION g0r-v3" $AN/budget-sweep-v3.sbatch || { echo "CHECK FAILED: sbatch versions"; exit 1; }
for f in $AN/rdgrowth/train_rd_adapt_v9b.py $AN/rdgrowth/rd_growth_v2.py $AN/rdgrowth/probe_bytes_v1.py $AN/vis_collapse-v12/scripts/run_garden.sh $AN/indoor_budget_check_v1.py; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
[ "$(md5sum < $AN/rdgrowth/train_rd_adapt_v9b.py | cut -c1-32)" = e379f9906f639d5d69acf127543ce690 ] || { echo "CHECK FAILED: train_rd_adapt_v9b.py is not the batch14 file"; exit 1; }
grep -q "if acc.mode == 'importance'" $AN/rdgrowth/rd_growth_v2.py || { echo "CHECK FAILED: rd_growth_v2.py has no importance mode (matched variant needs it)"; exit 1; }
for S in room kitchen counter bonsai garden bicycle stump flowers treehill; do [ -f $D/m360/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in truck train; do [ -f $D/tandt/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in room kitchen counter bonsai; do [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no --indoor FP32 original for $S (cap job needs it)"; exit 1; }; done
for S in garden bicycle stump flowers treehill truck train; do [ -f /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no FP32 original for $S (cap job needs it)"; exit 1; }; done
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
python $AN/indoor_budget_check_v1.py || { echo "CHECK FAILED: the indoor preset overwrites max_points on this upstream"; exit 1; }
busy=$(squeue -u $USER -h -o "%j" | grep -cE "abl24|g0r24"); [ "$busy" = 0 ] || echo "note: $busy batch24 jobs already in the queue (restartable: finished stages are skipped)"
scontrol show reservation $RES | grep -o "EndTime=[^ ]*\|NodeCnt=[^ ]*"
echo "CHECK OK"; k=0
sb () { j=$(sbatch --parsable --reservation=$RES --account=torch_pr_38_general --cpus-per-task=6 "$@") && { echo "submitted $j  ${@: -1}  (${*:3:1})" >&2; echo $j; } || { echo "FAILED: sbatch $*" >&2; echo ""; }; }
acc () { [ -n "$1" ] && k=$((k+1)); }
cfg () { case $1 in
  room)     echo "1 images_2 q14 '' '2198000 1020000 477000'";;     kitchen)  echo "1 images_2 q16 '' '1895000 928000 409000'";;
  counter)  echo "1 images_2 q15 '' '2040000 978000 449000'";;  bonsai)   echo "1 images_2 q15 '' '2090000 1005000 473000'";;
  garden)   echo "0 images_4 q15 '' '2015000 936000 405000'";;   bicycle)  echo "0 images_4 q16 '' '2029000 986000 440000'";;
  stump)    echo "0 images_4 q16 '' '2015000 983000 440000'";;    flowers)  echo "0 images_4 q15 '' '2068000 992000 447000'";;
  treehill) echo "0 images_4 q16 '' '2054000 1002000 449000'";; truck)    echo "0 images q18 ',DATA=$D/tandt/truck' '1853000 929000 421000'";;
  train)    echo "0 images q18 ',DATA=$D/tandt/train' '2040000 975000 436000'";; esac; }
for S in room kitchen counter bonsai garden bicycle stump flowers treehill truck train; do
  eval set -- $(cfg $S); IN=$1; IM=$2; X=$3; EX=$4; BUD=$5
  B="SCENE=$S,INDOOR=$IN,IMAGES=$IM,XYZ=$X,SMOKE=0,SELFCHECK=0$EX"
  jn=$(sb --time=03:30:00 --export=ALL,$B,ABL=nocollapse,TARGETS=10 --job-name=$S-nocoll-abl24 $AN/ablation-b24.sbatch); acc $jn
  j1=$(sb --time=06:00:00 --export=ALL,$B,ABL=constants,TARGETS="10 20" --job-name=$S-const-abl24 $AN/ablation-b24.sbatch); acc $j1
  j2=$(sb --time=06:00:00 --export=ALL,$B,ABL=constlam,LAMS="0.00185 0.0062" --job-name=$S-clam-abl24 $AN/ablation-b24.sbatch); acc $j2
  if [ -n "$jn" ]; then j3=$(sb --time=03:30:00 --dependency=afterok:$jn --export=ALL,$B,ABL=matched,TARGETS=10,SCHED=/scratch/zr2245/$S-abl24-nocollapse-t10mb/original/rd_log.jsonl --job-name=$S-match-abl24 $AN/ablation-b24.sbatch); acc $j3
  else echo "SKIP matched $S: nocollapse job not accepted"; fi
  j4=$(sb --time=05:30:00 --export=ALL,SCENE=$S,IMAGES=$IM,XYZ=$X,INDOOR=$IN,BUDGETS="$BUD"$EX --job-name=$S-g0r24 $AN/budget-sweep-v3.sbatch); acc $j4
done
echo "submitted $k jobs (expected 55: 11 x [nocollapse, constants, constlam, matched(after nocollapse), cap])"
squeue -u $USER -o "%.10i %.22j %.2t %.10M %.6D %R" | head -60
