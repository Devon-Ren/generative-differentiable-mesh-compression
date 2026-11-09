#!/usr/bin/env bash
# submit-batch21-v1.sh (2026-11-09, r2: rd-prune-blender-v5 = the Blender lambda sbatch that exists on HPC; v4 was superseded the same day): the completeness batch, everything at once, last GPU window (reservation ends 11/12 09:35).
# Fills every hole in the main tables so that each claim covers all four datasets, plus redundancy on the controller.
#   A  G0 (train-at-budget vs post-hoc collapse) on Deep Blending          2 jobs   budget-sweep-v3.sbatch (INDOOR=1, DB originals)
#   C  NeRF-Synthetic fixed-lambda 1.85e-3 and 1.85e-2 (4-point hull)       8 jobs   rd-prune-blender-v5.sbatch
#   C2 their val_select_v3 re-delivery (-best3, the hull's protocol)         8 jobs   redeliver-v3-blender.sbatch      afterok C
#   B  G0 on NeRF-Synthetic at all four lambda vertex counts                 8 jobs   budget-sweep-blender-v1.sbatch   afterok C2
#   D  lambda x dictionary fan + DC q8 on Deep Blending and NeRF-Synthetic  10 jobs   lambda-k-fan-v3.sbatch           (Blender: afterok C2)
#   E1 controller v9b (the headline version) on Deep Blending 5/10/20 MB     2 jobs   adapt-v9b.sbatch
#   E2 controller v9b on Mip-360 / T&T at 7.5 and 15 MB (5 targets total)   11 jobs   adapt-v9b.sbatch
#   E3 controller v9b on NeRF-Synthetic at 1.5 / 3 / 6 MB (5 targets total)  8 jobs   adapt-v9b-blender.sbatch
#   F  generic codecs (Draco / zstd / texture / MPEG) on the 9 remaining Mip/T&T scenes + Deep Blending   2 jobs   generic-codecs-b21.sbatch
# ~100 GPU-h; 4 GPUs -> ~26 h wall.  Every job name carries 'b21'.  Nothing here overwrites an existing run directory (new names throughout).
AN=/scratch/zr2245/analysis; RES=yw523-l40s; D=/scratch/zr2245/datasets; ACC=torch_pr_38_general
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
chk () { n=$(wc -l < $1); [ "$n" = $2 ] || { echo "CHECK FAILED: $1 has $n lines, expected $2 (paste truncated?)"; exit 1; }; m=$(md5sum < $1 | cut -c1-32); [ "$m" = $3 ] || { echo "CHECK FAILED: $1 md5 $m != $3 (paste altered?)"; exit 1; }; }
chk $AN/budget-sweep-blender-v1.sbatch 105 d720b8ccba85622f90f25ee0b52b77ee
chk $AN/lambda-k-fan-v3.sbatch 75 5d0dd7ff4012fa9a858c3a0ed1aff8d6
chk $AN/generic-codecs-b21.sbatch 45 28807d9a2ed0a77d4a29fd4e75f910a3
for f in $AN/budget-sweep-v3.sbatch $AN/rd-prune-blender-v5.sbatch $AN/redeliver-v3-blender.sbatch $AN/adapt-v9b.sbatch $AN/adapt-v9b-blender.sbatch $AN/budget-sweep-blender-v1.sbatch $AN/lambda-k-fan-v3.sbatch $AN/generic-codecs-b21.sbatch; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; bash -n $f || { echo "CHECK FAILED: syntax $f"; exit 1; }; done
grep -q "VERSION g0r-v3" $AN/budget-sweep-v3.sbatch && grep -q "VERSION v5" $AN/rd-prune-blender-v5.sbatch && grep -q "VERSION rd3" $AN/redeliver-v3-blender.sbatch && grep -q "VERSION g0rb-v1" $AN/budget-sweep-blender-v1.sbatch && grep -q "VERSION v3" $AN/lambda-k-fan-v3.sbatch && grep -q "VERSION genc-b21" $AN/generic-codecs-b21.sbatch || { echo "CHECK FAILED: sbatch versions"; exit 1; }
grep -q "train_rd_adapt_v9b.py" $AN/adapt-v9b.sbatch && grep -q "train_rd_adapt_v9b.py" $AN/adapt-v9b-blender.sbatch || { echo "CHECK FAILED: adapt-v9b sbatch files do not reference train_rd_adapt_v9b.py"; exit 1; }
for f in $AN/rdgrowth/train_rd_adapt_v9b.py $AN/rdgrowth/train_rd_v3.py $AN/rdgrowth/probe_bytes_v1.py $AN/blender/val_select_v3.py $AN/blender/val_select_v2.py $AN/blender/train_ckpt_v2.py $AN/vis_collapse-v12/coarse_vis.py $AN/coarse_finetune.py $AN/generic-v1/run_generic_v1.sh $AN/o3dgc-v1/run_o3dgc_v3.sh; do [ -f $f ] || { echo "CHECK FAILED: missing $f"; exit 1; }; done
python -m py_compile $AN/rdgrowth/train_rd_adapt_v9b.py $AN/blender/val_select_v3.py $AN/blender/train_ckpt_v2.py || { echo "CHECK FAILED: py_compile"; exit 1; }
python $AN/indoor_budget_check_v1.py || { echo "CHECK FAILED: the indoor preset overwrites max_points on this upstream (budget runs with --indoor would be wrong)"; exit 1; }
BL="chair drums ficus hotdog lego materials mic ship"; MIP="room kitchen counter bonsai garden bicycle stump flowers treehill truck train"; DB="drjohnson playroom"
for S in $BL; do [ -f $D/nerf_synthetic/$S/transforms_val.json ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; done
for S in $DB; do [ -f $D/db/$S/sparse/0/cameras.bin ] || { echo "CHECK FAILED: no dataset for $S"; exit 1; }; [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || { echo "CHECK FAILED: no --indoor FP32 original for $S at /scratch/zr2245/$S-indoor"; exit 1; }; done
for S in room kitchen counter bonsai; do [ -f /scratch/zr2245/$S-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] || echo "note: $S has no -indoor original (only matters for the generic-codec job's o3dgc branch, which uses -vis-collapse anyway)"; done
echo "=== Blender originals for the G0 control (val-selected -blender4-best preferred) ==="; declare -A BORIG
for S in $BL; do O=""; for c in /scratch/zr2245/$S-blender4-best /scratch/zr2245/$S-blender4 /scratch/zr2245/$S-blender2; do [ -f $c/original/point_cloud/iteration_30000/point_cloud_state_dict.pt ] && { O=$c; break; }; done
  [ -n "$O" ] && { BORIG[$S]=$O; echo "$S: $O"; } || echo "WARNING: $S has no Blender original; its G0 job will fail at preflight (train-at-budget rows still need it for the control)"; done
echo "=== existing Blender hull points (-best3) and DB lambda runs for the fan ==="
for S in $BL; do e=""; for L in 0.00062 0.0062; do [ -f /scratch/zr2245/$S-rd3b4-lam$L-mu1.0-best3/axisB/v1.0/deliver/summary.json ] && e="$e $L"; done; echo "$S best3:$e"; done
for S in $DB; do e=""; for L in 0.00062 0.00185 0.0062 0.0185; do [ -f /scratch/zr2245/$S-rd3i-lam$L-mu1.0/axisB/v1.0/deliver/summary.json ] && e="$e $L"; done; echo "$S rd3i:$e"; done
for S in $BL; do for L in 0.00185 0.0185; do [ -e /scratch/zr2245/$S-rd3b4-lam$L-mu1.0 ] && echo "note: /scratch/zr2245/$S-rd3b4-lam$L-mu1.0 exists (restart: finished stages skipped)"; done; done
busy=$(squeue -u $USER -h -o "%j" | grep -c "b21"); [ "$busy" = 0 ] || { echo "CHECK FAILED: $busy b21 jobs already in the queue (double submission?)"; squeue -u $USER -o "%.10i %.24j %.2t %.10M"; exit 1; }
scontrol show node gl030 | grep -o "AllocTRES=[^ ]*"; scontrol show reservation $RES | grep -o "EndTime=[^ ]*"
echo "CHECK OK"; k=0; declare -A JC JC2
sb () { j=$(sbatch --parsable --reservation=$RES --account=$ACC --cpus-per-task=6 "$@") && { echo "submitted $j  $(echo "$@" | grep -o 'job-name=[^ ]*' | cut -d= -f2)"; k=$((k+1)); LASTJ=$j; } || { echo "FAILED: sbatch $*"; LASTJ=""; }; }
# --- A: G0 Deep Blending (budgets = lambda asset V / 0.9 from deep-blending-fixed-lambda-2026-11-09: 1232667/478718/200920 and 1080549/396809/171479)
sb --time=06:00:00 --export=ALL,SCENE=drjohnson,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/drjohnson,BUDGETS="1370000 532000 224000" --job-name=drjohnson-b21-g0r $AN/budget-sweep-v3.sbatch
sb --time=06:00:00 --export=ALL,SCENE=playroom,IMAGES=images,XYZ=q15,INDOOR=1,DATA=$D/db/playroom,BUDGETS="1201000 441000 191000" --job-name=playroom-b21-g0r $AN/budget-sweep-v3.sbatch
# --- C: NeRF-Synthetic fixed-lambda 1.85e-3 / 1.85e-2 (same sbatch, prices, schedule, codec as the 11/04 hull points), then C2 re-delivery
for S in $BL; do
  sb --time=03:30:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,LAMS="0.00185 0.0185",BPV=10.0,BPF=2.0,MU=1.0,MAXF=0.15 --job-name=$S-b21-b4-185 $AN/rd-prune-blender-v5.sbatch; JC[$S]=$LASTJ
  if [ -n "${JC[$S]}" ]; then sb --time=01:30:00 --dependency=afterok:${JC[$S]} --export=ALL,RUNS="$S-rd3b4-lam0.00185-mu1.0 $S-rd3b4-lam0.0185-mu1.0",XYZ=q14 --job-name=$S-b21-redeliv3 $AN/redeliver-v3-blender.sbatch; JC2[$S]=$LASTJ; fi
done
# --- D (Deep Blending now; Blender after C2): lambda x dictionary fan + DC q8
for S in $DB; do sb --time=04:00:00 --export=ALL,SCENE=$S,TAG=rd3i,XYZ=q15,IMAGES=images,DATA=$D/db/$S --job-name=$S-b21-kfan $AN/lambda-k-fan-v3.sbatch; done
# --- E1: controller v9b on Deep Blending (the headline controller version; batch18 ran v9d there)
for S in $DB; do sb --time=06:00:00 --export=ALL,SCENE=$S,INDOOR=1,IMAGES=images,XYZ=q15,TARGETS="5 10 20",DATA=$D/db/$S,SMOKE=1,SELFCHECK=0 --job-name=$S-b21-adapt9x $AN/adapt-v9b.sbatch; done
# --- B + D(Blender): after each scene's C2 (the budgets and the fan read the -best3 deliveries; existing 6.2e-4 / 6.2e-3 points are used as they are)
for S in $BL; do dep=""; [ -n "${JC2[$S]}" ] && dep="--dependency=afterany:${JC2[$S]}"
  sb --time=04:00:00 $dep --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,ORIG=${BORIG[$S]:-/scratch/zr2245/$S-blender4-best},LAMS_AUTO="0.00062 0.00185 0.0062 0.0185" --job-name=$S-b21-g0rb $AN/budget-sweep-blender-v1.sbatch
  sb --time=03:00:00 $dep --export=ALL,SCENE=$S,TAG=rd3b4,SUFFIX=-best3,XYZ=q14,IMAGES=train,DATA=$D/nerf_synthetic/$S --job-name=$S-b21-kfan $AN/lambda-k-fan-v3.sbatch
done
# --- E2: controller v9b on Mip-360 / T&T at 7.5 and 15 MB (adds two targets to the 5/10/20 table)
m () { sb --time=03:30:00 --export=ALL,SCENE=$1,INDOOR=$2,IMAGES=$3,XYZ=$4,TARGETS="7.5 15"$5,SMOKE=0,SELFCHECK=0 --job-name=$1-b21-adapt9x $AN/adapt-v9b.sbatch; }
m room 1 images_2 q14; m kitchen 1 images_2 q16; m counter 1 images_2 q15; m bonsai 1 images_2 q15
m garden 0 images_4 q15; m bicycle 0 images_4 q16; m stump 0 images_4 q16; m flowers 0 images_4 q15; m treehill 0 images_4 q16
m truck 0 images q18 ",DATA=$D/tandt/truck"; m train 0 images q18 ",DATA=$D/tandt/train"
# --- E3: controller v9b on NeRF-Synthetic at 1.5 / 3 / 6 MB (adds three targets to the 2.5/4 table; ficus/mic at 6 MB report the capacity limit)
for S in $BL; do sb --time=02:30:00 --export=ALL,SCENE=$S,XYZ=q14,INDOOR=1,TARGETS="1.5 3 6",SMOKE=0,SELFCHECK=0 --job-name=$S-b21-adapt9xb $AN/adapt-v9b-blender.sbatch; done
# --- F: generic codecs on the remaining scenes (room / garden already have rows)
sb --time=10:00:00 --export=ALL,SCENES="kitchen counter bonsai drjohnson playroom truck" --job-name=b21-genc-a $AN/generic-codecs-b21.sbatch
sb --time=10:00:00 --export=ALL,SCENES="bicycle stump flowers treehill train" --job-name=b21-genc-b $AN/generic-codecs-b21.sbatch
echo "submitted $k of 59 jobs (A 2, C 8, C2 8, D 10, E1 2, B 8, E2 11, E3 8, F 2); dependent jobs wait on their scene's C / C2"
squeue -u $USER -o "%.10i %.26j %.2t %.10M %.12E" | head -70
