#!/usr/bin/env bash
#SBATCH --account=torch_pr_38_general
#SBATCH --job-name=rend23
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/zr2245/analysis/rend23-%x-%j.out
#SBATCH --error=/scratch/zr2245/analysis/rend23-%x-%j.err
# render-b23.sbatch (2026-11-10, VERSION rend23-v1): qualitative figure for one scene.  Each entry of ASSETS is NAME=<deliver dir> (decoded with the
# stage-2 decoder by decode_asset_v1.py, inside the stage-2 env) or NAME=<state_dict.pt> (an FP32 checkpoint, used as is).  Then render_ckpt_v1.py
# renders the chosen test VIEWS of every entry next to the ground truth at the metrics' supersampling (4x), writes per-view PNGs, an error row and
# montage.png under $OUT.  A failed decode falls back to the pre-quantisation shared mesh (file name *-prequant.pt; the montage column is then
# labelled NAME~).  Nothing is modified in the source run directories.
#   sbatch --export=ALL,SCENE=room,DATA=/scratch/zr2245/datasets/m360/room,IMAGES=images_2,MODEL=/scratch/zr2245/room-indoor/original,VIEWS="2 10",OUT=/scratch/zr2245/renders-b23/room,ASSETS="FP32=/scratch/zr2245/room-indoor/original/point_cloud/iteration_30000/point_cloud_state_dict.pt|20MB=/scratch/zr2245/room-adapt9x-t20mb/axisB/v1.0/deliver|..." --job-name=room-b23-rend .../render-b23.sbatch
set -o pipefail
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=6
SCENE=${SCENE:?}; DATA=${DATA:?}; IMAGES=${IMAGES:?}; MODEL=${MODEL:?}; VIEWS=${VIEWS:-"2 10"}; OUT=${OUT:?}; ASSETS=${ASSETS:?set ASSETS="NAME=path|NAME=path" via --export}; SS=${SS:-4}
LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; AN=/scratch/zr2245/analysis; PKG=$AN/vis_collapse-v12; R=$AN/render
echo "render-b23 | scene $SCENE | data $DATA/$IMAGES | cameras from $MODEL | views $VIEWS | ss $SS | out $OUT | job $SLURM_JOB_ID on $(hostname) $(date)"
for f in $R/decode_asset_v1.py $R/render_ckpt_v1.py $PKG/stage2_run.sh $MODEL/point_cloud/iteration_30000/point_cloud_state_dict.pt $LAB/upstream/train.py; do [ -e $f ] || { echo "FAILED: MISSING $f"; exit 1; }; done
[ -e $DATA/$IMAGES/. ] || [ -f $DATA/transforms_test.json ] || { echo "FAILED: dataset $DATA has neither $IMAGES/ nor transforms_test.json"; exit 1; }
mkdir -p $OUT/decoded; CK=""; nf=0
IFS='|' read -ra ENTRIES <<< "$ASSETS"
for e in "${ENTRIES[@]}"; do
  n=${e%%=*}; p=${e#*=}
  if [ -f "$p" ]; then CK="$CK --ckpt $n=$p"; echo "asset $n: checkpoint $p"; continue; fi
  if [ -d "$p" ]; then
    echo "=== decode $n from $p ($(date +%H:%M:%S)) ==="
    bash $PKG/stage2_run.sh $R/decode_asset_v1.py --deliver-dir $p --out $OUT/decoded/$n.pt > $OUT/decoded/$n.log 2>&1; grep -h "^\[decode\]\|DECODE OK\|Error\|Traceback" $OUT/decoded/$n.log | tail -6
    if [ -f $OUT/decoded/$n.pt ]; then CK="$CK --ckpt $n=$OUT/decoded/$n.pt"
    elif [ -f $OUT/decoded/$n-prequant.pt ]; then CK="$CK --ckpt $n~=$OUT/decoded/$n-prequant.pt"; echo "note: $n rendered from the pre-quantisation mesh"
    else echo "FAILED: decode $n"; nf=$((nf+1)); fi
  else echo "FAILED: $n: $p is neither a file nor a directory"; nf=$((nf+1)); fi
done
[ -n "$CK" ] || { echo "FAILED: nothing to render"; exit 1; }
echo "=== render ($(date +%H:%M:%S)) ==="
python $R/render_ckpt_v1.py --lab $LAB --dataset $DATA --images $IMAGES --model $MODEL --views $VIEWS --supersample $SS --out $OUT $CK > $OUT/render.log 2>&1 || { echo "FAILED: RENDER"; grep -n -m 8 -i 'error\|Traceback' $OUT/render.log; tail -12 $OUT/render.log; exit 1; }
grep -h "^\[render\]" $OUT/render.log; cat $OUT/psnr.txt; ls -la $OUT/montage.png
echo "render-b23 $SCENE exit 0 ($nf decode failures)  $(date)"
