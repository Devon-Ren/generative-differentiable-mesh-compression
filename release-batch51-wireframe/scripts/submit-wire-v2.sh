#!/usr/bin/env bash
# submit-wire-v2.sh (2026-11-20): triangle-id maps for the wireframe figure.  One GPU job (about 10 minutes): garden, room, bicycle, truck; one test view
# each (the views of the qualitative figure); three meshes per scene, all after the same fine-tuning:
#   FP32 : the MeshSplatting model with released settings        CAP : MeshSplatting-cap with the middle cap (about 1.08 million vertices)
#   OURS : ByteMesh at a target of 10 MB
# Output /scratch/zr2245/wire-b52/<scene>/ and, packed, /scratch/zr2245/wire-b52.zip (JPG renders and the id maps of one crop per scene).
# Usage: bash submit-wire-v2.sh          needs render_ids_v2.py in /scratch/zr2245/analysis
AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245
[ -f $AN/render_ids_v2.py ] || { echo "FAILED: missing $AN/render_ids_v2.py"; exit 1; }
cat > $AN/b52-wire.sh <<'JOB'
#!/usr/bin/env bash
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
AN=/scratch/zr2245/analysis; ROOT=/scratch/zr2245; D=$ROOT/datasets; LAB=$ROOT/meshsplatting-original-lossless-v1; OUT=$ROOT/wire-b52; mkdir -p $OUT
python $AN/render_ids_v2.py --selftest || { echo "FAILED: selftest"; exit 1; }
cfg () { case $1 in
  garden)  echo "$ROOT/garden-vis-collapse $D/m360/garden images_4 1073000 DSC08076 480,300,320,210";;
  room)    echo "$ROOT/room-indoor $D/m360/room images_2 1078000 DSCF4715 820,300,320,210";;
  bicycle) echo "$ROOT/bicycle-vis-collapse $D/m360/bicycle images_4 1080000 _DSC8727 140,400,320,210";;
  truck)   echo "$ROOT/truck-vis-collapse $D/tandt/truck images 1078000 000049 400,220,320,210";; esac; }
for S in garden room bicycle truck; do read ORIG DATA IMAGES MID VIEW CROP <<< "$(cfg $S)"
  FP=$ORIG/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; CA=$ROOT/$S-budget-$MID/axisB/v1.0/finetune/best.pt; OU=$ROOT/$S-abl24-hold-t10mb/axisB/v1.0/finetune/best.pt; CK=""
  for kv in FP32=$FP CAP=$CA OURS=$OU; do [ -f ${kv#*=} ] && CK="$CK --ckpt $kv" || echo "note: $S: missing ${kv#*=}"; done
  [ -n "$CK" ] || { echo "skipped: $S"; continue; }
  python $AN/render_ids_v2.py --lab $LAB --dataset $DATA --images $IMAGES --model $ORIG/original --names $VIEW --crop $CROP $CK --out $OUT/$S > $OUT/$S.log 2>&1 && grep -h "^WIRE" $OUT/$S.log || { echo "FAILED: $S"; grep -n -m 6 -i "error\|Traceback\|FAILED" $OUT/$S.log; tail -6 $OUT/$S.log; }
done
(cd $ROOT && python - <<'PY'
import zipfile, glob, os
fs = sorted(glob.glob('wire-b52/*/*.jpg') + glob.glob('wire-b52/*/*.npz') + glob.glob('wire-b52/*/stats.txt'))
with zipfile.ZipFile('wire-b52.zip', 'w', zipfile.ZIP_STORED) as z:
    for f in fs: z.write(f)
print('packed %d files, %.1f MB -> /scratch/zr2245/wire-b52.zip' % (len(fs), os.path.getsize('wire-b52.zip') / 1e6))
PY
)
echo "b52-wire exit 0  $(date)"
JOB
j=$(sbatch --parsable --account=torch_pr_38_general --gres=gpu:1 --constraint=l40s --cpus-per-task=6 --mem=48G --time=00:45:00 --job-name=b52-wire --output=$AN/b52-wire-%j.out $AN/b52-wire.sh) && echo "submitted $j  b52-wire" || echo "FAILED: sbatch"
