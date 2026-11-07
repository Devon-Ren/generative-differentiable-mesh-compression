#!/bin/bash
#SBATCH --job-name=crossbank
#SBATCH --account=torch_pr_38_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/zr2245/garden-vis-collapse/crossbank.%j.log
# crossbank-v1.sbatch (2026-10-16): (A) cross-scene codebook test on garden v0.8 (q15/q7, final format):
#   own        = garden's own k-means K4096 R1024 (reference, same K/R as the Room seed asset)
#   xfer-frozen= Room-trained K4096 R1024 banks, indices assigned on garden, banks NOT refined (rounds 0) and NOT fitted (lr 0); only DC fits
#   xfer-adapt = Room banks as init, rounds 0, banks fine-tuned 400 steps (transfer + adaptation)
# (B) index spatial-locality stats over every Room K x R dir + the three garden dirs -> $G/index-locality.csv (CPU).
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
G=/scratch/zr2245/garden-vis-collapse; W=/scratch/zr2245/room-vis-collapse; A=/scratch/zr2245/analysis; PKG=$A/vis_collapse-v12
DATA=/scratch/zr2245/datasets/m360/garden; IMAGES=images_4; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
C=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; MESH=$G/axisB/v0.8/finetune/best.pt; XYZ=q15; DC=q7
HIST=/scratch/zr2245/room-shared-collapse-QGQgyr/run/config.json
SEED=$(python -c "import json,sys; print(json.load(open('$HIST'))['seed_asset'])" 2>/dev/null)
[ -n "$SEED" ] || SEED=/scratch/zr2245/room-fixed-sharing-0N4OSU/run/shared/residual_k4096_r1024_f1/asset.mslb
for f in $MESH $SEED $PKG/deliver_coarse.sh $A/index_locality.py; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
echo "Room seed asset: $SEED"; nvidia-smi --query-gpu=name,memory.total --format=csv
echo '{"bank_lr": 0.0, "residual_lr": 0.0}' > $A/frozen-banks.json
OUT=$G/crossbank; mkdir -p $OUT
run () { name=$1; shift
  if [ -f $OUT/$name/summary.json ]; then echo "exists: $name"; return; fi
  rm -rf $OUT/$name; echo "=== $name ==="
  bash $PKG/deliver_coarse.sh --coarse $MESH --output $OUT/$name --source $C --dataset $DATA --lab $LAB --images $IMAGES --xyz $XYZ --dc $DC "$@" > $OUT/$name.log 2>&1 \
    || { echo "FAILED $name"; tail -5 $OUT/$name.log; return; }
}
run own         --historical-config none            --k 4096 --r 1024 --dictionary-rounds 4
run xfer-frozen --historical-config $A/frozen-banks.json --k 0 --seed-asset $SEED --dictionary-rounds 0
run xfer-adapt  --historical-config none            --k 0 --seed-asset $SEED --dictionary-rounds 0
python - $OUT <<'PY' | tee $OUT/crossbank.txt
import json, sys, pathlib
for name in ('own', 'xfer-frozen', 'xfer-adapt'):
    d = pathlib.Path(sys.argv[1]) / name
    if not (d / 'summary.json').exists(): print(name, 'MISSING'); continue
    s = json.load(open(d / 'summary.json')); st = {r['stage']: r['validation']['psnr'] for r in s['stages']}; t = s['delivery_test']; pb = s['codec']['payload_bytes']
    print(f"{name:12s} | {s['file_MB']:.3f} MB test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f} | val coarse {st['coarse_input']:.3f} shared_initial {st['shared_initial']:.3f} fitted {st['shared_fitted']:.3f} decoded {st['decoded_delivery']:.3f} | banks {s['banks_MB']:.3f} MB mapping {s['mapping_MB']:.3f} MB")
PY
echo "=== index locality (CPU) ==="
python $A/index_locality.py $G/index-locality.csv $(ls -d $W/axisA-v0.8/k*_r* $W/axisA-q14q7-v*/k*_r* 2>/dev/null | grep -v '\.log$') $OUT/own $OUT/xfer-frozen $OUT/xfer-adapt
echo "crossbank exit 0"
