#!/bin/bash
#SBATCH --job-name=crossbank2
#SBATCH --account=torch_pr_38_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/zr2245/garden-vis-collapse/crossbank2.%j.log
# crossbank-v2.sbatch (2026-10-16): controls for the cross-scene codebook result (crossbank-v1: own 24.393 / xfer-frozen 23.745 / xfer-adapt 24.162).
#   garden rand     : random-sample codebook (k-means 0 rounds), no refinement, 400-step fit  -> is the Room init better than random?
#   garden own-r0   : garden k-means codebook, --dictionary-rounds 0 (same refinement budget as the xfer runs) -> removes the rounds-4 confound
#   room  room-own-r0 / rev-frozen / rev-adapt : the reverse direction, garden's own codebook on the Room v0.8 mesh (q14/q7)
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
G=/scratch/zr2245/garden-vis-collapse; W=/scratch/zr2245/room-vis-collapse; A=/scratch/zr2245/analysis; PKG=$A/vis_collapse-v12; S=/scratch/zr2245/shared-collapse-stage2-v2
GDATA=/scratch/zr2245/datasets/m360/garden; GIMAGES=images_4; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
GC=$G/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; GMESH=$G/axisB/v0.8/finetune/best.pt
RMESH=$W/axisB/v0.8/finetune/best.pt; HIST=/scratch/zr2245/room-shared-collapse-QGQgyr/run/config.json
GSEED=$G/crossbank/own/delivery/asset.mslb
for f in $GMESH $RMESH $GSEED $HIST $PKG/deliver_coarse.sh; do [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "--- what --kmeans-rounds 0 means (appearance.initialize source head) ---"
python - <<PY 2>&1 | head -40
import sys, inspect; sys.path[:0] = ['$S/renderer', '$S', '$S/codec']
try:
    from appearance import initialize; print(inspect.getsource(initialize))
except Exception as e: print('could not print initialize:', repr(e))
PY
python - <<PY
import json; d = json.load(open('$HIST')); d['bank_lr'] = 0.0; d['residual_lr'] = 0.0; json.dump(d, open('$A/frozen-banks-room.json', 'w'), indent=1); print('frozen-banks-room.json keys', sorted(d))
PY
OUT=$G/crossbank2; mkdir -p $OUT
run () { name=$1; shift
  if [ -f $OUT/$name/summary.json ]; then echo "exists: $name"; return; fi
  rm -rf $OUT/$name; echo "=== $name === $(date +%H:%M:%S)"
  bash $PKG/deliver_coarse.sh --output $OUT/$name "$@" > $OUT/$name.log 2>&1 \
    || { echo "FAILED $name"; tail -8 $OUT/$name.log; return; }
}
GARGS="--coarse $GMESH --source $GC --dataset $GDATA --lab $LAB --images $GIMAGES --xyz q15 --dc q7 --historical-config none"
run rand        $GARGS --k 4096 --r 1024 --kmeans-rounds 0 --dictionary-rounds 0
run own-r0      $GARGS --k 4096 --r 1024 --dictionary-rounds 0
RARGS="--coarse $RMESH --xyz q14 --dc q7"
run room-own-r0 $RARGS --k 4096 --r 1024 --dictionary-rounds 0
run rev-frozen  $RARGS --historical-config $A/frozen-banks-room.json --k 0 --seed-asset $GSEED --dictionary-rounds 0
run rev-adapt   $RARGS --k 0 --seed-asset $GSEED --dictionary-rounds 0
python - $OUT <<'PY' | tee $OUT/crossbank2.txt
import json, sys, pathlib
print('# garden v0.8 q15/q7: crossbank-v1 own 28.056 MB / 24.393 (rounds 4), xfer-frozen 23.745, xfer-adapt 24.162 | Room v0.8 q14/q7 K4096 R1024 own (rounds 4, hull-v2): 26.237 MB / 27.907')
for name in ('rand', 'own-r0', 'room-own-r0', 'rev-frozen', 'rev-adapt'):
    d = pathlib.Path(sys.argv[1]) / name
    if not (d / 'summary.json').exists(): print(name, 'MISSING'); continue
    s = json.load(open(d / 'summary.json')); st = {r['stage']: r['validation']['psnr'] for r in s['stages']}; t = s['delivery_test']
    print(f"{name:12s} | {s['file_MB']:.3f} MB test {t['psnr']:.3f} {t['ssim']:.4f} {t['lpips']:.4f} | val coarse {st['coarse_input']:.3f} shared_initial {st['shared_initial']:.3f} fitted {st['shared_fitted']:.3f} decoded {st['decoded_delivery']:.3f} | banks {s['banks_MB']:.3f} MB mapping {s['mapping_MB']:.3f} MB")
PY
echo "crossbank2 exit 0"
