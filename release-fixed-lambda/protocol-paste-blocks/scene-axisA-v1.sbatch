#!/bin/bash
#SBATCH --job-name=scene-axisA
#SBATCH --account=torch_pr_38_general
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/zr2245/analysis/axisA-%x-%j.out
# scene-axisA-v1.sbatch (2026-10-17): bonsai-axisA-v1 made scene-generic (SCENE / IMAGES via --export), on a scene already run by scene-v1.sbatch:
#   (1) precision rule: geom_precision.py on the v0.8 and v1.0 deliveries           -> $B/geom-precision.txt
#   (2) precision re-export (no training, codec_variants const/ctx): XYZ q13..q17 x DC q7 on levels 1.0 / 0.8 / 0.7, + v0.8 q16/q8 (DC check)
#       -> $B/precision-$SCENE.txt.  Sharing+quant cost at 1.0 was 0.4-1.5 dB on these scenes (Room 0.17): this tells whether it is XYZ precision.
#   (3) XYZ for the backbone chosen by the allocator rule argmax_q (test PSNR - 0.05 dB/MB x MB) on v0.8 (lambda = typical mid-hull slope)
#   (4) K x R backbone on v0.8 at that XYZ (k-means rounds 4): K1024 R1, K4096 R1 / R256 / R1024 / R4096, K16384 R4096 -> $B/axisA-$SCENE.txt
# Submit:  sbatch --export=ALL,SCENE=counter,IMAGES=images_2 --job-name=counter-axisA /scratch/zr2245/analysis/scene-axisA-v1.sbatch
set -o pipefail
export TORCH_HOME=/scratch/zr2245/cache/torch
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
SCENE=${SCENE:?set SCENE=counter (etc.) via --export}; IMAGES=${IMAGES:-images_2}
B=/scratch/zr2245/$SCENE-vis-collapse; A=/scratch/zr2245/analysis; PKG=$A/vis_collapse-v12
DATA=/scratch/zr2245/datasets/m360/$SCENE; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1
C=$B/original/point_cloud/iteration_30000/point_cloud_state_dict.pt; MESH=$B/axisB/v0.8/finetune/best.pt
echo "scene $SCENE | B $B | IMAGES $IMAGES | job $SLURM_JOB_ID on $(hostname) $(date)"
for f in $MESH $C $PKG/deliver_coarse.sh $PKG/stage2_run.sh $A/geom_precision.py $B/axisB/v0.8/deliver/shared-final.pt $B/axisB/v1.0/deliver/summary.json $B/axisB/v0.7/deliver/summary.json $DATA/$IMAGES/.; do
  [ -e "$f" ] || { echo "MISSING $f"; exit 2; }; done
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=== (1) precision rule ==="
python $A/geom_precision.py $B/axisB/v0.8/deliver $B/axisB/v1.0/deliver 2>&1 | tee $B/geom-precision.txt
echo "=== (2) precision re-export ($(date +%H:%M)) ==="
: > $B/precision-$SCENE.txt
reexport () { d=$1; XYZ=$2; DC=$3
  python -c "import json,sys; s=json.load(open('$d/codec-variants/sweep.json')); sys.exit(0 if any((r['weight'],r['index_codec'],r['xyz'],r['dc'])==('const','ctx','$XYZ','$DC') and r.get('test') for r in s['rows']) else 1)" 2>/dev/null && { echo "exists: $d $XYZ/$DC"; return; }
  bash $PKG/stage2_run.sh codec_variants.py --deliver-dir $d --xyz $XYZ --dc $DC --variants const/ctx > $d/codec-variants-$XYZ-$DC.log 2>&1 || { echo "FAILED re-export $d $XYZ/$DC"; tail -3 $d/codec-variants-$XYZ-$DC.log; }
}
for L in 0.8 1.0 0.7; do d=$B/axisB/v$L/deliver
  for XYZ in q13 q14 q15 q16 q17; do reexport $d $XYZ q7; done
  [ "$L" = 0.8 ] && reexport $d q16 q8
  python - "$d" "$L" "$SCENE" >> $B/precision-$SCENE.txt <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); L = sys.argv[2]; sc = sys.argv[3]; s = json.load(open(d / 'codec-variants' / 'sweep.json'))
seen = set()
for row in s['rows']:
    key = (row['weight'], row['index_codec'], row['xyz'], row['dc'])
    if key[:2] != ('const', 'ctx') or key in seen: continue
    seen.add(key); t = row['test'] or {}; fl = (row.get('quantization_geometry') or {}).get('flipped_faces', '?')
    print(f"{sc} v{L} | vertices {s['vertices']} | {row['xyz']}/{row['dc']} {row['file_MB']:.3f} MB test {t.get('psnr', float('nan')):.3f} {t.get('ssim', float('nan')):.4f} {t.get('lpips', float('nan')):.4f} | flips {fl}")
PY
done
echo "----- $B/precision-$SCENE.txt -----"; cat $B/precision-$SCENE.txt
echo "=== (3) choose XYZ for the backbone (v0.8, argmax psnr - 0.05*MB, DC q7) ==="
XYZBB=$(python - $B/precision-$SCENE.txt <<'PY'
import sys, re
best = None
for line in open(sys.argv[1]):
    m = re.search(r' v0\.8 \| .*\| (q\d+)/q7 ([\d.]+) MB test ([\d.]+) ', line)
    if not m: continue
    q, mb, ps = m.group(1), float(m.group(2)), float(m.group(3)); score = ps - 0.05 * mb
    print(f"  {q}: {mb:.3f} MB {ps:.3f} dB score {score:.3f}", file=sys.stderr)
    if best is None or score > best[0]: best = (score, q)
print(best[1] if best else 'q14')
PY
)
echo "backbone XYZ = $XYZBB"
echo "=== (4) K x R backbone at v0.8 $XYZBB/q7 ($(date +%H:%M)) ==="
OUT=$B/axisA-v0.8-$XYZBB; mkdir -p $OUT
for KR in "1024 1" "4096 1" "4096 256" "4096 1024" "4096 4096" "16384 4096"; do set -- $KR; K=$1; R=$2; name=$OUT/k${K}_r${R}
  if [ -f $name/summary.json ]; then echo "exists: k${K}_r${R}"; continue; fi
  rm -rf $name; echo "=== K $K R $R === $(date +%H:%M:%S)"
  bash $PKG/deliver_coarse.sh --coarse $MESH --output $name --source $C --dataset $DATA --lab $LAB --images $IMAGES --xyz $XYZBB --dc q7 \
    --historical-config none --k $K --r $R --dictionary-rounds 4 > $name.log 2>&1 || { echo "FAILED k${K}_r${R}"; tail -8 $name.log; }
done
python - $OUT $XYZBB $SCENE <<'PY' | tee $B/axisA-$SCENE.txt
import json, sys, pathlib
out = pathlib.Path(sys.argv[1]); xyz = sys.argv[2]; sc = sys.argv[3]
print('scene,level,xyz,K,R,MB,test_psnr,test_ssim,test_lpips,vertices')
for K, R in ((1024, 1), (4096, 1), (4096, 256), (4096, 1024), (4096, 4096), (16384, 4096)):
    d = out / f'k{K}_r{R}'
    if not (d / 'summary.json').exists(): print(f'{sc},0.8,{xyz},{K},{R},MISSING'); continue
    s = json.load(open(d / 'summary.json')); t = s['delivery_test']
    print(f"{sc},0.8,{xyz},{K},{R},{s['file_MB']:.3f},{t['psnr']:.3f},{t['ssim']:.4f},{t['lpips']:.4f},{s['stages'][-1]['vertices']}")
PY
echo "$SCENE-axisA exit 0  $(date)"
