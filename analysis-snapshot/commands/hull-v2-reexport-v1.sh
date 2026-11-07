#!/usr/bin/env bash
# hull-v2-reexport-v1.sh (2026-10-15): re-express every measured Room RD point in the stage2-v2 format (const vertex_weight
# + ctx indices) at q14/q7 WITHOUT training: one codec_variants.py const/ctx export + decoded val/test per deliver dir.
# ~1-2 min per point on an L40S (export seconds + test render).  Restartable (codec_variants skips measured variants).
# Output table: $W/hull-v2.txt.   Usage (GPU node, fresh shell):
#   source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
#   bash /scratch/zr2245/analysis/hull-v2-reexport-v1.sh
set -o pipefail
W=/scratch/zr2245/room-vis-collapse
PKG=/scratch/zr2245/analysis/vis_collapse-v12
XYZ=${XYZ:-q14}; DC=${DC:-q7}
DIRS=${DIRS:-"$W/axisB/v1.0/deliver $W/axisB/v0.9/deliver $W/axisB/v0.8/deliver $W/axisB/v0.7/deliver $W/axisB/v0.6/deliver $W/axisB/v0.5/deliver $W/deliver-geo-only $(ls -d $W/axisA-v0.8/k*_r* 2>/dev/null)"}
: > $W/hull-v2.tmp
for d in $DIRS; do
  [ -f $d/shared-final.pt ] || { echo "skip (no shared-final.pt): $d"; continue; }
  echo "=== $d ==="
  bash $PKG/stage2_run.sh codec_variants.py --deliver-dir $d --xyz $XYZ --dc $DC --variants const/ctx > $d/codec-variants-const-ctx.log 2>&1 || { echo "FAILED $d"; tail -3 $d/codec-variants-const-ctx.log; continue; }
  python - "$d" "$XYZ" "$DC" >> $W/hull-v2.tmp <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1]); xyz, dc = sys.argv[2], sys.argv[3]
s = json.load(open(d / 'codec-variants' / 'sweep.json'))
cfg = json.load(open(d / 'config.json')); k = cfg.get('k', 0); r = cfg.get('r', 0)
old = json.load(open(d / 'summary.json')) if (d / 'summary.json').exists() else {}
for row in s['rows']:
    if (row['weight'], row['index_codec'], row['xyz'], row['dc']) != ('const', 'ctx', xyz, dc): continue
    t = row['test'] or {}
    print(f"{d.relative_to(d.parents[1]) if len(d.parents) > 1 else d} | vertices {s['vertices']} | K {k or 'seed4096'} R {r or 'seed1024'} | v2 {xyz}/{dc} {row['file_MB']:.3f} MB test {t.get('psnr', float('nan')):.3f} {t.get('ssim', float('nan')):.4f} {t.get('lpips', float('nan')):.4f} | v1-format {old.get('xyz','?')}/{old.get('dc','?')} {old.get('file_MB', float('nan')):.3f} MB test {old.get('delivery_test', {}).get('psnr', float('nan')):.3f}")
PY
done
mv $W/hull-v2.tmp $W/hull-v2.txt; echo "===== $W/hull-v2.txt ====="; cat $W/hull-v2.txt
