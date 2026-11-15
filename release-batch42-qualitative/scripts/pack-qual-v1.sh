#!/usr/bin/env bash
# pack-qual-v1.sh (2026-11-15): collect the rendered views of submit-qual-v1.sh into one zip (JPEG quality 97, error maps left out).
ROOT=/scratch/zr2245; OUTR=$ROOT/renders-qual; P=$OUTR/pack; rm -rf $P; mkdir -p $P
[ -f /scratch/zr2245/anaconda3/etc/profile.d/conda.sh ] && { source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; }
for S in garden room bicycle truck; do
  [ -f $OUTR/$S/psnr.txt ] || { echo "not finished: $S ($(ls -t $ROOT/analysis/rend23-$S-qual-*.out 2>/dev/null | head -1 | xargs -r tail -2 | tr '\n' ' '))"; continue; }
  mkdir -p $P/$S; cp $OUTR/$S/psnr.txt $P/$S/
  python - "$OUTR/$S" "$P/$S" <<'PY'
import sys, glob, os
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]; n = 0
for f in sorted(glob.glob(src + '/view-*.png')):
    if f.endswith('-err.png'): continue
    Image.open(f).convert('RGB').save(os.path.join(dst, os.path.basename(f)[:-4] + '.jpg'), quality=97, subsampling=0); n += 1
print(os.path.basename(src), n, 'images')
PY
done
cd $OUTR && rm -f qual-renders.zip && zip -q -r qual-renders.zip pack && ls -la $OUTR/qual-renders.zip
