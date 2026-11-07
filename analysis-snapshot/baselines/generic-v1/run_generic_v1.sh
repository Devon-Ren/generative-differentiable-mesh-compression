#!/bin/bash
# run_generic_v1.sh (2026-10-24): generic-codec baselines on one scene's ORIGINAL (level-1.0) mesh. Run on the GPU node:
#   bash /scratch/zr2245/analysis/generic-v1/run_generic_v1.sh room    (then garden)
# CPU part: zstd / Draco / xatlas texture rows -> $OUT/rows.json ; GPU part: render each decoded checkpoint on the test views.
# Rows are appended to /scratch/zr2245/analysis/generic-codecs.txt. Restartable: existing rows / eval.json are skipped.
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
AN=/scratch/zr2245/analysis; G=$AN/generic-v1; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; TAB=$AN/generic-codecs.txt
SCENE=${1:?usage: run_generic_v1.sh room|garden}; TEX=${TEX:-"4096 8192"}
case $SCENE in
  room)   RUN=/scratch/zr2245/room-fixed-sharing-0N4OSU/run/original; DATA=/scratch/zr2245/datasets/m360/room;   IMAGES=images_2; QP=14;;
  garden) RUN=/scratch/zr2245/garden-vis-collapse/original;            DATA=/scratch/zr2245/datasets/m360/garden; IMAGES=images_4; QP=15;;
  *)      RUN=${RUN:?set RUN=<dir with point_cloud/iteration_30000>}; DATA=${DATA:?}; IMAGES=${IMAGES:-images_4}; QP=${QP:?};;
esac
CK=$RUN/point_cloud/iteration_30000/point_cloud_state_dict.pt; OUT=$G/$SCENE; mkdir -p $OUT
for f in $CK $DATA/$IMAGES/. $LAB/upstream/train.py $G/generic_codec_v1.py $G/eval_ckpt_v1.py; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
python -c "import DracoPy, xatlas, zstandard, PIL" 2>/dev/null || { echo "=== pip install DracoPy xatlas zstandard pillow ==="; pip install DracoPy xatlas zstandard pillow > $OUT/pip.log 2>&1 || { echo "FAILED: pip (see $OUT/pip.log)"; tail -5 $OUT/pip.log; exit 1; }; }
python -c "import DracoPy, xatlas, zstandard" || { echo "FAILED: imports after pip"; exit 1; }
echo "=== $SCENE: CPU rows (zstd, Draco, texture $TEX) -> $OUT  $(date) ==="
python $G/generic_codec_v1.py --ckpt $CK --qp $QP --out $OUT --scene $SCENE --texture $TEX > $OUT/codec.log 2>&1 || { echo "WARNING: generic_codec_v1.py exited non-zero (rows written so far are kept)"; tail -15 $OUT/codec.log; }
grep '^\[.*ROW' $OUT/codec.log
[ -f $OUT/rows.json ] || { echo "FAILED: no rows.json"; tail -20 $OUT/codec.log; exit 1; }
echo "=== $SCENE: GPU eval of decoded checkpoints  $(date) ==="
ev () { # label ckpt
  [ -f $OUT/$1.eval.json ] && { echo "skip eval $1 (exists)"; return; }
  python $G/eval_ckpt_v1.py --lab $LAB --model $RUN --dataset $DATA --images $IMAGES --ckpt $2 --out $OUT/$1.eval.json --label "$SCENE $1" > $OUT/$1.eval.log 2>&1 \
    && grep '^EVAL' $OUT/$1.eval.log || { echo "FAILED: eval $1"; tail -8 $OUT/$1.eval.log; }
}
ev original $CK
for row in quant_ply_zstd19 draco_full_sh8 draco_full_sh16 draco_rgb8; do [ -f $OUT/$row/point_cloud_state_dict.pt ] && ev $row $OUT/$row/point_cloud_state_dict.pt || echo "no checkpoint for $row"; done
python - <<ASM
import json, os
out='$OUT'; rows=json.load(open(out+'/rows.json')); ev={}
for f in os.listdir(out):
    if f.endswith('.eval.json'): ev[f[:-10]]=json.load(open(out+'/'+f))
o=ev.get('original'); lines=[]
def q(r): return f"{r['psnr']:.3f} / {r['ssim']:.4f} / {r['lpips']:.4f}" if r else 'n/a'
lines.append(f"$SCENE original | V {o['vertices']} F {o['faces']} | test {q(o)}" if o else "$SCENE original | eval missing")
for name,r in rows.items():
    mb=r.get('MB', r.get('total_jpeg_MB')); e=ev.get(name) or (ev.get('draco_rgb8') if name.startswith('texture_') and 'total_jpeg_MB' in r else None)
    extra=f" | PNG total {r['total_png_MB']} MB" if 'total_png_MB' in r else ''
    dv=f" | decoded V {r['decoded_vertices']}" if 'decoded_vertices' in r else ''
    lines.append(f"$SCENE {name} | {mb} MB{extra}{dv} | test {q(e)}" + (f" ({e['psnr']-o['psnr']:+.3f} dB)" if e and o else '') + f" | {r.get('note','')}")
open('$TAB','a').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
ASM
echo "generic $SCENE done $(date)  -> $TAB"
