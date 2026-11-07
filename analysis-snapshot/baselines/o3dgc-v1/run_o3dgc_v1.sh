#!/bin/bash
# run_o3dgc_v1.sh (2026-11-01, VERSION o3dgc-v1): MPEG-4 SC3DMC/TFAN (Open3DGC) rows on one scene's ORIGINAL mesh.  CPU; eval on GPU when available.
#   bash /scratch/zr2245/analysis/o3dgc-v1/run_o3dgc_v1.sh room      (any of the 11 scenes; EVAL=1 forces the GPU eval, EVAL=0 skips it)
# Steps: clone Open3DGC (amd/rest3d, server/o3dgc) once -> build o3dgc_driver with g++ once -> encode/decode -> rows.json in generic-v1/<scene>/ ->
# (GPU) eval_ckpt_v1.py on the decoded checkpoints -> rows appended to /scratch/zr2245/analysis/generic-codecs.txt.  Restartable.
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
AN=/scratch/zr2245/analysis; O=$AN/o3dgc-v1; G=$AN/generic-v1; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; TAB=$AN/generic-codecs.txt; D=/scratch/zr2245/datasets
SCENE=${1:?usage: run_o3dgc_v1.sh <scene>}
case $SCENE in
  room)   RUN=/scratch/zr2245/room-fixed-sharing-0N4OSU/run/original; DATA=$D/m360/room; IMAGES=images_2; QP=14;;
  kitchen|counter|bonsai) RUN=/scratch/zr2245/$SCENE-vis-collapse/original; DATA=$D/m360/$SCENE; IMAGES=images_2; QP=$([ $SCENE = kitchen ] && echo 16 || echo 15);;
  garden|flowers)   RUN=/scratch/zr2245/$SCENE-vis-collapse/original; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=15;;
  bicycle|stump|treehill) RUN=/scratch/zr2245/$SCENE-vis-collapse/original; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=16;;
  truck|train) RUN=/scratch/zr2245/$SCENE-vis-collapse/original; DATA=$D/tandt/$SCENE; IMAGES=images; QP=18;;
  *) RUN=${RUN:?set RUN=<dir with point_cloud/iteration_30000>}; DATA=${DATA:?}; IMAGES=${IMAGES:-images_4}; QP=${QP:?};;
esac
CK=$RUN/point_cloud/iteration_30000/point_cloud_state_dict.pt; OUT=$G/$SCENE; mkdir -p $OUT $O
for f in $CK $G/generic_codec_v1.py $G/eval_ckpt_v1.py $O/o3dgc_codec_v1.py $O/o3dgc_driver.cpp; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
# --- 1. Open3DGC sources (once)
SRC=$O/open3dgc
if [ ! -d $SRC/src ]; then
  echo "=== cloning Open3DGC (amd/rest3d) $(date) ==="
  git clone --depth 1 https://github.com/amd/rest3d $O/rest3d > $O/clone.log 2>&1 || { echo "FAILED: git clone amd/rest3d (see $O/clone.log); tell Claude, a mirror is needed"; tail -3 $O/clone.log; exit 1; }
  [ -d $O/rest3d/server/o3dgc/src ] || { echo "FAILED: server/o3dgc/src not in the clone; listing:"; ls $O/rest3d $O/rest3d/server 2>/dev/null; exit 1; }
  ln -sfn $O/rest3d/server/o3dgc $SRC
fi
INC=$(find $SRC/src -type d -name inc | sed 's/^/-I/' | tr '\n' ' '); LIBSRC=$(find $SRC/src -path '*lib*' -name '*.cpp' | grep -v -i test | tr '\n' ' ')
# --- 2. driver (once); on failure print the API lines Claude needs
if [ ! -x $O/o3dgc_driver ]; then
  echo "=== building o3dgc_driver $(date) ==="; echo "headers: $INC"; echo "lib sources: $(echo $LIBSRC | wc -w) files"
  g++ -O2 -std=c++11 -w $INC $O/o3dgc_driver.cpp $LIBSRC -o $O/o3dgc_driver > $O/build.log 2>&1 || {
    echo "FAILED: driver build (see $O/build.log). First errors:"; grep -m 12 -n "error" $O/build.log
    echo "--- API lines (paste these to Claude) ---"
    grep -h -n "DecodeP\|SetFloatAttributeQuantBits\|SetNumFloatAttributes\|SetNumIntAttributes\|SetIntAttribute(\|SetFloatAttribute(\|SetIntAttributeDim\|SetFloatAttributeDim\|SetNFloatAttribute\|SetNIntAttribute\|ComputeMinMax\|SetCoordIndex\|SetEncodeMode\|SetStreamType\|GetBuffer\|GetSize\|MAX_DIM\|PREDICTION\|TYPE_UNKOWN\|TYPE_UNKNOWN" $(find $SRC/src -name '*.h') | head -60
    exit 1; }
  echo "driver built: $O/o3dgc_driver"
fi
# --- 3. encode/decode (CPU)
echo "=== $SCENE: Open3DGC rows (qp $QP) $(date) ==="
python $O/o3dgc_codec_v1.py --ckpt $CK --qp $QP --out $OUT --driver $O/o3dgc_driver --scene $SCENE > $OUT/o3dgc.log 2>&1 || { echo "FAILED: o3dgc_codec_v1.py"; tail -15 $OUT/o3dgc.log; exit 1; }
grep 'ROW\|O3DGC' $OUT/o3dgc.log
# --- 4. eval on GPU (skipped on a CPU node unless EVAL=1)
if [ "${EVAL:-auto}" = "0" ] || { [ "${EVAL:-auto}" = "auto" ] && ! nvidia-smi > /dev/null 2>&1; }; then echo "no GPU here: rerun on a GPU node with EVAL=1 to render the decoded checkpoints (rows are saved)"; exit 0; fi
ev () { [ -f $OUT/$1.eval.json ] && { echo "skip eval $1 (exists)"; return; }
  python $G/eval_ckpt_v1.py --lab $LAB --model $RUN --dataset $DATA --images $IMAGES --ckpt $2 --out $OUT/$1.eval.json --label "$SCENE $1" > $OUT/$1.eval.log 2>&1 && grep '^EVAL' $OUT/$1.eval.log || { echo "FAILED: eval $1"; tail -8 $OUT/$1.eval.log; }; }
ev original $CK
for row in o3dgc_tfan_sh8int o3dgc_tfan_sh8f; do [ -f $OUT/$row/point_cloud_state_dict.pt ] && ev $row $OUT/$row/point_cloud_state_dict.pt; done
python - <<ASM
import json, os
out='$OUT'; rows=json.load(open(out+'/rows.json')); ev={f[:-10]: json.load(open(out+'/'+f)) for f in os.listdir(out) if f.endswith('.eval.json')}
o=ev.get('original'); q=lambda r: f"{r['psnr']:.3f} / {r['ssim']:.4f} / {r['lpips']:.4f}" if r else 'n/a'; lines=[]
for name in ('o3dgc_tfan_sh8int','o3dgc_tfan_sh8f'):
    r=rows.get(name); e=ev.get(name)
    if r: lines.append(f"$SCENE {name} | {r['MB']} MB | decoded V {r['decoded_vertices']} | test {q(e)}" + (f" ({e['psnr']-o['psnr']:+.3f} dB)" if e and o else '') + f" | {r['note']}")
open('$TAB','a').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
ASM
echo "o3dgc $SCENE done $(date) -> $TAB"
