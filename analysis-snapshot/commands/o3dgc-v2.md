# o3dgc-v2 (2026-11-01): Open3DGC mirror fix

`amd/rest3d` 已从 GitHub 消失（"Repository not found"），v1 的 runner 死在 clone 这一步。v2 只换 runner（driver / codec py 不变，仍在 `$AN/o3dgc-v1/`）：

- 按顺序试三个 mirror：`rbsheth/Open3DGC`（独立拷贝，就是原 `server/o3dgc` 目录）、`kmammou/rest3d`（原作者 Khaled Mamou 的 fork，`server/o3dgc`）、`KhronosGroup/COLLADA2GLTF`（历史上把 o3dgc 作为依赖带着）。
- 不再假设目录结构：在 clone 里 `find` 名为 `o3dgc_common_lib` 的目录，它的上一级就是 `src`；头文件 = 所有 `inc` 目录，库源码 = `*lib*/**/*.cpp`。
- 三个都失败时：浏览器下载 https://github.com/rbsheth/Open3DGC/archive/refs/heads/master.zip ，scp 到 Torch，`O3DGC_ZIP=<zip路径> bash run_o3dgc_v2.sh room`；或者已经有源码树就 `O3DGC_SRC=<dir>`。
- 编译失败时仍会打印头文件里的 API 行（贴给 Claude）。

CPU 节点即可（cpu-torch）。GPU 评估以后 `EVAL=1` 重跑同一行。

## §1 paste block（写入 v2 runner，跑 room）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/o3dgc-v1
cat > $AN/o3dgc-v1/run_o3dgc_v2.sh <<'EOS'
#!/bin/bash
# run_o3dgc_v2.sh (2026-11-01, VERSION o3dgc-v2: amd/rest3d is gone -> mirror list + layout-agnostic source discovery): MPEG-4 SC3DMC/TFAN (Open3DGC) rows on one scene's ORIGINAL mesh.  CPU; eval on GPU when available.
#   bash /scratch/zr2245/analysis/o3dgc-v1/run_o3dgc_v2.sh room      (any of the 11 scenes; EVAL=1 forces the GPU eval, EVAL=0 skips it)
# Steps: clone Open3DGC from the first working mirror once (or O3DGC_SRC=<dir> / O3DGC_ZIP=<zip> you fetched yourself) -> build o3dgc_driver with g++ once -> encode/decode -> rows.json in generic-v1/<scene>/ ->
# (GPU) eval_ckpt_v1.py on the decoded checkpoints -> rows appended to /scratch/zr2245/analysis/generic-codecs.txt.  Restartable.
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
AN=/scratch/zr2245/analysis; O=$AN/o3dgc-v1; G=$AN/generic-v1; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; TAB=$AN/generic-codecs.txt; D=/scratch/zr2245/datasets
SCENE=${1:?usage: run_o3dgc_v2.sh <scene>}
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
# --- 1. Open3DGC sources (once).  amd/rest3d was deleted; try mirrors in order.  Override: O3DGC_SRC=<dir containing o3dgc_common_lib somewhere> or O3DGC_ZIP=<zip of such a tree>.
SRC=$O/open3dgc
find_o3dgc () { find "$1" -type d -name o3dgc_common_lib 2>/dev/null | head -1 | xargs -r dirname; }   # -> the "src" dir that holds o3dgc_*_lib
if [ ! -d $SRC/src ]; then
  rm -f $SRC; ROOT=""
  if [ -n "$O3DGC_SRC" ]; then ROOT=$(find_o3dgc "$O3DGC_SRC"); [ -n "$ROOT" ] || { echo "FAILED: O3DGC_SRC=$O3DGC_SRC holds no o3dgc_common_lib"; exit 1; }
  elif [ -n "$O3DGC_ZIP" ]; then mkdir -p $O/zip && unzip -q -o "$O3DGC_ZIP" -d $O/zip && ROOT=$(find_o3dgc $O/zip); [ -n "$ROOT" ] || { echo "FAILED: O3DGC_ZIP=$O3DGC_ZIP holds no o3dgc_common_lib"; exit 1; }
  else
    for M in rbsheth/Open3DGC kmammou/rest3d KhronosGroup/COLLADA2GLTF; do
      echo "=== cloning Open3DGC mirror $M $(date) ==="; DEST=$O/mirror-$(echo $M | tr / _); rm -rf $DEST
      if git clone --depth 1 https://github.com/$M $DEST >> $O/clone.log 2>&1; then ROOT=$(find_o3dgc $DEST); [ -n "$ROOT" ] && { echo "found o3dgc sources: $ROOT"; break; }; echo "mirror $M has no o3dgc_common_lib, next"; else echo "mirror $M: clone failed ($(tail -1 $O/clone.log))"; fi
    done
    [ -n "$ROOT" ] || { echo "FAILED: no mirror worked (see $O/clone.log). Download https://github.com/rbsheth/Open3DGC/archive/refs/heads/master.zip in a browser, copy it to Torch, rerun with O3DGC_ZIP=<path>"; exit 1; }
  fi
  mkdir -p $SRC && ln -sfn "$ROOT" $SRC/src
fi
ls $SRC/src | grep -q o3dgc_common_lib || { echo "FAILED: $SRC/src is not an o3dgc tree:"; ls $SRC/src; exit 1; }
INC=$(find -L $SRC/src -type d -name inc | sed 's/^/-I/' | tr '\n' ' '); LIBSRC=$(find -L $SRC/src -path '*lib*' -name '*.cpp' | grep -v -i test | tr '\n' ' ')
[ -n "$INC" ] && [ -n "$LIBSRC" ] || { echo "FAILED: no inc dirs / lib sources under $SRC/src (layout differs); listing:"; find -L $SRC/src -maxdepth 2 | head -40; exit 1; }
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
EOS
cd $AN/o3dgc-v1 && bash -n run_o3dgc_v2.sh && echo "syntax ok" && bash run_o3dgc_v2.sh room 2>&1 | tee $AN/o3dgc-v1/room-v2.log | tail -40
```

成功的标志：`found o3dgc sources: ...`、`driver built: ...`、然后两行 `ROW ...`。

## §2 其余场景（room 通过后）

```bash
AN=/scratch/zr2245/analysis
for S in garden kitchen counter bonsai bicycle stump flowers treehill truck train; do bash $AN/o3dgc-v1/run_o3dgc_v2.sh $S 2>&1 | tee $AN/o3dgc-v1/$S-v2.log | grep -i 'ROW\|FAILED\|done'; done
```

## §3 看结果

```bash
grep o3dgc /scratch/zr2245/analysis/generic-codecs.txt
```
