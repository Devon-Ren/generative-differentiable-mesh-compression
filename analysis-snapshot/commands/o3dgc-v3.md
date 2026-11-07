# o3dgc-v3 (2026-11-01): + MPEG geometry-only row + MPEG-geometry texture row

Adds to the Room/garden Table 1 (CPU, minutes; driver unchanged, already built in `$AN/o3dgc-v1/`):

- `o3dgc_tfan_geo`: positions q + faces, no attributes → the MPEG geometry budget. Ablation "MPEG geometry + our SH streams" = this row + (our level-1.0 bytes − our connectivity/XYZ streams, ≈ 30.9 − 16 MB on Room).
- `o3dgc_texture_geo_uv` + `o3dgc_texture_8192`: the texture route with MPEG geometry (atlas mesh + 12-bit UV through Open3DGC, colour = the existing baked JPEG). Quality = the draco_rgb8 / texture row by construction (25.29 on Room). Needs `atlas.npz` + `texture_8192` in `generic-v1/<scene>/rows.json` — only room and garden have them; other scenes get the geo row only and print a skip line.
- The v1 rows (sh8int / sh8f) are kept and skipped if present.

## §1 paste block（写两个文件，跑 room + garden）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/o3dgc-v1
cat > $AN/o3dgc-v1/o3dgc_codec_v2.py <<'EOS'
"""o3dgc_codec_v2.py (2026-11-01, VERSION o3dgc-v2 = v1 rows + geometry-only row + MPEG-geometry texture row): MPEG-4 Part 16 SC3DMC / TFAN (Open3DGC) rows on the SAME MeshSplatting asset,
in the generic-v1 row format (rows.json next to the generic rows; decoded checkpoints for eval_ckpt_v1.py).  CPU only.
Rows:
  o3dgc_tfan_sh8int   positions q<qp> (parallelogram) + SH pre-quantised per channel to uint8 (same grid as quant_ply_zstd19) carried as
                      six 8-dim INTEGER attributes (lossless, differential prediction) -> decoded quality == quant_ply_zstd19 row.
  o3dgc_tfan_sh8f     positions q<qp> + SH as six 8-dim FLOAT attributes quantised by Open3DGC itself at 8 bits (its range rule) -> own checkpoint.
  o3dgc_tfan_geo      positions q<qp> + faces only (no attributes): the MPEG geometry budget, for the "MPEG geometry + our SH streams" ablation.
  o3dgc_texture_<R>   texture route with MPEG geometry: Open3DGC(positions[vmapping] q<qp>, atlas faces, UV as a 2-dim float attribute at 12 bit)
                      + the JPEG bytes of the existing texture_<R> row (generic_codec_v1 --texture); needs atlas.npz + texture_<R> in rows.json.
                      Quality = draco_rgb8 / texture_<R> row by construction (vertex colours only).
usage: python o3dgc_codec_v2.py --ckpt <state_dict.pt> --qp 14 --out <generic-v1/room dir> --driver <o3dgc_driver> --scene room"""
import argparse, json, struct, subprocess, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'generic-v1'))
from generic_codec_v1 import load_ckpt, sh_quant, sh_dequant, save_ckpt, log  # noqa: E402
VERSION = 'o3dgc-v2 2026-11-01'
def write_bundle(path, xyz, faces, sh, mode):
    with open(path, 'wb') as f:
        f.write(b'O3DGCB01' + struct.pack('<IIII', len(xyz), len(faces), mode, sh.shape[1]))
        f.write(np.ascontiguousarray(xyz, np.float32).tobytes()); f.write(np.ascontiguousarray(faces, np.int32).tobytes())
        f.write(np.ascontiguousarray(sh, np.float32 if mode == 0 else np.uint8).tobytes())
def read_bundle(path):
    b = open(path, 'rb').read(); assert b[:8] == b'O3DGCB01', 'bad decoded bundle'
    V, F, mode, D = struct.unpack('<IIII', b[8:24]); o = 24
    xyz = np.frombuffer(b, np.float32, V * 3, o).reshape(V, 3); o += V * 12
    faces = np.frombuffer(b, np.int32, F * 3, o).reshape(F, 3); o += F * 12
    sh = (np.frombuffer(b, np.float32, V * D, o) if mode == 0 else np.frombuffer(b, np.uint8, V * D, o)).reshape(V, D)
    return xyz.copy(), faces.astype(np.int64), sh.copy(), mode
def main():
    p = argparse.ArgumentParser(); p.add_argument('--ckpt', required=True); p.add_argument('--qp', type=int, required=True); p.add_argument('--out', required=True)
    p.add_argument('--driver', required=True); p.add_argument('--scene', default=''); p.add_argument('--qattr', type=int, default=8); a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows_path = out / 'rows.json'; rows = json.load(open(rows_path)) if rows_path.exists() else {}
    def put(name, **kw):
        rows[name] = dict(name=name, scene=a.scene, qp=a.qp, **kw); json.dump(rows, open(rows_path, 'w'), indent=1); log(f'ROW {name}: ' + ', '.join(f'{k}={v}' for k, v in kw.items() if k != 'ckpt'))
    log(VERSION, '| loading', a.ckpt); m = load_ckpt(a.ckpt); xyz, faces, dc, rest, w = m['xyz'], m['faces'], m['dc'], m['rest'], m['weight']
    V, F = len(xyz), len(faces); sh = np.concatenate([dc, rest], 1); wlogit = float(np.median(w)) if w is not None else 7.0; MB = lambda b: round(b / 1e6, 3)
    log(f'{V} vertices, {F} faces')
    for name, mode in (('o3dgc_tfan_sh8int', 1), ('o3dgc_tfan_sh8f', 0)):
        if name in rows: log('skip', name, '(exists)'); continue
        d = out / name; d.mkdir(exist_ok=True); bundle = d / 'in.bundle'; stream = d / 'mesh.o3dgc'; dec = d / 'out.decoded'
        if mode == 1: sq, slo, shi = sh_quant(sh, 8); write_bundle(bundle, xyz, faces, sq, 1)
        else: write_bundle(bundle, xyz, faces, sh, 0)
        t = time.time(); r = subprocess.run([a.driver, str(bundle), str(stream), str(dec), str(a.qp), str(a.qattr)], capture_output=True, text=True); enc = time.time() - t
        log(r.stdout.strip()); 
        if r.returncode != 0: log('FAILED driver:', r.stderr.strip()[-800:]); json.dump(rows, open(rows_path, 'w'), indent=1); sys.exit(1)
        xyz2, faces2, sh2, _ = read_bundle(dec); n = stream.stat().st_size + (96 * 4 if mode == 1 else 0)
        shr = sh_dequant(sh2, slo, shi, 8) if mode == 1 else sh2.astype(np.float32)
        ck = d / 'point_cloud_state_dict.pt'; save_ckpt(ck, xyz2, faces2, shr[:, :3], shr[:, 3:], wlogit)
        note = (f'MPEG-4 SC3DMC TFAN (Open3DGC): positions q{a.qp} parallelogram + 48-ch uint8 SH as 6x8-dim int attributes (lossless; same grid as quant_ply_zstd19)' if mode == 1
                else f'MPEG-4 SC3DMC TFAN (Open3DGC): positions q{a.qp} + 48-ch SH as 6x8-dim float attributes, Open3DGC {a.qattr}-bit quantisation')
        put(name, bytes=n, MB=MB(n), decoded_vertices=int(len(xyz2)), vertex_split=round(len(xyz2) / V, 3), encode_decode_s=round(enc, 1), note=note, ckpt=str(ck))
        bundle.unlink(); dec.unlink()
    # --- geometry only (mode 1, D = 0): MPEG budget for connectivity + positions
    if 'o3dgc_tfan_geo' in rows: log('skip o3dgc_tfan_geo (exists)')
    else:
        d = out / 'o3dgc_tfan_geo'; d.mkdir(exist_ok=True); bundle = d / 'in.bundle'; stream = d / 'mesh.o3dgc'; dec = d / 'out.decoded'
        write_bundle(bundle, xyz, faces, np.zeros((V, 0), np.uint8), 1)
        t = time.time(); r = subprocess.run([a.driver, str(bundle), str(stream), str(dec), str(a.qp), str(a.qattr)], capture_output=True, text=True); enc = time.time() - t
        log(r.stdout.strip())
        if r.returncode != 0: log('FAILED driver (geo):', r.stderr.strip()[-800:]); sys.exit(1)
        xyz2, faces2, _, _ = read_bundle(dec); n = stream.stat().st_size
        put('o3dgc_tfan_geo', bytes=n, MB=MB(n), decoded_vertices=int(len(xyz2)), decoded_faces=int(len(faces2)), vertex_split=round(len(xyz2) / V, 3), encode_decode_s=round(enc, 1),
            note=f'MPEG-4 SC3DMC TFAN (Open3DGC): positions q{a.qp} + faces only, no attributes (geometry budget)')
        bundle.unlink(); dec.unlink()
    # --- texture route with MPEG geometry: atlas mesh + 12-bit UV through Open3DGC, colour = the existing baked JPEG
    atlas = out / 'atlas.npz'; tex_rows = [k for k in rows if k.startswith('texture_') and 'jpeg_bytes' in rows[k]]
    if not atlas.exists() or not tex_rows: log('no atlas.npz / texture_<R> rows in', out, '-> skipping o3dgc_texture rows (run generic_codec_v1 --texture first)')
    else:
        z = np.load(atlas); vmapping, indices, uvs = z['vmapping'], z['indices'], z['uvs'].astype(np.float32)
        name_geo = 'o3dgc_texture_geo_uv'
        if name_geo in rows: log('skip', name_geo, '(exists)')
        else:
            d = out / name_geo; d.mkdir(exist_ok=True); bundle = d / 'in.bundle'; stream = d / 'mesh.o3dgc'; dec = d / 'out.decoded'
            write_bundle(bundle, xyz[vmapping], indices.astype(np.int32), uvs, 0)
            t = time.time(); r = subprocess.run([a.driver, str(bundle), str(stream), str(dec), str(a.qp), '12'], capture_output=True, text=True); enc = time.time() - t
            log(r.stdout.strip())
            if r.returncode != 0: log('FAILED driver (texture geo+uv):', r.stderr.strip()[-800:]); sys.exit(1)
            xyz2, faces2, uv2, _ = read_bundle(dec); n = stream.stat().st_size
            put(name_geo, bytes=n, MB=MB(n), decoded_vertices=int(len(xyz2)), atlas_vertices=int(len(vmapping)), encode_decode_s=round(enc, 1),
                note=f'Open3DGC positions q{a.qp} + atlas faces + 12-bit UV (2-dim float attribute) on the xatlas mesh')
            bundle.unlink(); dec.unlink()
        geo = rows[name_geo]['bytes']
        for k in tex_rows:
            name = 'o3dgc_' + k
            if name in rows: log('skip', name, '(exists)'); continue
            jb = rows[k]['jpeg_bytes']; put(name, bytes=geo + jb, MB=MB(geo + jb), geo_bytes=geo, jpeg_bytes=jb,
                note=f'texture route with MPEG geometry: {name_geo} + JPEG of {k}; quality = draco_rgb8 row (vertex colours only, no SH)')
    log('done ->', rows_path)
if __name__ == '__main__': main()
EOS
cat > $AN/o3dgc-v1/run_o3dgc_v3.sh <<'EOS'
#!/bin/bash
# run_o3dgc_v3.sh (2026-11-01, VERSION o3dgc-v3 = v2 runner + o3dgc_codec_v2.py: geometry-only row + MPEG-geometry texture row): MPEG-4 SC3DMC/TFAN (Open3DGC) rows on one scene's ORIGINAL mesh.  CPU; eval on GPU when available.
#   bash /scratch/zr2245/analysis/o3dgc-v1/run_o3dgc_v3.sh room      (any of the 11 scenes; EVAL=1 forces the GPU eval, EVAL=0 skips it)
# Steps: clone Open3DGC from the first working mirror once (or O3DGC_SRC=<dir> / O3DGC_ZIP=<zip> you fetched yourself) -> build o3dgc_driver with g++ once -> encode/decode -> rows.json in generic-v1/<scene>/ ->
# (GPU) eval_ckpt_v1.py on the decoded checkpoints -> rows appended to /scratch/zr2245/analysis/generic-codecs.txt.  Restartable.
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
AN=/scratch/zr2245/analysis; O=$AN/o3dgc-v1; G=$AN/generic-v1; LAB=/scratch/zr2245/meshsplatting-original-lossless-v1; TAB=$AN/generic-codecs.txt; D=/scratch/zr2245/datasets
SCENE=${1:?usage: run_o3dgc_v3.sh <scene>}
case $SCENE in
  room)   RUN=/scratch/zr2245/room-fixed-sharing-0N4OSU/run/original; DATA=$D/m360/room; IMAGES=images_2; QP=14;;
  kitchen|counter|bonsai) RUN=/scratch/zr2245/$SCENE-vis-collapse/original; DATA=$D/m360/$SCENE; IMAGES=images_2; QP=$([ $SCENE = kitchen ] && echo 16 || echo 15);;
  garden|flowers)   RUN=/scratch/zr2245/$SCENE-vis-collapse/original; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=15;;
  bicycle|stump|treehill) RUN=/scratch/zr2245/$SCENE-vis-collapse/original; DATA=$D/m360/$SCENE; IMAGES=images_4; QP=16;;
  truck|train) RUN=/scratch/zr2245/$SCENE-vis-collapse/original; DATA=$D/tandt/$SCENE; IMAGES=images; QP=18;;
  *) RUN=${RUN:?set RUN=<dir with point_cloud/iteration_30000>}; DATA=${DATA:?}; IMAGES=${IMAGES:-images_4}; QP=${QP:?};;
esac
CK=$RUN/point_cloud/iteration_30000/point_cloud_state_dict.pt; OUT=$G/$SCENE; mkdir -p $OUT $O
for f in $CK $G/generic_codec_v1.py $G/eval_ckpt_v1.py $O/o3dgc_codec_v2.py $O/o3dgc_driver.cpp; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
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
python $O/o3dgc_codec_v2.py --ckpt $CK --qp $QP --out $OUT --driver $O/o3dgc_driver --scene $SCENE > $OUT/o3dgc.log 2>&1 || { echo "FAILED: o3dgc_codec_v2.py"; tail -15 $OUT/o3dgc.log; exit 1; }
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
for name in ('o3dgc_tfan_sh8int','o3dgc_tfan_sh8f','o3dgc_tfan_geo') + tuple(k for k in rows if k.startswith('o3dgc_texture_') and k != 'o3dgc_texture_geo_uv'):
    r=rows.get(name); e=ev.get(name)
    if r: lines.append(f"$SCENE {name} | {r['MB']} MB | decoded V {r.get('decoded_vertices', '-')} | test {q(e)}" + (f" ({e['psnr']-o['psnr']:+.3f} dB)" if e and o else '') + f" | {r['note']}")
open('$TAB','a').write('\n'.join(lines)+'\n'); print('\n'.join(lines))
ASM
echo "o3dgc $SCENE done $(date) -> $TAB"
EOS
cd $AN/o3dgc-v1 && python -m py_compile o3dgc_codec_v2.py && bash -n run_o3dgc_v3.sh && echo "syntax ok"
for S in room garden; do bash run_o3dgc_v3.sh $S 2>&1 | tee $AN/o3dgc-v1/$S-v3.log | grep -i 'ROW\|FAILED\|skip\|done'; done
```

## §2 其余 9 个场景（只会多出 geo 行；sh8int/sh8f 若已跑过会跳过）

```bash
AN=/scratch/zr2245/analysis
for S in kitchen counter bonsai bicycle stump flowers treehill truck train; do bash $AN/o3dgc-v1/run_o3dgc_v3.sh $S 2>&1 | tee $AN/o3dgc-v1/$S-v3.log | grep -i 'ROW\|FAILED\|done'; done
```

## §3 看结果

```bash
grep -h "ROW" /scratch/zr2245/analysis/o3dgc-v1/*-v3.log | sed 's/.*ROW //' | cut -c1-120
```
