# node-batch39-v1 (2026-11-14): encode / decode time, our codec and zstd / Draco / TFAN on the same cap-trained mesh, 11 scenes, CPU only

Both timing scripts were run locally on synthetic meshes with the real codec code (round trip exact).
Expected `FILES OK: 56d176a0 f091c8b0 7176d06c`.

## Paste (writes the three files)

```bash
AN=/scratch/zr2245/analysis
cat > $AN/codec_time_v1.py <<'PYOURS39'
"""codec_time_v1.py (2026-11-14): encode and decode time of our codec on ONE delivered mesh.  CPU only.  Run inside the stage2 env:
    bash $PKG/stage2_run.sh $AN/codec_time_v1.py --shared <deliver>/shared-final.pt --xyz q15 --dc q7 --out <json> --label "garden cap1073000"
Input is the shared representation the delivery wrote (mesh + base colour + codebooks + indices), i.e. what the encoder receives.
Timed, with the same calls as stage2 runtime.export_asset / codec topology.read_asset:
  compile  : building the native coders (once per machine; not part of encode or decode)
  encode   : quantise + reorder vertices + predict + entropy-code every stream + write the one file
  decode   : read the file + entropy-decode every stream (repeated, the fastest run is reported) ; restore = de-quantise to float arrays
The decoded mesh is compared with the encoder's input order exactly as export_asset does."""
import argparse, json, os, platform, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch

def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage2', required=True); p.add_argument('--shared', required=True); p.add_argument('--out', required=True)
    p.add_argument('--xyz', default='q14'); p.add_argument('--dc', default='q7'); p.add_argument('--label', default=''); p.add_argument('--repeats', type=int, default=3)
    a = p.parse_args(); stage2 = Path(a.stage2).resolve()
    for d in (stage2 / 'renderer', stage2, stage2 / 'codec'): sys.path.insert(0, str(d))
    import delivery, topology as tc
    from codec_benchmark import pack, unpack, equal
    delivery.XYZ_MODES = tuple(dict.fromkeys(delivery.XYZ_MODES + ('q15', 'q13', 'q12'))); delivery.DC_MODES = tuple(dict.fromkeys(delivery.DC_MODES + ('q7', 'q6', 'q5', 'q4')))
    wire = torch.load(a.shared, map_location='cpu', weights_only=False); wire.setdefault('opacity_floor', .999)
    V, F = len(wire['triangles_points']), len(wire['_triangle_indices'])
    work = Path(tempfile.mkdtemp(prefix='codec-time-', dir=str(Path(a.out).resolve().parent))); native = work / 'native'; T = {}
    try:
        t = time.time(); exe = tc.lc.native(native); edge = tc.edge_native(native); ctx = tc.ic.native(native); T['compile'] = time.time() - t
        enc = []
        for i in range(2):   # the second run has warm files; the faster one is reported
            tmp = work / f'enc{i}'; tmp.mkdir(); t = time.time()
            quant = delivery.prepare(wire, a.xyz, a.dc, True, 'const'); ordered, perm = tc.reorder(quant, 'strip-fetch')
            pred = tc.lc.graph(exe, ordered, tmp); encoded = tc.encode(ordered, 'edge-arithmetic', 1, exe, edge, tmp, pred, ctx)
            path = work / f'asset{i}.mslb'; pack(encoded, path, 'auto'); enc.append(time.time() - t); del encoded
        size = path.stat().st_size; dec, res = [], []
        for i in range(max(1, a.repeats)):
            tmp = work / f'dec{i}'; tmp.mkdir(); t = time.time(); decoded = tc.read_asset(path, native, tmp); dec.append(time.time() - t)
            t = time.time(); state = delivery.restore(decoded); res.append(time.time() - t)
        equal({k: v for k, v in ordered.items() if k != '_triangle_indices'}, {k: decoded[k] for k in ordered if k != '_triangle_indices'})
        if not np.array_equal(tc.cyclic(ordered['_triangle_indices'].numpy()), tc.cyclic(decoded['_triangle_indices'].numpy())): raise AssertionError('topology roundtrip')
    finally:
        shutil.rmtree(work, ignore_errors=True)
    out = dict(label=a.label, vertices=V, faces=F, file_MB=round(size / 1e6, 3), xyz=a.xyz, dc=a.dc, compile_s=round(T['compile'], 2), encode_s=round(min(enc), 2), encode_runs=[round(x, 2) for x in enc],
               decode_s=round(min(dec), 2), decode_runs=[round(x, 2) for x in dec], restore_s=round(min(res), 3), threads=os.environ.get('OMP_NUM_THREADS', ''), cpu=platform.processor() or platform.machine(), host=platform.node(), roundtrip='exact')
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"TIME {a.label} | V {V} F {F} | {out['file_MB']} MB | ours encode {out['encode_s']} s | decode {out['decode_s']} s (+ restore {out['restore_s']} s) | compile once {out['compile_s']} s | roundtrip exact", flush=True)

if __name__ == '__main__':
    main()
PYOURS39
cat > $AN/generic_time_v1.py <<'PYGEN39'
"""generic_time_v1.py (2026-11-14): encode and decode time of the generic codecs on ONE mesh, same settings as generic_rd_v1.py.  CPU only.
  zstd  : quantised arrays + int32 faces, zstd -19 (all threads), compress / decompress
  draco : Edgebreaker level 10, positions q<qp> + faces + 48 SH channels as a generic attribute, encode / decode
  tfan  : MPEG-4 SC3DMC / TFAN through the Open3DGC driver: ONE call that encodes and decodes (the driver does both), reported as one number
  python generic_time_v1.py --ckpt <mesh .pt> --qp 15 --bits 8 --out <json> --generic <dir with generic_codec_v1.py> [--driver <o3dgc_driver>]"""
import argparse, json, os, platform, struct, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser(); p.add_argument('--ckpt', required=True); p.add_argument('--qp', type=int, required=True); p.add_argument('--out', required=True)
p.add_argument('--bits', type=int, default=8); p.add_argument('--driver', default=''); p.add_argument('--generic', required=True); p.add_argument('--label', default='')
a = p.parse_args(); sys.path.insert(0, a.generic)
from generic_codec_v1 import load_ckpt, xyz_quant  # noqa: E402
m = load_ckpt(a.ckpt); xyz, faces = m['xyz'], m['faces']; sh = np.concatenate([m['dc'], m['rest']], 1); V, F = len(xyz), len(faces)
xq, xlo, xstep = xyz_quant(xyz, a.qp); lo = sh.min(0); hi = sh.max(0); L = 2 ** a.bits - 1
sq = np.rint((sh - lo) * np.where(hi > lo, L / np.maximum(hi - lo, 1e-12), 0.0)).astype(np.uint8)
r = dict(label=a.label, vertices=V, faces=F, bits=a.bits, qp=a.qp, cpu=platform.processor() or platform.machine(), host=platform.node(), threads=os.cpu_count())
try:
    import zstandard as zstd
    blob = xq.tobytes() + sq.tobytes() + faces.astype(np.int32).tobytes()
    t = time.time(); c = zstd.ZstdCompressor(level=19, threads=-1).compress(blob); r['zstd_encode_s'] = round(time.time() - t, 2)
    t = time.time(); d = zstd.ZstdDecompressor().decompress(c); r['zstd_decode_s'] = round(time.time() - t, 2); assert d == blob; r['zstd_MB'] = round(len(c) / 1e6, 3); del blob, c, d
except Exception as e: r['zstd_error'] = repr(e)
try:
    import DracoPy
    t = time.time(); buf = DracoPy.encode(xyz.astype(np.float32), faces.astype(np.uint32), quantization_bits=a.qp, compression_level=10, preserve_order=False, generic_attributes={'sh': sq}); r['draco_encode_s'] = round(time.time() - t, 2)
    t = time.time(); dm = DracoPy.decode(buf); r['draco_decode_s'] = round(time.time() - t, 2); r['draco_MB'] = round(len(buf) / 1e6, 3); assert len(dm.faces) == F; del buf, dm
except Exception as e: r['draco_error'] = repr(e)
if a.driver:
    try:
        with tempfile.TemporaryDirectory(dir=str(Path(a.out).resolve().parent)) as td:
            d = Path(td); bundle, stream, dec = d / 'in.bundle', d / 'mesh.o3dgc', d / 'out.decoded'
            with open(bundle, 'wb') as f:
                f.write(b'O3DGCB01' + struct.pack('<IIII', V, F, 1, sq.shape[1])); f.write(np.ascontiguousarray(xyz, np.float32).tobytes()); f.write(np.ascontiguousarray(faces, np.int32).tobytes()); f.write(np.ascontiguousarray(sq, np.uint8).tobytes())
            t = time.time(); pr = subprocess.run([a.driver, str(bundle), str(stream), str(dec), str(a.qp), '8'], capture_output=True, text=True); dt = time.time() - t
            if pr.returncode == 0 and stream.exists(): r['tfan_encode_plus_decode_s'] = round(dt, 2); r['tfan_MB'] = round(stream.stat().st_size / 1e6, 3); r['tfan_driver_output'] = pr.stdout.strip()[-400:]
            else: r['tfan_error'] = pr.stderr.strip()[-300:]
    except Exception as e: r['tfan_error'] = repr(e)
Path(a.out).write_text(json.dumps(r, indent=1))
g = lambda k: r.get(k, 'n/a')
print(f"TIME {a.label} | V {V} F {F} | sh{a.bits} | zstd encode {g('zstd_encode_s')} s decode {g('zstd_decode_s')} s | draco encode {g('draco_encode_s')} s decode {g('draco_decode_s')} s | tfan encode+decode {g('tfan_encode_plus_decode_s')} s", flush=True)
PYGEN39
cat > $AN/codec-time-v1.sh <<'SH39'
#!/usr/bin/env bash
# codec-time-v1.sh (2026-11-14): encode / decode time of our codec and of zstd, Draco, MPEG TFAN on the SAME mesh per scene: the MeshSplatting
# mesh trained with a vertex cap near 1.08 million (fine-tuned), 11 Mip-NeRF 360 / T&T scenes.  CPU only, 8 threads, about 20 minutes.
# Ours reads <cap run>/axisB/v1.0/deliver/shared-final.pt (what the encoder receives); the generic codecs read <cap run>/axisB/v1.0/finetune/best.pt.
# Rows -> $AN/codec-time.txt.   Run:  sbatch ... codec-time-v1.sh    or directly in an interactive CPU session:  bash codec-time-v1.sh
[ -f /scratch/zr2245/anaconda3/etc/profile.d/conda.sh ] && { source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; }
[ -f /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate ] && source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=8
AN=${AN:-/scratch/zr2245/analysis}; ROOT=${ROOT:-/scratch/zr2245}; PKG=${PKG:-$AN/vis_collapse-v12}; G=$AN/generic-v1; DRV=$AN/o3dgc-v1/o3dgc_driver; OUT=$AN/codec-time; TAB=$AN/codec-time.txt
SCENES=${SCENES:-"garden room truck kitchen counter bonsai bicycle stump flowers treehill train"}; mkdir -p $OUT
mid () { case $1 in room) echo 1078000;; kitchen) echo 1086000;; counter) echo 1083000;; bonsai|garden) echo 1073000;; bicycle) echo 1080000;; stump) echo 1069000;; flowers) echo 1071000;; treehill) echo 1079000;; truck) echo 1078000;; train) echo 1085000;; esac; }
qp () { case $1 in room) echo 14;; kitchen|bicycle|stump|treehill) echo 16;; counter|bonsai|garden|flowers) echo 15;; truck|train) echo 18;; esac; }
for f in $AN/codec_time_v1.py $AN/generic_time_v1.py $PKG/stage2_run.sh $G/generic_codec_v1.py; do [ -e $f ] || { echo "FAILED: missing $f"; exit 1; }; done
python -c "import DracoPy, zstandard" 2>/dev/null || pip install DracoPy zstandard > $OUT/pip.log 2>&1 || echo "note: DracoPy / zstandard not installable, those columns stay empty"
DOPT=""; [ -x $DRV ] && DOPT="--driver $DRV" || echo "note: no Open3DGC driver at $DRV, the TFAN column stays empty"
echo "codec-time-v1 | host $(hostname) | $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs) | threads $OMP_NUM_THREADS | $(date)"
nf=0
for S in $SCENES; do B=$(mid $S); Q=$(qp $S); CG=$ROOT/$S-budget-$B/axisB/v1.0
  if [ -f $CG/deliver/shared-final.pt ]; then
    bash $PKG/stage2_run.sh $AN/codec_time_v1.py --shared $CG/deliver/shared-final.pt --xyz q$Q --dc q7 --out $OUT/$S-ours.json --label "$S cap$B ours" > $OUT/$S-ours.log 2>&1 && grep -h "^TIME" $OUT/$S-ours.log | tee -a $TAB || { echo "FAILED: ours $S"; tail -6 $OUT/$S-ours.log; nf=$((nf+1)); }
  else echo "note: $S: no $CG/deliver/shared-final.pt, skipped"; nf=$((nf+1)); fi
  if [ -f $CG/finetune/best.pt ]; then
    python $AN/generic_time_v1.py --ckpt $CG/finetune/best.pt --qp $Q --bits 8 --out $OUT/$S-generic.json --generic $G $DOPT --label "$S cap$B generic" > $OUT/$S-generic.log 2>&1 && grep -h "^TIME" $OUT/$S-generic.log | tee -a $TAB || { echo "FAILED: generic $S"; tail -6 $OUT/$S-generic.log; nf=$((nf+1)); }
  else echo "note: $S: no $CG/finetune/best.pt, skipped"; nf=$((nf+1)); fi
done
echo "codec-time-v1 exit 0 ($nf failures)  $(date)"
SH39
python3 -m py_compile $AN/codec_time_v1.py $AN/generic_time_v1.py && bash -n $AN/codec-time-v1.sh && echo "FILES OK: $(md5sum < $AN/codec_time_v1.py | cut -c1-8) $(md5sum < $AN/generic_time_v1.py | cut -c1-8) $(md5sum < $AN/codec-time-v1.sh | cut -c1-8)"
```

## Run

```bash
bash /scratch/zr2245/analysis/codec-time-v1.sh 2>&1 | tee /scratch/zr2245/analysis/codec-time-direct.out
```

## Collect

```bash
AN=/scratch/zr2245/analysis; cat $AN/codec-time.txt; echo; tail -3 $AN/codec-time-*.out 2>/dev/null
```
