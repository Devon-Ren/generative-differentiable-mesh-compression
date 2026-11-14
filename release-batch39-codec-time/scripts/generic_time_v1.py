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
