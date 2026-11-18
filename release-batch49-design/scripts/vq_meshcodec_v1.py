"""vq_meshcodec_v1.py (2026-11-18): the same vector quantization as BM-Codec, followed by an existing mesh codec instead of our lossless coding.
The quantised mesh is exactly the one BM-Codec writes (positions on the q<b> grid, DC at 7 bits, the two SH indices per vertex), so only the
lossless part differs.  CPU only.
  Draco : DracoPy, compression level 10; positions given as the integer grid (Draco's own quantiser at the same bit depth is then lossless on
          the longest axis); DC (uint8, 3) and the two indices as integer generic attributes (lossless).
  TFAN  : Open3DGC driver of the baselines (MPEG SC3DMC/TFAN, parallelogram prediction for positions); DC and the indices as one 6-dim
          INTEGER attribute [r, g, b, main >> 8, main & 255, residual] with differential prediction (lossless).
Both add the two codebooks compressed with zstd, as in BM-Codec.  Also printed: BM-Codec and VQ + zstd (vq_zstd_v1.py) on the same arrays.
    bash $PKG/stage2_run.sh $AN/vq_meshcodec_v1.py --shared <deliver>/shared-final.pt --xyz q15 --dc q7 --driver $AN/o3dgc-v1/o3dgc_driver --out <json> --label garden
    python vq_meshcodec_v1.py --selftest [--driver <o3dgc_driver>]"""
import argparse, json, shutil, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vq_zstd_v1 import zs, small, streams, take          # noqa: E402

def draco_bytes(xyz, faces, qp, dc, im, ir):
    import DracoPy
    V = len(xyz); pts = np.ascontiguousarray(xyz, np.float32); f = np.ascontiguousarray(faces, np.uint32)
    att = {'dc': np.ascontiguousarray(dc.astype(np.uint8).reshape(V, 3)), 'im': np.ascontiguousarray(im.astype(np.uint16).reshape(V, 1)), 'ir': np.ascontiguousarray(ir.astype(np.uint16).reshape(V, 1))}
    try:
        buf = DracoPy.encode(pts, f, quantization_bits=qp, compression_level=10, preserve_order=False, generic_attributes=att); how = 'three attributes'
    except Exception as e:                                   # older DracoPy: one attribute only
        one = np.ascontiguousarray(np.concatenate([dc.reshape(V, 3), im.reshape(V, 1), ir.reshape(V, 1)], 1).astype(np.uint16))
        buf = DracoPy.encode(pts, f, quantization_bits=qp, compression_level=10, preserve_order=False, generic_attributes={'a': one}); how = 'one attribute (%s)' % type(e).__name__
    d = DracoPy.decode(buf)
    return len(buf), how, int(len(d.points)), int(len(d.faces))

def tfan_bytes(driver, xyz, faces, qp, dc, im, ir, work):
    V = len(xyz); a = np.concatenate([dc.reshape(V, 3), (im >> 8).reshape(V, 1), (im & 255).reshape(V, 1), ir.reshape(V, 1)], 1)
    if a.min() < 0 or a.max() > 255: raise AssertionError('TFAN attribute out of uint8 range')
    b, s, o = work / 'in.bundle', work / 'out.o3dgc', work / 'out.decoded'
    with open(b, 'wb') as fh:
        fh.write(b'O3DGCB01' + struct.pack('<IIII', V, len(faces), 1, 6)); fh.write(np.ascontiguousarray(xyz, np.float32).tobytes())
        fh.write(np.ascontiguousarray(faces, np.int32).tobytes()); fh.write(np.ascontiguousarray(a, np.uint8).tobytes())
    r = subprocess.run([str(driver), str(b), str(s), str(o), str(qp), '8'], capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError('TFAN driver failed: ' + r.stderr.strip()[-400:])
    raw = open(o, 'rb').read(); V2, F2, mode, D = struct.unpack('<IIII', raw[8:24]); off = 24 + V2 * 12 + F2 * 12
    dec = np.frombuffer(raw, np.uint8, V2 * D, off).reshape(V2, D)
    ok = V2 == V and np.array_equal(np.unique(dec, axis=0), np.unique(a.astype(np.uint8), axis=0))
    return s.stat().st_size, ok

def selftest(driver):
    rng = np.random.default_rng(0); n = 60; idx = np.arange(n * n).reshape(n, n); V = n * n
    f = np.concatenate([np.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:]], -1).reshape(-1, 3), np.stack([idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 3)])
    y, x = np.mgrid[0:n, 0:n]; xyz = np.stack([x * 50, y * 50, 1000 + 300 * np.sin(x / 7)], -1).reshape(-1, 3).astype(np.float32)
    dc = rng.integers(0, 128, (V, 3)); im = rng.integers(0, 4096, V); ir = rng.integers(0, 256, V)
    try: print('selftest draco:', draco_bytes(xyz, f, 14, dc, im, ir))
    except ImportError: print('selftest draco: DracoPy not installed here')
    if driver:
        w = Path(tempfile.mkdtemp()); print('selftest tfan:', tfan_bytes(driver, xyz, f, 14, dc, im, ir, w)); shutil.rmtree(w)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage2'); p.add_argument('--shared'); p.add_argument('--out'); p.add_argument('--xyz', default='q14'); p.add_argument('--dc', default='q7')
    p.add_argument('--driver'); p.add_argument('--label', default=''); p.add_argument('--selftest', action='store_true'); a = p.parse_args()
    if a.selftest: return selftest(a.driver)
    import torch
    stage2 = Path(a.stage2).resolve()
    for d in (stage2 / 'renderer', stage2, stage2 / 'codec'): sys.path.insert(0, str(d))
    import delivery, topology as tc
    from codec_benchmark import pack
    delivery.XYZ_MODES = tuple(dict.fromkeys(delivery.XYZ_MODES + ('q15', 'q13', 'q12'))); delivery.DC_MODES = tuple(dict.fromkeys(delivery.DC_MODES + ('q7', 'q6', 'q5', 'q4')))
    wire = torch.load(a.shared, map_location='cpu', weights_only=False); wire.setdefault('opacity_floor', .999); qp = int(a.xyz[1:])
    work = Path(tempfile.mkdtemp(prefix='vq-mesh-', dir=str(Path(a.out).resolve().parent))); native = work / 'native'
    try:
        exe = tc.lc.native(native); edge = tc.edge_native(native); ctx = tc.ic.native(native); tmp = work / 'enc'; tmp.mkdir()
        quant = delivery.prepare(wire, a.xyz, a.dc, True, 'const'); ordered, perm = tc.reorder(quant, 'strip-fetch')
        pred = tc.lc.graph(exe, ordered, tmp); encoded = tc.encode(ordered, 'edge-arithmetic', 1, exe, edge, tmp, pred, ctx); path = work / 'asset.mslb'; pack(encoded, path, 'auto'); size = path.stat().st_size
        O = take(ordered); V = len(O['dc']); faces = np.asarray(O['faces']).reshape(-1, 3).astype(np.int64); xyz = np.asarray(O['xyz']).reshape(V, 3).astype(np.int64)
        dc = small(np.asarray(O['dc']).reshape(V, 3)).astype(np.int64); im = small(np.asarray(O['im']).reshape(-1)).astype(np.int64); ir = small(np.asarray(O['ir']).reshape(-1)).astype(np.int64)
        if xyz.min() < 0 or xyz.max() >= (1 << qp): raise AssertionError('positions are not on the q%d grid: range %d..%d' % (qp, xyz.min(), xyz.max()))
        cb = zs(np.asarray(O['bank'], np.float32).reshape(len(O['bank']), -1)) + (zs(np.asarray(O['rbank'], np.float32).reshape(len(O['rbank']), -1)) if O['rbank'] is not None and len(O['rbank']) else 0)
        vz = sum(streams(**O).values())
        try: db, how, dv, df = draco_bytes(xyz, faces, qp, dc, im, ir); dr = db + cb
        except Exception as e: dr, how, dv, df = None, 'FAILED %s: %s' % (type(e).__name__, str(e)[:200]), 0, 0
        tr, tok = None, None
        if a.driver:
            try: tb, tok = tfan_bytes(a.driver, xyz, faces, qp, dc, im, ir, work); tr = tb + cb
            except Exception as e: print('note: %s: TFAN failed (%s: %s)' % (a.label, type(e).__name__, str(e)[:300]))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    MB = 1e6; pc = lambda x: 'n/a' if x is None else '%.3f MB (%+.1f %%)' % (x / MB, 100 * (x - size) / size)
    out = dict(label=a.label, vertices=V, faces=len(faces), bm_codec_bytes=int(size), codebooks_zstd_bytes=int(cb), vq_zstd_bytes=int(vz), vq_draco_bytes=dr, draco_how=how, draco_decoded_vertices=dv, draco_decoded_faces=df, vq_tfan_bytes=tr, tfan_attributes_roundtrip=tok)
    print('VQMESH %s | V %d F %d | BM-Codec %.3f MB | VQ + Draco %s | VQ + TFAN %s | VQ + zstd %s | codebooks %.3f MB in each | draco: %s, decoded V %d F %d | tfan attributes round trip %s' % (
        a.label, V, len(faces), size / MB, pc(dr), pc(tr), pc(vz), cb / MB, how, dv, df, tok), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))

if __name__ == '__main__':
    main()
