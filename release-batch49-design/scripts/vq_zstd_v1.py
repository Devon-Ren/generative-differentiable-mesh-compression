"""vq_zstd_v1.py (2026-11-17): the baseline "vector quantization + zstd".  Same lossy steps as BM-Codec (same quantised mesh, so the decoded mesh
and the PSNR are identical); the lossless part is replaced by zstd on the plain arrays.  CPU only.
  ours     : the BM-Codec file (encoded here with the shipped encoder, as in codec_choices_v2.py)
  zstd-ord : zstd on the arrays in the ORDER OF BM-CODEC (strip order of the faces, vertices numbered by first use)
  zstd-raw : zstd on the arrays in the order of the trained mesh (no reordering at all) -- the "trivial pipeline"
Arrays: faces uint32 [F,3]; positions uint16 or uint32 [V,3]; DC uint8 [V,3]; main index uint16 [V]; residual index uint8/uint16 [V];
the two codebooks float32.  Each array is compressed alone, at zstd level 19, raw or byte-shuffled, whichever is smaller (the same choice the
container of BM-Codec makes for the DC colour and the codebooks at level 9).
    bash $PKG/stage2_run.sh $AN/vq_zstd_v1.py --shared <deliver>/shared-final.pt --xyz q15 --dc q7 --out <json> --label garden
    python vq_zstd_v1.py --selftest"""
import argparse, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np

def small(a):
    a = np.asarray(a).astype(np.int64)
    if a.size and a.min() < 0: a = np.where(a < 0, a.max() + 1, a)        # a 'none' marker becomes one extra symbol
    m = int(a.max()) if a.size else 0
    return a.astype(np.uint8 if m < 256 else (np.uint16 if m < 65536 else np.uint32))

def zs(a, level=19):
    import zstandard
    a = np.ascontiguousarray(a); raw = a.tobytes(); c = zstandard.ZstdCompressor(level=level); best = len(c.compress(raw))
    if a.dtype.itemsize > 1: best = min(best, len(c.compress(a.view(np.uint8).reshape(-1, a.dtype.itemsize).T.copy().tobytes())))
    if a.ndim == 2 and a.shape[1] > 1: best = min(best, len(c.compress(np.ascontiguousarray(a.T).tobytes())))     # one column after the other
    return best

def streams(faces, xyz, dc, im, ir, bank, rbank):
    return dict(connectivity=zs(np.asarray(faces).reshape(-1, 3).astype(np.uint32)), positions=zs(small(xyz)), dc=zs(small(dc)), sh_indices=zs(small(np.asarray(im).reshape(-1))) + zs(small(np.asarray(ir).reshape(-1))),
                codebooks=zs(np.asarray(bank, np.float32).reshape(len(bank), -1)) + (zs(np.asarray(rbank, np.float32).reshape(len(rbank), -1)) if rbank is not None and len(rbank) else 0))

def selftest():
    rng = np.random.default_rng(0); n = 80; idx = np.arange(n * n).reshape(n, n)
    f = np.concatenate([np.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:]], -1).reshape(-1, 3), np.stack([idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 3)]); V = n * n
    s = streams(f, rng.integers(0, 1 << 15, (V, 3)), rng.integers(0, 128, (V, 3)), rng.integers(0, 4096, V), rng.integers(0, 256, V), rng.normal(size=(4096, 45)), rng.normal(size=(256, 45)))
    assert all(v > 0 for v in s.values()) and s['dc'] <= V * 3 and s['positions'] <= V * 6 + 64, s
    print('selftest ok:', s)

def arr(d, names):
    for k in names:
        if k in d:
            v = d[k]; return (v.numpy() if hasattr(v, 'numpy') else np.asarray(v)), k
    return None, None

def take(d):
    V = arr(d, ['features_dc'])[0].shape[0]
    return dict(faces=arr(d, ['_triangle_indices'])[0], xyz=arr(d, ['triangles_points'])[0].reshape(V, -1), dc=arr(d, ['features_dc'])[0].reshape(V, -1), im=arr(d, ['app_mapping', 'mapping'])[0], ir=arr(d, ['app_residual_mapping', 'residual_mapping'])[0],
                bank=arr(d, ['app_bank'])[0], rbank=arr(d, ['app_residual_bank'])[0])

def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage2'); p.add_argument('--shared'); p.add_argument('--out'); p.add_argument('--xyz', default='q14'); p.add_argument('--dc', default='q7'); p.add_argument('--label', default=''); p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    if a.selftest: return selftest()
    import torch
    stage2 = Path(a.stage2).resolve()
    for d in (stage2 / 'renderer', stage2, stage2 / 'codec'): sys.path.insert(0, str(d))
    import delivery, topology as tc
    from codec_benchmark import pack
    delivery.XYZ_MODES = tuple(dict.fromkeys(delivery.XYZ_MODES + ('q15', 'q13', 'q12'))); delivery.DC_MODES = tuple(dict.fromkeys(delivery.DC_MODES + ('q7', 'q6', 'q5', 'q4')))
    wire = torch.load(a.shared, map_location='cpu', weights_only=False); wire.setdefault('opacity_floor', .999)
    work = Path(tempfile.mkdtemp(prefix='vq-zstd-', dir=str(Path(a.out).resolve().parent))); native = work / 'native'
    try:
        exe = tc.lc.native(native); edge = tc.edge_native(native); ctx = tc.ic.native(native); tmp = work / 'enc'; tmp.mkdir()
        quant = delivery.prepare(wire, a.xyz, a.dc, True, 'const'); ordered, perm = tc.reorder(quant, 'strip-fetch')
        pred = tc.lc.graph(exe, ordered, tmp); encoded = tc.encode(ordered, 'edge-arithmetic', 1, exe, edge, tmp, pred, ctx); path = work / 'asset.mslb'; pack(encoded, path, 'auto'); size = path.stat().st_size
        nb = lambda v: int(np.asarray(v.numpy() if hasattr(v, 'numpy') else v).nbytes)
        ship = {k: nb(v) for k, v in encoded.items() if k in ('top_stream', 'local_residual', 'ctx_mapping', 'ctx_residual')}
        O = take(ordered)
        try: Q = take(quant)
        except Exception as e: Q = None; print('note: %s: the unordered mesh could not be read (%s: %s)' % (a.label, type(e).__name__, e))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    V, F = len(O['dc']), len(np.asarray(O['faces']).reshape(-1, 3)); so = streams(**O); to = sum(so.values()); MB = 1e6
    if Q is not None and any(Q[k] is None for k in ('faces', 'xyz', 'dc', 'im', 'ir', 'bank')): Q = None; print('note: %s: the unordered mesh lacks some arrays' % a.label)
    sq = streams(**Q) if Q is not None else None; tq = sum(sq.values()) if sq else None
    ours = dict(connectivity=ship.get('top_stream'), positions=ship.get('local_residual'), sh_indices=(ship.get('ctx_mapping') or 0) + (ship.get('ctx_residual') or 0) or None)
    rest = size - sum(v for v in ours.values() if v)      # DC colour, codebooks and header of the BM-Codec file
    out = dict(label=a.label, vertices=V, faces=F, file_bytes=int(size), ours_streams=ours, ours_dc_codebooks_header=int(rest), zstd_ordered=so, zstd_ordered_total=int(to), zstd_unordered=sq, zstd_unordered_total=tq)
    f = lambda x: 'n/a' if x is None else '%.3f' % (x / MB)
    print('VQZSTD %s | V %d F %d | BM-Codec %.3f MB | VQ + zstd, our order %.3f MB (%+.1f %%) | VQ + zstd, training order %s MB (%s) | connectivity ours %s / zstd %s / zstd-unordered %s | positions %s / %s / %s | SH indices %s / %s / %s | DC + codebooks + header ours %s, zstd DC %s codebooks %s' % (
        a.label, V, F, size / MB, to / MB, 100 * (to - size) / size, f(tq), ('%+.1f %%' % (100 * (tq - size) / size)) if tq else 'n/a', f(ours['connectivity']), f(so['connectivity']), f(sq['connectivity'] if sq else None),
        f(ours['positions']), f(so['positions']), f(sq['positions'] if sq else None), f(ours['sh_indices']), f(so['sh_indices']), f(sq['sh_indices'] if sq else None), f(rest), f(so['dc']), f(so['codebooks'])), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))

if __name__ == '__main__':
    main()
