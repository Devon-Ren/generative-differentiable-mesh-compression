"""dc_predict_v1.py (2026-11-15): how many bytes does neighbour prediction save on the DC colour stream?  CPU only.
The shipped codec stores the quantised DC colour (q7) with zstd, no prediction.  This script takes the SAME quantised, reordered mesh the
encoder writes and codes the DC colour a second way, with a real, decodable bitstream:
    prediction : from vertices decoded earlier (vertex ids are in first-reference order, so the decoder has them)
                 'first' = the earlier neighbour with the smallest id (the predictor the codec uses for positions)
                 'mean'  = rounded mean of all earlier neighbours
                 a vertex without earlier neighbour is predicted by vertex id-1; vertex 0 by 0
    residual   : zigzag(q - prediction), coded MSB-first down a binary tree with an adaptive binary range coder
    context    : (channel, bucket of the previous residual: channel 0 -> channel 0 of the previous vertex, channels 1,2 -> the
                 previous channel of the same vertex, buckets {<=1, <=4, <=16, >16}, tree prefix)
Every stream is decoded again and compared with the input (round trip must be exact) before its size is reported.
Reported: the shipped file, the zstd bytes of the DC stream (raw / byte-shuffled, level 9, the smaller, as the container does), the
predicted bytes, and the file size with the DC stream replaced.  Nothing in the shipped codec is changed.
    bash $PKG/stage2_run.sh $AN/dc_predict_v1.py --shared <deliver>/shared-final.pt --xyz q15 --dc q7 --out <json> --label "garden cap1073000"
    python dc_predict_v1.py --selftest        (no stage2 needed: round trip on a synthetic mesh)"""
import argparse, json, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np

TOP, BITS, MOVE = 1 << 24, 11, 5
class Enc:
    def __init__(s): s.low = 0; s.rng = 0xFFFFFFFF; s.cache = 0; s.csize = 1; s.out = bytearray()
    def shift(s):
        if s.low < 0xFF000000 or s.low >= (1 << 32):
            c = s.low >> 32; t = s.cache
            while True:
                s.out.append((t + c) & 0xFF); t = 0xFF; s.csize -= 1
                if s.csize == 0: break
            s.cache = (s.low >> 24) & 0xFF
        s.csize += 1; s.low = (s.low & 0x00FFFFFF) << 8
    def finish(s):
        for _ in range(5): s.shift()
        return bytes(s.out)

def encode(sym, ctxb, nb):
    """sym (V,3) zigzag residuals < 2**nb.  Returns the bitstream."""
    e = Enc(); P = [1024] * (3 * 4 * (1 << nb)); V = len(sym); S = sym.tolist(); low, rng = e.low, e.rng; shift = e.shift; N = 1 << nb
    prev0 = 0
    for i in range(V):
        row = S[i]; pb = prev0
        for ch in range(3):
            v = row[ch]; b = 0 if pb <= 1 else (1 if pb <= 4 else (2 if pb <= 16 else 3)); base = (ch * 4 + b) * N; node = 1
            for k in range(nb - 1, -1, -1):
                bit = (v >> k) & 1; j = base + node; p = P[j]; bound = (e.rng >> BITS) * p
                if bit == 0: e.rng = bound; P[j] = p + ((2048 - p) >> MOVE)
                else: e.low += bound; e.rng -= bound; P[j] = p - (p >> MOVE)
                while e.rng < TOP: e.rng = (e.rng << 8) & 0xFFFFFFFF; shift()
                node = (node << 1) | bit
            pb = v
        prev0 = row[0]
    return e.finish()

def decode(buf, V, nb):
    P = [1024] * (3 * 4 * (1 << nb)); N = 1 << nb; out = np.zeros((V, 3), np.int64); pos = 5; code = int.from_bytes(buf[1:5], 'big'); rng = 0xFFFFFFFF; prev0 = 0; n = len(buf)
    for i in range(V):
        pb = prev0; row = [0, 0, 0]
        for ch in range(3):
            b = 0 if pb <= 1 else (1 if pb <= 4 else (2 if pb <= 16 else 3)); base = (ch * 4 + b) * N; node = 1
            for k in range(nb):
                j = base + node; p = P[j]; bound = (rng >> BITS) * p
                if code < bound: rng = bound; P[j] = p + ((2048 - p) >> MOVE); bit = 0
                else: code -= bound; rng -= bound; P[j] = p - (p >> MOVE); bit = 1
                while rng < TOP:
                    rng = (rng << 8) & 0xFFFFFFFF; code = ((code << 8) | (buf[pos] if pos < n else 0)) & 0xFFFFFFFF; pos += 1
                node = (node << 1) | bit
            v = node - N; row[ch] = v; pb = v
        out[i] = row; prev0 = row[0]
    return out

def earlier_neighbours(faces, V):
    f = np.asarray(faces, np.int64); e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]); e = np.concatenate([e, e[:, ::-1]]); e = e[e[:, 0] < e[:, 1]]
    e = np.unique(e, axis=0); return e[:, 0], e[:, 1]      # u < i: u is an earlier neighbour of i

def predict(q, faces, mode):
    """q (V,3) int.  Causal prediction; the decoder can rebuild it vertex by vertex (see decode_stream)."""
    V = len(q); u, i = earlier_neighbours(faces, V); pred = np.zeros_like(q); has = np.zeros(V, bool); has[i] = True
    if mode == 'first':
        p0 = np.full(V, V, np.int64); np.minimum.at(p0, i, u); pred[has] = q[p0[has]]
    else:
        s = np.zeros((V, 3), np.int64); c = np.zeros(V, np.int64); np.add.at(s, i, q[u]); np.add.at(c, i, 1); pred[has] = (2 * s[has] + c[has, None]) // (2 * c[has, None])
    lone = np.where(~has)[0]; lone = lone[lone > 0]; pred[lone] = q[lone - 1]
    return pred

zig = lambda d: np.where(d >= 0, 2 * d, -2 * d - 1)
unzig = lambda z: np.where(z % 2 == 0, z // 2, -(z + 1) // 2)

def decode_stream(buf, faces, V, nb, mode):
    """Decoder: residuals from the bitstream, then the colours vertex by vertex from already-decoded vertices only."""
    r = unzig(decode(buf, V, nb)); u, i = earlier_neighbours(faces, V); order = np.argsort(i, kind='stable'); u, i = u[order], i[order]; start = np.searchsorted(i, np.arange(V + 1))
    q = np.zeros((V, 3), np.int64)
    for v in range(V):
        a, b = start[v], start[v + 1]
        if b > a:
            nb_ = u[a:b]
            if mode == 'first': p = q[nb_.min()]
            else: s = q[nb_].sum(0); c = b - a; p = (2 * s + c) // (2 * c)
        else: p = q[v - 1] if v > 0 else 0
        q[v] = r[v] + p
    return q

def measure(q, faces, bits):
    V = len(q); nb = bits + 1; res = {}
    for mode in ('first', 'mean'):
        t = time.time(); sym = zig(q - predict(q, faces, mode)); assert sym.max() < (1 << nb), 'residual out of range'
        buf = encode(sym, None, nb); te = time.time() - t; t = time.time(); back = decode_stream(buf, faces, V, nb, mode); td = time.time() - t
        if not np.array_equal(back, q): raise AssertionError('round trip failed (%s)' % mode)
        res[mode] = dict(bytes=len(buf), bits_per_vertex=round(8 * len(buf) / V, 3), encode_s=round(te, 1), decode_s=round(td, 1))
    return res

def zstd_bytes(q):
    import zstandard; a = np.ascontiguousarray(q.astype(np.uint16)); raw = a.tobytes(); sh = a.view(np.uint8).reshape(-1, 2).T.copy().tobytes(); c = zstandard.ZstdCompressor(level=9)
    return min(len(c.compress(raw)), len(c.compress(sh)))

def selftest():
    rng = np.random.default_rng(0); n = 60; idx = np.arange(n * n).reshape(n, n); f = np.concatenate([np.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:]], -1).reshape(-1, 3), np.stack([idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 3)])
    y, x = np.mgrid[0:n, 0:n]; q = np.stack([60 + 30 * np.sin(x / 9) + rng.integers(-2, 3, (n, n)), 64 + 25 * np.cos(y / 7) + rng.integers(-2, 3, (n, n)), 50 + x / 2 + rng.integers(-1, 2, (n, n))], -1).reshape(-1, 3).astype(np.int64).clip(0, 127)
    q = np.concatenate([q, rng.integers(0, 128, (5, 3))])      # 5 vertices used by no face
    r = measure(q, f, 7); print('selftest: V %d | first %d B | mean %d B | raw %d B | round trips exact' % (len(q), r['first']['bytes'], r['mean']['bytes'], len(q) * 3 * 7 // 8))

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
    wire = torch.load(a.shared, map_location='cpu', weights_only=False); wire.setdefault('opacity_floor', .999); bits = int(a.dc[1:])
    work = Path(tempfile.mkdtemp(prefix='dc-predict-', dir=str(Path(a.out).resolve().parent))); native = work / 'native'
    try:
        exe = tc.lc.native(native); edge = tc.edge_native(native); ctx = tc.ic.native(native); tmp = work / 'enc'; tmp.mkdir()
        quant = delivery.prepare(wire, a.xyz, a.dc, True, 'const'); ordered, perm = tc.reorder(quant, 'strip-fetch')
        pred = tc.lc.graph(exe, ordered, tmp); encoded = tc.encode(ordered, 'edge-arithmetic', 1, exe, edge, tmp, pred, ctx); path = work / 'asset.mslb'; pack(encoded, path, 'auto'); size = path.stat().st_size
        dc = ordered['features_dc']; dc = dc.numpy() if hasattr(dc, 'numpy') else np.asarray(dc); V = dc.shape[0]; q = dc.reshape(V, -1).astype(np.int64)
        if q.shape[1] != 3 or q.min() < 0 or q.max() >= (1 << bits): raise AssertionError('unexpected DC array: shape %s range %d..%d for %s' % (dc.shape, q.min(), q.max(), a.dc))
        faces = ordered['_triangle_indices']; faces = faces.numpy() if hasattr(faces, 'numpy') else np.asarray(faces)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    z = zstd_bytes(q); r = measure(q, faces, bits); best = min(r, key=lambda k: r[k]['bytes']); nb = r[best]['bytes']
    out = dict(label=a.label, vertices=int(V), faces=int(len(faces)), file_bytes=int(size), dc_zstd_bytes=int(z), dc_zstd_bits_per_vertex=round(8 * z / V, 3), predicted=r, best=best,
               file_bytes_with_predicted_dc=int(size - z + nb), file_change_percent=round(100 * (nb - z) / size, 2), dc_change_percent=round(100 * (nb - z) / z, 2), roundtrip='exact')
    Path(a.out).write_text(json.dumps(out, indent=1))
    print('DCPRED %s | V %d | file %.3f MB | DC zstd %.3f MB (%.2f b/v, %.1f %% of file) | first %.3f MB (%.2f b/v) | mean %.3f MB (%.2f b/v) | best %s: DC %+.1f %%, file %+.2f %% | roundtrip exact' % (
        a.label, V, size / 1e6, z / 1e6, 8 * z / V, 100 * z / size, r['first']['bytes'] / 1e6, r['first']['bits_per_vertex'], r['mean']['bytes'] / 1e6, r['mean']['bits_per_vertex'], best, out['dc_change_percent'], out['file_change_percent']), flush=True)

if __name__ == '__main__':
    main()
