"""codec_choices_v2.py (2026-11-15): three measurements on the SAME quantised, reordered mesh the encoder writes.  CPU only.
  DC   : DC colour, shipped = zstd without prediction; tested = prediction from earlier-decoded neighbours (first / mean) + adaptive coder
  XYZ  : positions, four predictors with ONE coder so that only the predictor differs:
           prev  = previous vertex in the order            first = earlier neighbour with the smallest id (the shipped predictor)
           mean  = rounded mean of all earlier neighbours  para  = parallelogram b + c - d where a face (v,b,c) and a face (b,c,d) exist with
                                                                    b, c, d earlier than v; otherwise 'first'
         (a vertex without earlier neighbour falls back to the previous vertex; vertex 0 to 0).  The shipped stream size is printed next to them.
  IDX  : the two SH index streams, shipped (context = indices of earlier-decoded neighbours) against the same indices coded with an
         adaptive model WITHOUT neighbour context, and against fixed-length codes.
Every tested stream is a real bitstream from an adaptive binary range coder.  DC and IDX streams are all decoded and compared with the input;
for XYZ the smallest stream is decoded and compared (the others are encoded only, to keep the run short).  The shipped codec is not changed.
    bash $PKG/stage2_run.sh $AN/codec_choices_v2.py --shared <deliver>/shared-final.pt --xyz q15 --dc q7 --out <json> --label "garden cap1073000"
    python codec_choices_v2.py --selftest"""
import argparse, json, math, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np

TOP, BITS, MOVE = 1 << 24, 11, 5
def bucket_fn(th):
    if th is None: return lambda r: 0
    a, b, c = th; return lambda r: 0 if r <= a else (1 if r <= b else (2 if r <= c else 3))

def encode(sym, nb, mode, th):
    """sym (V,C) non-negative ints < 2**nb.  mode 'tree': context (channel, bucket, prefix);  'pos': context (channel, bucket, bit position).
    bucket: of the previous residual (channel 0: channel 0 of the previous vertex; other channels: the previous channel of this vertex)."""
    V, C = sym.shape; N = (1 << nb) if mode == 'tree' else nb + 1; P = [1024] * (C * 4 * N); S = sym.tolist(); bk = bucket_fn(th); tree = mode == 'tree'
    low = 0; rng = 0xFFFFFFFF; cache = 0; csize = 1; out = bytearray(); prev0 = 0
    for i in range(V):
        row = S[i]; pb = prev0
        for ch in range(C):
            v = row[ch]; base = (ch * 4 + bk(pb)) * N; node = 1
            for k in range(nb - 1, -1, -1):
                bit = (v >> k) & 1; j = base + (node if tree else k); p = P[j]; bound = (rng >> BITS) * p
                if bit == 0: rng = bound; P[j] = p + ((2048 - p) >> MOVE)
                else: low += bound; rng -= bound; P[j] = p - (p >> MOVE)
                while rng < TOP:
                    rng = (rng << 8) & 0xFFFFFFFF
                    if low < 0xFF000000 or low >= 0x100000000:
                        c = low >> 32; t = cache
                        while True:
                            out.append((t + c) & 0xFF); t = 0xFF; csize -= 1
                            if csize == 0: break
                        cache = (low >> 24) & 0xFF
                    csize += 1; low = (low & 0x00FFFFFF) << 8
                node = (node << 1) | bit
            pb = v
        prev0 = row[0]
    for _ in range(5):
        if low < 0xFF000000 or low >= 0x100000000:
            c = low >> 32; t = cache
            while True:
                out.append((t + c) & 0xFF); t = 0xFF; csize -= 1
                if csize == 0: break
            cache = (low >> 24) & 0xFF
        csize += 1; low = (low & 0x00FFFFFF) << 8
    return bytes(out)

def decode(buf, V, C, nb, mode, th):
    N = (1 << nb) if mode == 'tree' else nb + 1; P = [1024] * (C * 4 * N); bk = bucket_fn(th); tree = mode == 'tree'; out = np.zeros((V, C), np.int64)
    pos = 5; code = int.from_bytes(buf[1:5], 'big'); rng = 0xFFFFFFFF; prev0 = 0; n = len(buf)
    for i in range(V):
        pb = prev0; row = [0] * C
        for ch in range(C):
            base = (ch * 4 + bk(pb)) * N; node = 1
            for k in range(nb - 1, -1, -1):
                j = base + (node if tree else k); p = P[j]; bound = (rng >> BITS) * p
                if code < bound: rng = bound; P[j] = p + ((2048 - p) >> MOVE); bit = 0
                else: code -= bound; rng -= bound; P[j] = p - (p >> MOVE); bit = 1
                while rng < TOP:
                    rng = (rng << 8) & 0xFFFFFFFF; code = ((code << 8) | (buf[pos] if pos < n else 0)) & 0xFFFFFFFF; pos += 1
                node = (node << 1) | bit
            v = node - (1 << nb); row[ch] = v; pb = v
        out[i] = row; prev0 = row[0]
    return out

zig = lambda d: np.where(d >= 0, 2 * d, -2 * d - 1)
unzig = lambda z: np.where(z % 2 == 0, z // 2, -(z + 1) // 2)

class Graph:
    """Everything a predictor needs, built from the face list only (the decoder has the faces before any attribute)."""
    def __init__(s, faces, V):
        f = np.asarray(faces, np.int64); s.V = V
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]); e = np.concatenate([e, e[:, ::-1]]); e = np.unique(e[e[:, 0] < e[:, 1]], axis=0)
        s.u, s.i = e[:, 0], e[:, 1]; s.has = np.zeros(V, bool); s.has[s.i] = True          # u is an earlier neighbour of i (sorted by i? no: by u) -> CSR by i below
        o = np.argsort(s.i, kind='stable'); s.cu, ci = s.u[o], s.i[o]; s.start = np.searchsorted(ci, np.arange(V + 1))
        s.p0 = np.full(V, -1, np.int64); m = np.full(V, V, np.int64); np.minimum.at(m, s.i, s.u); s.p0[s.has] = m[s.has]
        # parallelogram: corner v of face (v,b,c) with b,c < v, and d = the smallest opposite vertex of edge (b,c) over all faces, valid if d < v
        v = np.concatenate([f[:, 0], f[:, 1], f[:, 2]]); b = np.concatenate([f[:, 1], f[:, 2], f[:, 0]]); c = np.concatenate([f[:, 2], f[:, 0], f[:, 1]])
        key = np.minimum(b, c) * V + np.maximum(b, c); uk, inv = np.unique(key, return_inverse=True); mo = np.full(len(uk), V, np.int64); np.minimum.at(mo, inv, v); d = mo[inv]
        ok = (b < v) & (c < v) & (d < v); idx = np.where(ok)[0]; s.pb = np.full(V, -1, np.int64); s.pc = s.pb.copy(); s.pd = s.pb.copy()
        first = np.full(V, len(v), np.int64); np.minimum.at(first, v[idx], idx); hp = first < len(v); s.pb[hp] = b[first[hp]]; s.pc[hp] = c[first[hp]]; s.pd[hp] = d[first[hp]]
    def predict(s, q, mode):
        V = s.V; pred = np.zeros_like(q); lone = np.where(~s.has)[0]; lone = lone[lone > 0]
        if mode == 'prev': pred[1:] = q[:-1]; return pred
        if mode == 'mean':
            a = np.zeros(q.shape, np.int64); c = np.zeros(V, np.int64); np.add.at(a, s.i, q[s.u]); np.add.at(c, s.i, 1); pred[s.has] = (2 * a[s.has] + c[s.has, None]) // (2 * c[s.has, None])
        else:
            pred[s.has] = q[s.p0[s.has]]
            if mode == 'para': hp = s.pb >= 0; pred[hp] = q[s.pb[hp]] + q[s.pc[hp]] - q[s.pd[hp]]
        pred[lone] = q[lone - 1]; return pred
    def reconstruct(s, r, mode):
        """Decoder side: vertex by vertex, from already-reconstructed vertices only."""
        V = s.V; q = np.zeros_like(r); cu, st, p0, pb, pc, pd = s.cu, s.start, s.p0, s.pb, s.pc, s.pd
        for v in range(V):
            if mode == 'prev': p = q[v - 1] if v > 0 else 0
            elif st[v + 1] > st[v]:
                if mode == 'mean': nb_ = cu[st[v]:st[v + 1]]; c = len(nb_); p = (2 * q[nb_].sum(0) + c) // (2 * c)
                elif mode == 'para' and pb[v] >= 0: p = q[pb[v]] + q[pc[v]] - q[pd[v]]
                else: p = q[p0[v]]
            else: p = q[v - 1] if v > 0 else 0
            q[v] = r[v] + p
        return q

def stream(g, q, mode, nb, cmode, th, verify=True):
    sym = zig(q - g.predict(q, mode)); assert sym.max() < (1 << nb), 'residual out of range (%d bits, max %d)' % (nb, sym.max())
    t = time.time(); buf = encode(sym, nb, cmode, th); te = time.time() - t
    if verify and not np.array_equal(g.reconstruct(unzig(decode(buf, len(q), q.shape[1], nb, cmode, th)), mode), q): raise AssertionError('round trip failed (%s)' % mode)
    return dict(bytes=len(buf), bits_per_vertex=round(8 * len(buf) / len(q), 3), encode_s=round(te, 1), verified=bool(verify))

DC_TH, XYZ_TH = (1, 4, 16), (1, 15, 255)
def measure_dc(g, q, bits): return {m: stream(g, q, m, bits + 1, 'tree', DC_TH) for m in ('first', 'mean')}
def measure_xyz(g, q, bits):
    r = {m: stream(g, q, m, bits + 2, 'pos', XYZ_TH, verify=False) for m in ('prev', 'first', 'mean', 'para')}; best = min(r, key=lambda k: r[k]['bytes'])
    r[best] = stream(g, q, best, bits + 2, 'pos', XYZ_TH, verify=True); return r, best
def measure_idx(idx, K):
    nb = max(1, math.ceil(math.log2(max(2, K)))); a = np.asarray(idx, np.int64).reshape(-1, 1); buf = encode(a, nb, 'tree', None)
    if not np.array_equal(decode(buf, len(a), 1, nb, 'tree', None), a): raise AssertionError('index round trip failed')
    return dict(no_context_bytes=len(buf), fixed_length_bytes=int(math.ceil(len(a) * nb / 8)), bits=nb)

def zstd_bytes(q):
    import zstandard; a = np.ascontiguousarray(q.astype(np.uint16)); raw = a.tobytes(); sh = a.view(np.uint8).reshape(-1, 2).T.copy().tobytes(); c = zstandard.ZstdCompressor(level=9)
    return min(len(c.compress(raw)), len(c.compress(sh)))

def selftest():
    rng = np.random.default_rng(0); n = 60; idx = np.arange(n * n).reshape(n, n); f = np.concatenate([np.stack([idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:]], -1).reshape(-1, 3), np.stack([idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 3)])
    y, x = np.mgrid[0:n, 0:n]; q = np.stack([60 + 30 * np.sin(x / 9) + rng.integers(-2, 3, (n, n)), 64 + 25 * np.cos(y / 7) + rng.integers(-2, 3, (n, n)), 50 + x / 2 + rng.integers(-1, 2, (n, n))], -1).reshape(-1, 3).astype(np.int64).clip(0, 127)
    q = np.concatenate([q, rng.integers(0, 128, (5, 3))]); V = len(q); g = Graph(f, V); r = measure_dc(g, q, 7)
    xyz = np.stack([x * 37 + rng.integers(-3, 4, (n, n)), y * 41 + rng.integers(-3, 4, (n, n)), 2000 + 300 * np.sin(x / 8) * np.cos(y / 8)], -1).reshape(-1, 3).astype(np.int64); xyz = np.concatenate([xyz, rng.integers(0, 4096, (5, 3))])
    for m in ('prev', 'first', 'mean', 'para'): stream(g, xyz, m, 14, 'pos', XYZ_TH, verify=True)
    rx, best = measure_xyz(g, xyz, 12); ii = rng.integers(0, 4096, V); ri = measure_idx(ii, 4096)
    print('selftest: V %d | DC first %d mean %d B | XYZ prev %d first %d mean %d para %d B (best %s) | IDX no-context %d fixed %d B | round trips exact' % (
        V, r['first']['bytes'], r['mean']['bytes'], rx['prev']['bytes'], rx['first']['bytes'], rx['mean']['bytes'], rx['para']['bytes'], best, ri['no_context_bytes'], ri['fixed_length_bytes']))

def arr(d, names):
    for k in names:
        if k in d:
            v = d[k]; return (v.numpy() if hasattr(v, 'numpy') else np.asarray(v)), k
    return None, None

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
    wire = torch.load(a.shared, map_location='cpu', weights_only=False); wire.setdefault('opacity_floor', .999); dbits = int(a.dc[1:]); xbits = int(a.xyz[1:])
    work = Path(tempfile.mkdtemp(prefix='codec-choices-', dir=str(Path(a.out).resolve().parent))); native = work / 'native'
    try:
        exe = tc.lc.native(native); edge = tc.edge_native(native); ctx = tc.ic.native(native); tmp = work / 'enc'; tmp.mkdir()
        quant = delivery.prepare(wire, a.xyz, a.dc, True, 'const'); ordered, perm = tc.reorder(quant, 'strip-fetch')
        pred = tc.lc.graph(exe, ordered, tmp); encoded = tc.encode(ordered, 'edge-arithmetic', 1, exe, edge, tmp, pred, ctx); path = work / 'asset.mslb'; pack(encoded, path, 'auto'); size = path.stat().st_size
        dc, _ = arr(ordered, ['features_dc']); V = dc.shape[0]; qdc = dc.reshape(V, -1).astype(np.int64); faces, _ = arr(ordered, ['_triangle_indices'])
        xyz, _ = arr(ordered, ['triangles_points']); qx = xyz.reshape(V, -1).astype(np.int64)
        shipped = {k: (int(np.asarray(v.numpy() if hasattr(v, 'numpy') else v).nbytes) if hasattr(v, '__len__') and not isinstance(v, (str, bytes)) else None) for k, v in encoded.items() if k in ('top_stream', 'local_residual', 'ctx_mapping', 'ctx_residual')}
        im, kim = arr(ordered, ['app_mapping', 'mapping']); ir, kir = arr(ordered, ['app_residual_mapping', 'residual_mapping']); bank, _ = arr(ordered, ['app_bank']); rbank, _ = arr(ordered, ['app_residual_bank'])
        keys_o, keys_e = sorted(ordered.keys()), sorted(encoded.keys())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if qdc.shape[1] != 3 or qdc.min() < 0 or qdc.max() >= (1 << dbits): raise AssertionError('unexpected DC array: shape %s range %d..%d' % (dc.shape, qdc.min(), qdc.max()))
    out = dict(label=a.label, vertices=int(V), faces=int(len(faces)), file_bytes=int(size), shipped_stream_bytes=shipped); g = Graph(faces, V); L = a.label; MB = 1e6
    # ---- DC
    z = zstd_bytes(qdc); r = measure_dc(g, qdc, dbits); best = min(r, key=lambda k: r[k]['bytes']); nbt = r[best]['bytes']
    out['dc'] = dict(zstd_bytes=int(z), predicted=r, best=best, file_change_percent=round(100 * (nbt - z) / size, 2), dc_change_percent=round(100 * (nbt - z) / z, 2))
    print('DCPRED %s | V %d | file %.3f MB | DC zstd %.3f MB (%.2f b/v) | first %.2f b/v | mean %.2f b/v | best %s: DC %+.1f %%, file %+.2f %% | roundtrip exact' % (L, V, size / MB, z / MB, 8 * z / V, r['first']['bits_per_vertex'], r['mean']['bits_per_vertex'], best, out['dc']['dc_change_percent'], out['dc']['file_change_percent']), flush=True)
    # ---- XYZ
    if qx.shape[1] == 3 and qx.min() >= 0 and qx.max() < (1 << xbits):
        rx, bx = measure_xyz(g, qx, xbits); sh = shipped.get('local_residual'); out['xyz'] = dict(predictors=rx, best=bx, shipped_bytes=sh, parallelogram_available=round(float((g.pb >= 0).mean()), 4), no_earlier_neighbour=round(float((~g.has).mean()), 4))
        f1 = rx['first']['bytes']
        print('XYZPRED %s | q%d | shipped %s | prev %.2f b/v | first %.2f b/v | mean %.2f b/v (%+.1f %% vs first) | para %.2f b/v (%+.1f %% vs first, available %.0f %% of vertices) | best %s (round trip exact), file %+.2f %% vs first' % (
            L, xbits, ('%.3f MB (%.2f b/v)' % (sh / MB, 8 * sh / V)) if sh else 'n/a', rx['prev']['bits_per_vertex'], rx['first']['bits_per_vertex'], rx['mean']['bits_per_vertex'], 100 * (rx['mean']['bytes'] - f1) / f1, rx['para']['bits_per_vertex'], 100 * (rx['para']['bytes'] - f1) / f1,
            100 * (g.pb >= 0).mean(), bx, 100 * (rx[bx]['bytes'] - f1) / size), flush=True)
    else: print('XYZPRED %s | skipped: unexpected position array shape %s range %d..%d for q%d' % (L, xyz.shape, qx.min(), qx.max(), xbits), flush=True)
    # ---- IDX
    if im is not None and ir is not None and shipped.get('ctx_mapping') and shipped.get('ctx_residual'):
        K = int(len(bank)) if bank is not None else int(im.max()) + 1; R = int(len(rbank)) if rbank is not None else int(ir.max()) + 1; a1 = measure_idx(im, K); a2 = measure_idx(ir, R); s1, s2 = shipped['ctx_mapping'], shipped['ctx_residual']
        out['idx'] = dict(K=K, R=R, main=dict(shipped_bytes=s1, **a1), residual=dict(shipped_bytes=s2, **a2))
        tot_s, tot_n = s1 + s2, a1['no_context_bytes'] + a2['no_context_bytes']
        print('IDXCTX %s | main K %d: shipped %.2f b/v, no context %.2f b/v, fixed %d b | residual R %d: shipped %.2f b/v, no context %.2f b/v, fixed %d b | context saves %.1f %% of the index streams, %.2f %% of the file | roundtrip exact' % (
            L, K, 8 * s1 / V, 8 * a1['no_context_bytes'] / V, a1['bits'], R, 8 * s2 / V, 8 * a2['no_context_bytes'] / V, a2['bits'], 100 * (tot_n - tot_s) / tot_n, 100 * (tot_n - tot_s) / size), flush=True)
    else: print('IDXCTX %s | skipped: index arrays %s / %s, shipped streams %s | ordered keys %s | encoded keys %s' % (L, kim, kir, shipped, keys_o, keys_e), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))

if __name__ == '__main__':
    main()
