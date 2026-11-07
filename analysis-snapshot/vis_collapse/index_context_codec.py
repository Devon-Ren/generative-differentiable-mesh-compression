"""Real bytes for the neighbour-context index coder (native/index_ctx.cpp) on the index_rd lambda sweep.

index_rd.py showed that the RD reassignment lowers the ESTIMATED context-coder rate (21.4 -> 16.1 bits/vertex) but
the delivered zstd mapping only falls 21.9 -> 19.6: zstd has no notion of mesh neighbours.  This script codes the
same index arrays with a decodable neighbour-context arithmetic coder (reusing the stage2 coder) and verifies the
round trip, so the table gives REAL bytes for both coders per lambda.  No rendering, no training (seconds on GPU).

    bash stage2_run.sh build_index_ctx.py                         # once per package
    bash stage2_run.sh index_context_codec.py --deliver-dir $W/deliver-geo-only --lambdas 0,0.1,0.3,1,3

Output: <deliver-dir>/index_ctx/sweep.json + sweep.txt (merged with index_rd/sweep.json metrics when present).
Caveat: coded in checkpoint vertex order; the delivered asset is in meshoptimizer order, hit rates are statistically
the same but not bit-identical.  Only imports stage2 + index_rd.py (same directory).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import zstandard as zstd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'native'))
from index_rd import PACKAGE_VERSION, adjacency, reassign, context_bits, log


def earlier_csr(ptr, dst):
    """Keep only neighbours with a smaller index (= already decoded), as int64 CSR."""
    p = ptr.numpy(); d = dst.numpy(); n = len(p) - 1
    src = np.repeat(np.arange(n), np.diff(p)); keep = d < src
    counts = np.bincount(src[keep], minlength=n); ep = np.zeros(n + 1, np.int64); ep[1:] = np.cumsum(counts)
    return ep, d[keep].astype(np.int64)


def zstd_bytes(codes):
    """What the delivered asset does: uint16 container, min(zstd, byte-shuffle + zstd) at level 9, whole array."""
    a = codes.astype(np.uint16); raw = a.tobytes(); shuf = a.view(np.uint8).reshape(-1, 2).T.copy().tobytes()
    c = zstd.ZstdCompressor(level=9); return min(len(c.compress(raw)), len(c.compress(shuf)), len(raw))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--deliver-dir', required=True); p.add_argument('--shared', help='default: <deliver-dir>/shared-final.pt')
    p.add_argument('--lambdas', default='0,0.1,0.3,1,3'); p.add_argument('--sweeps', type=int, default=3); p.add_argument('--maxdeg', type=int, default=32)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--stage2', default='/scratch/zr2245/shared-collapse-stage2-v2')
    a = p.parse_args()
    import _index_ctx
    d = Path(a.deliver_dir); dev = a.device
    shared = torch.load(a.shared or d / 'shared-final.pt', map_location='cpu', weights_only=False)
    b = torch.as_tensor(shared['app_bank'], device=dev).float().reshape(len(shared['app_bank']), -1)
    r = torch.as_tensor(shared['app_residual_bank'], device=dev).float().reshape(len(shared['app_residual_bank']), -1)
    z0 = torch.as_tensor(np.asarray(shared['app_mapping']).astype(np.int64), device=dev)
    u0 = torch.as_tensor(np.asarray(shared['app_residual_mapping']).astype(np.int64), device=dev)
    n = len(z0); K, R = len(b), len(r)
    ptr, dst, maxdeg = adjacency(np.asarray(shared['_triangle_indices']), n); maxdeg = min(maxdeg, a.maxdeg)
    if (ptr[1:] - ptr[:-1]).max() > maxdeg:  # same degree cap as index_rd.py: keep the first maxdeg neighbours
        p0 = ptr.numpy(); dd = dst.numpy(); keep = [dd[p0[v]:min(p0[v + 1], p0[v] + maxdeg)] for v in range(n)]
        dst = torch.from_numpy(np.concatenate(keep)); counts = np.minimum(np.diff(p0), maxdeg); ptr = torch.from_numpy(np.concatenate([[0], np.cumsum(counts)]))
    ep, ed = earlier_csr(ptr, dst)
    full = b[z0] + r[u0]; unit = float((full - b[z0]).square().sum(1).median())
    prev = {}
    if (d / 'index_rd' / 'sweep.json').exists():
        for q in json.loads((d / 'index_rd' / 'sweep.json').read_text())['rows']:
            if q['lam'] is not None: prev[float(q['lam'])] = q
    log(f'N {n} K {K} R {R} maxdeg {maxdeg} earlier-neighbour entries {len(ed)} ({len(ed) / n:.2f}/vertex) | unit {unit:.4g}')

    def code_bytes(codes, Kc):
        c = codes.cpu().numpy().astype(np.int32); t = time.time(); enc = _index_ctx.encode(c, ep, ed, int(Kc)); te = time.time() - t
        back = _index_ctx.decode(enc, ep, ed, int(Kc), n)
        if not np.array_equal(np.asarray(back), c): raise RuntimeError('index_ctx round trip FAILED')
        return len(enc), zstd_bytes(c), te

    rows = []
    for lam in [float(x) for x in a.lambdas.split(',')]:
        z = reassign(full - r[u0], b, z0, ptr, dst, maxdeg, lam, unit, K, a.sweeps) if lam > 0 else z0.clone()
        u = reassign(full - b[z], r, u0, ptr, dst, maxdeg, lam, unit, R, a.sweeps) if lam > 0 else u0.clone()
        hz, bz = context_bits(z, ptr, dst, maxdeg, K, dev); hu, bu = context_bits(u, ptr, dst, maxdeg, R, dev)
        cz, zz, tz = code_bytes(z, K); cu, zu, tu = code_bytes(u, R)
        row = dict(lam=lam, hit_z=hz, hit_u=hu, est_bits=bz + bu, zstd_MB=(zz + zu) / 1e6, ctx_MB=(cz + cu) / 1e6, ctx_bits=8 * (cz + cu) / n, zstd_bits=8 * (zz + zu) / n,
                   ctx_z_MB=cz / 1e6, ctx_u_MB=cu / 1e6, zstd_z_MB=zz / 1e6, zstd_u_MB=zu / 1e6, encode_seconds=tz + tu, proxy_mse=float((b[z] + r[u] - full).square().sum(1).mean()))
        q = prev.get(lam)
        if q: row.update({k: q[k] for k in ('validation', 'validation_refit', 'file_MB', 'mapping_MB') if k in q}); row['test'] = q.get('test')
        rows.append(row)
        log(f'lambda {lam:g}: hit z {hz:.3f} u {hu:.3f} | est {bz + bu:.2f} bits/v | zstd {row["zstd_bits"]:.2f} bits/v ({row["zstd_MB"]:.3f} MB) | ctx {row["ctx_bits"]:.2f} bits/v ({row["ctx_MB"]:.3f} MB) round-trip OK')
    out = d / 'index_ctx'; out.mkdir(exist_ok=True)
    (out / 'sweep.json').write_text(json.dumps(dict(package_version=PACKAGE_VERSION, N=n, K=K, R=R, maxdeg=maxdeg, unit=unit, rows=rows), indent=1))
    lines = [f"{'lambda':>7s} {'hit_z':>6s} {'hit_u':>6s} {'est':>6s} {'zstd':>6s} {'ctx':>6s} {'zstdMB':>7s} {'ctxMB':>7s} {'saved':>6s} {'fileMB':>7s} {'val_refit':>9s} {'test':>7s}"]
    for q in rows:
        tp = (q.get('test') or {}).get('psnr', float('nan')); fmb = q.get('file_MB', float('nan'))
        lines.append(f"{q['lam']:7g} {q['hit_z']:6.3f} {q['hit_u']:6.3f} {q['est_bits']:6.2f} {q['zstd_bits']:6.2f} {q['ctx_bits']:6.2f} {q['zstd_MB']:7.3f} {q['ctx_MB']:7.3f} {q['zstd_MB'] - q['ctx_MB']:6.3f} {fmb - q['zstd_MB'] + q['ctx_MB']:7.3f} {q.get('validation_refit', float('nan')):9.3f} {tp:7.3f}")
    lines.append('bits = bits/vertex for app_mapping + app_residual_mapping; saved = zstdMB - ctxMB; fileMB = index_rd file_MB with the mapping re-coded by ctx')
    (out / 'sweep.txt').write_text('\n'.join(lines) + '\n'); print('\n'.join(lines))


if __name__ == '__main__':
    main()
