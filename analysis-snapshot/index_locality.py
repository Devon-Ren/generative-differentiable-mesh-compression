"""index_locality.py v1 (2026-10-16): how spatially correlated are the shared-SH codebook indices on the mesh?
For each deliver dir (shared-final.pt + summary.json): fraction of mesh edges whose endpoints share the primary code z
(and residual code u), the same fraction expected if codes were independent (sum_k p_k^2), the lift, and the real
bits/vertex the context coder achieved vs raw log2(K). H_given_nbr is a plug-in estimate from the edge pair table and is
biased LOW when K^2 is comparable to the edge count (K16384): quote the ctx coder's real bits as the honest number. CPU only.  Usage:
    python index_locality.py <out.csv> <deliver_dir> [<deliver_dir> ...]
"""
import sys, json, csv, math, time
from pathlib import Path
import numpy as np, torch


def undirected_edges(faces, n):
    f = np.asarray(faces, np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]); e = e[e[:, 0] != e[:, 1]]
    lo = np.minimum(e[:, 0], e[:, 1]); hi = np.maximum(e[:, 0], e[:, 1])
    key = np.unique(lo * n + hi); return key // n, key % n


def stats(code, a, b):
    code = np.asarray(code, np.int64); K = int(code.max()) + 1
    p = np.bincount(code, minlength=K) / len(code); indep = float((p * p).sum())
    hit = float((code[a] == code[b]).mean()); ent = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    # conditional entropy of a vertex code given ONE neighbour code (pairwise), empirical over edges
    pair = np.concatenate([code[a] * K + code[b], code[b] * K + code[a]])          # symmetric: both directions
    keys, cnt = np.unique(pair, return_counts=True); pj = cnt / cnt.sum()
    pa = np.bincount(keys // K, weights=pj, minlength=K)                            # marginal of the conditioning vertex
    cond = float(-(pj * np.log2(pj / pa[keys // K])).sum())                        # H(code_j | code_i) over edges, sparse
    return dict(K=K, hit=hit, indep=indep, lift=hit / max(indep, 1e-12), entropy_bits=ent, cond_entropy_bits=float(cond), used=int((p > 0).sum()))


def main():
    out = Path(sys.argv[1]); dirs = [Path(d) for d in sys.argv[2:]]; rows = []
    for d in dirs:
        t0 = time.time(); pt = d / 'shared-final.pt'
        if not pt.exists(): print('skip (no shared-final.pt):', d, flush=True); continue
        w = torch.load(pt, map_location='cpu', weights_only=False)
        faces = w['_triangle_indices'].numpy() if isinstance(w['_triangle_indices'], torch.Tensor) else np.asarray(w['_triangle_indices'])
        z = np.asarray(w['app_mapping']); u = np.asarray(w['app_residual_mapping']); n = len(z)
        a, b = undirected_edges(faces, n)
        sz = stats(z, a, b); su = stats(u, a, b) if len(np.asarray(w['app_residual_bank'])) > 1 else None
        summ = json.loads((d / 'summary.json').read_text()) if (d / 'summary.json').exists() else {}
        pb = summ.get('codec', {}).get('payload_bytes', {}); cfg = json.loads((d / 'config.json').read_text()) if (d / 'config.json').exists() else {}
        ctx_z = 8 * pb.get('ctx_mapping', pb.get('app_mapping', 0)) / n; ctx_u = 8 * pb.get('ctx_residual', pb.get('app_residual_mapping', 0)) / n
        row = dict(dir=str(d), level=cfg.get('coarse', ''), vertices=n, edges=len(a), K=sz['K'], K_used=sz['used'], hit_z=round(sz['hit'], 4), indep_z=round(sz['indep'], 6),
                   lift_z=round(sz['lift'], 1), H_z=round(sz['entropy_bits'], 3), H_z_given_nbr=round(sz['cond_entropy_bits'], 3), raw_bits_z=round(math.log2(sz['K']), 2), ctx_bits_z=round(ctx_z, 3),
                   R=su['K'] if su else 1, hit_u=round(su['hit'], 4) if su else '', indep_u=round(su['indep'], 6) if su else '', lift_u=round(su['lift'], 1) if su else '',
                   H_u=round(su['entropy_bits'], 3) if su else '', H_u_given_nbr=round(su['cond_entropy_bits'], 3) if su else '', ctx_bits_u=round(ctx_u, 3) if su else '',
                   file_MB=summ.get('file_MB', ''), test_psnr=summ.get('delivery_test', {}).get('psnr', ''))
        rows.append(row)
        print(f"{d.name:24s} N {n} K {sz['K']} hit_z {sz['hit']:.3f} (indep {sz['indep']:.2e}, lift {sz['lift']:.0f}x) H {sz['entropy_bits']:.2f} -> H|nbr {sz['cond_entropy_bits']:.2f} bits, ctx coder {ctx_z:.2f} b/v"
              + (f" | R {su['K']} hit_u {su['hit']:.3f} lift {su['lift']:.0f}x H|nbr {su['cond_entropy_bits']:.2f} ctx {ctx_u:.2f}" if su else '') + f" | {time.time() - t0:.0f} s", flush=True)
    with out.open('w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print('wrote', out, len(rows), 'rows')


if __name__ == '__main__':
    main()
