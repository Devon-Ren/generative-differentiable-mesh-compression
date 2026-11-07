"""CPU redundancy statistics for a MeshSplatting asset (no GPU, does not modify anything).

Measures how much of the current 41.4 MB bitstream is predictable from mesh connectivity:
  1. topology structure (manifoldness, components, valence)
  2. DC (q8) entropy: raw vs. causal 1-ring neighbour prediction
  3. shared-dictionary index entropy: raw vs. "same code as a causal neighbour" scheme
  4. vertex_weight (opacity) range under the 0.999 floor
  5. how many mesh faces are faces of the 3D Delaunay tetrahedralization of the
     (q18-quantized) vertices, i.e. whether connectivity is predictable from geometry

"Causal" = neighbours earlier in a BFS order that a decoder can rebuild from the
already decoded topology, so the numbers are achievable rates up to entropy-coder
overhead (zero-order empirical entropy, adaptive models are usually close).

Usage (HPC, any CPU node; Delaunay 'full' needs a large-memory node):
  python redundancy_stats.py --ckpt .../iteration_30000/point_cloud_state_dict.pt \
      --shared .../shared/residual_k4096_r1024_f1/asset.mslb \
      --delaunay crop --out room-redundancy.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


def entropy_bits(x):
    """Zero-order empirical entropy (bits/symbol) of an integer array (rows = symbols)."""
    x = np.asarray(x)
    if x.ndim > 1:
        x = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1]))).ravel()
    if x.size == 0:
        return 0.0
    _, c = np.unique(x, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def to_np(t):
    if hasattr(t, 'detach'):
        t = t.detach().cpu().numpy()
    return np.asarray(t)


def load_ckpt(path):
    import torch
    s = torch.load(path, map_location='cpu', weights_only=False)
    out = dict(xyz=to_np(s['triangles_points']).astype(np.float32),
               faces=to_np(s['_triangle_indices']).astype(np.int64),
               dc=to_np(s['features_dc']).reshape(-1, 3).astype(np.float32))
    if 'vertex_weight' in s:
        out['vertex_weight'] = to_np(s['vertex_weight']).reshape(-1).astype(np.float32)
    del s  # features_rest (~575 MB) is not needed; keeps peak memory low on small CPU nodes
    return out


def load_shared(path, pkg=None):
    """Read an MSLB001 asset. Self-contained copy of codec_benchmark.unpack (no checksum
    check, numpy only); --pkg is accepted for compatibility but no longer needed."""
    import struct, zstandard
    state = {}
    with Path(path).open('rb') as f:
        if f.read(8) != b'MSLB001\0': raise ValueError('Not an MSLB001 asset')
        length = struct.unpack('<Q', f.read(8))[0]
        meta = json.loads(f.read(length)); base = f.tell()
        for field in meta['fields']:
            if field['kind'] == 'scalar':
                state[field['name']] = field['value']; continue
            dtype = np.dtype(field['dtype']); parts = []
            for b in field['blocks']:
                f.seek(base + b['offset']); data = f.read(b['size'])
                if b['compressed']: data = zstandard.ZstdDecompressor().decompress(data)
                if b['mode'] == 'shuffle':
                    data = np.frombuffer(data, np.uint8).reshape(dtype.itemsize, -1).T.copy().tobytes()
                elif b['mode'] == 'index-sequence':
                    import meshoptimizer as mo
                    data = mo.decode_index_sequence(b['rows'] * b['cols'], 4, data).astype(dtype).tobytes()
                elif b['mode'] == 'vertex-buffer':
                    import meshoptimizer as mo
                    data = mo.decode_vertex_buffer(b['rows'], 12, data).tobytes()
                elif b['mode'] != 'raw': raise ValueError('Unknown block mode ' + b['mode'])
                parts.append(np.frombuffer(data, dtype))
            state[field['name']] = np.concatenate(parts).reshape(field['shape'])
    return state


def load_shared_any(path):
    """asset.mslb (packed) or best.pt (ResidualDirectional state_dict: bank/mapping/rows/correction/residual_mapping)."""
    if str(path).endswith('.mslb'):
        return load_shared(path)
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=True)
    used, z = np.unique(to_np(sd['mapping']), return_inverse=True)
    out = dict(app_mapping=z, app_bank=to_np(sd['bank'])[used])
    if 'residual_mapping' in sd:
        out.update(app_rows=to_np(sd['rows']), app_residual_mapping=to_np(sd['residual_mapping']),
                   app_residual_bank=to_np(sd['correction']))
    return out


def quantize(a, bits):
    """Same per-axis min-max uniform quantizer as stage2/codec/delivery.py."""
    flat = a.reshape(-1, 3).astype(np.float64)
    low = flat.min(0); span = flat.max(0) - low
    scale = span / ((1 << bits) - 1); safe = np.where(scale > 0, scale, 1.)
    return np.rint((flat - low) / safe).clip(0, (1 << bits) - 1).astype(np.int64)


# ---------------------------------------------------------------- topology
def adjacency(faces, n):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = e[e[:, 0] != e[:, 1]]
    e = np.concatenate([e, e[:, ::-1]])
    a = sparse.csr_matrix((np.ones(len(e), np.int8), (e[:, 0], e[:, 1])), shape=(n, n))
    a.sum_duplicates(); a.data[:] = 1
    return a


def topology_stats(faces, n):
    und = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
    _, inv, mult = np.unique(und, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    hist = {str(k): int((mult == k).sum()) for k in (1, 2, 3)}
    hist['4+'] = int((mult >= 4).sum())
    face_mult = mult[inv].reshape(3, -1).T                     # per face, per edge
    isolated = int((face_mult.max(1) == 1).sum())
    sf = np.sort(faces, 1)
    dup = len(sf) - len(np.unique(sf, axis=0))
    adj = adjacency(faces, n)
    ncomp, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels)
    valence = np.diff(adj.indptr)
    used = np.zeros(n, bool); used[faces.ravel()] = True
    return dict(vertices=n, faces=len(faces), unused_vertices=int((~used).sum()),
                edges=len(mult), edge_face_multiplicity=hist,
                nonmanifold_edge_fraction=float((mult >= 3).mean()),
                boundary_edge_fraction=float((mult == 1).mean()),
                isolated_faces=isolated, duplicate_faces=int(dup),
                components=int(ncomp), components_size1_or_2=int((sizes <= 2).sum()),
                largest_component_fraction=float(sizes.max() / n),
                mean_valence=float(valence.mean()),
                valence_p50_p90_p99=[float(np.percentile(valence, q)) for q in (50, 90, 99)]), adj, labels


def csr_gather(indptr, indices, rows):
    counts = indptr[rows + 1] - indptr[rows]
    starts = np.repeat(indptr[rows], counts)
    offs = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(rows, counts), indices[starts + offs]


def bfs_rank(adj, labels):
    """Decoder-reproducible order: per component, BFS from its smallest vertex id."""
    n = adj.shape[0]
    level = np.full(n, -1, np.int64)
    order = np.argsort(labels, kind='stable')
    first = np.r_[True, labels[order][1:] != labels[order][:-1]]
    frontier = order[first]
    level[frontier] = 0; d = 0
    while frontier.size:
        _, nb = csr_gather(adj.indptr, adj.indices, frontier)
        nb = np.unique(nb[level[nb] < 0])
        d += 1; level[nb] = d; frontier = nb
    order = np.lexsort((np.arange(n), level, labels))
    rank = np.empty(n, np.int64); rank[order] = np.arange(n)
    return rank, int(d)


def causal_edges(adj, rank):
    """Directed pairs (i, j) with j a 1-ring neighbour decoded before i."""
    coo = adj.tocoo()
    keep = rank[coo.col] < rank[coo.row]
    return coo.row[keep], coo.col[keep]


# ---------------------------------------------------------------- attributes
def dc_stats(q, n, ci, cj, rank):
    res = {}
    res['raw_bits_per_vertex'] = float(sum(entropy_bits(q[:, c]) for c in range(3)))
    res['raw_joint_bits_per_vertex'] = entropy_bits(q)
    # previous vertex in decode order (what a generic delta coder could do)
    order = np.argsort(rank); prev = np.empty(n, np.int64); prev[order[1:]] = order[:-1]; prev[order[0]] = order[0]
    res['delta_prev_bits_per_vertex'] = float(sum(entropy_bits(q[:, c] - q[prev, c]) for c in range(3)))
    cnt = np.bincount(ci, minlength=n)
    for name, (ii, jj) in {'causal_mean': (ci, cj)}.items():
        pred = np.zeros((n, 3))
        for c in range(3):
            pred[:, c] = np.bincount(ii, weights=q[jj, c], minlength=n) / np.maximum(cnt, 1)
        pred = np.rint(pred).astype(np.int64)
        nocausal = cnt == 0
        pred[nocausal] = q[prev[nocausal]]
        r = q - pred
        res[name + '_bits_per_vertex'] = float(sum(entropy_bits(r[:, c]) for c in range(3)))
        res[name + '_residual_abs_p50_p90'] = [float(np.percentile(np.abs(r), p)) for p in (50, 90)]
    res['vertices_without_causal_neighbour'] = int((cnt == 0).sum())
    return res


def index_stats(z, n, ci, cj, K):
    """Cost of: flag(hit) + [hit: which distinct causal-neighbour code | miss: zero-order code]."""
    res = dict(raw_bits=float(np.log2(K)), zero_order_bits=entropy_bits(z))
    pairs = np.unique(np.stack([ci, z[cj]], 1), axis=0)          # distinct (vertex, neighbour code)
    distinct = np.bincount(pairs[:, 0], minlength=n)
    key = pairs[:, 0] * (int(z.max()) + 1) + pairs[:, 1]
    hit = np.isin(np.arange(n) * (int(z.max()) + 1) + z, key)
    p = hit.mean()
    h_flag = float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))) if 0 < p < 1 else 0.0
    sel = np.log2(np.maximum(distinct[hit], 1)).sum() / n          # uniform choice among distinct codes
    miss = entropy_bits(z[~hit]) * (~hit).mean()
    res.update(hit_rate=float(p), mean_distinct_causal_codes=float(distinct.mean()),
               neighbour_scheme_bits=float(h_flag + sel + miss))
    res['best_simple_bits'] = min(res['zero_order_bits'], res['neighbour_scheme_bits'])
    return res


def knn_stats(xyz, faces, n, ks=(4, 8, 12, 16, 24)):
    """Can connectivity be predicted from decoded geometry at all?  For every undirected mesh
    edge (a,b): rank of b among a's nearest neighbours (and vice versa, take the better one).
    Fraction with rank < k = how many edges a 'pick one of my k nearest' scheme could express."""
    from scipy.spatial import cKDTree
    kmax = max(ks)
    tree = cKDTree(xyz.astype(np.float64))
    _, idx = tree.query(xyz.astype(np.float64), k=kmax + 1, workers=-1)
    idx = idx[:, 1:].astype(np.int64)                      # drop self (duplicates only lower the hit rate)
    und = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), 1)
    und = np.unique(und[und[:, 0] != und[:, 1]], axis=0)
    rank = np.full(len(und), kmax, np.int64)
    for start in range(0, len(und), 1 << 20):
        a, b = und[start:start + (1 << 20)].T
        for x, y in ((a, b), (b, a)):
            m = idx[x] == y[:, None]
            r = np.where(m.any(1), m.argmax(1), kmax)
            rank[start:start + len(a)] = np.minimum(rank[start:start + len(a)], r)
    edges_per_vertex = len(und) / n
    out = dict(edges=int(len(und)), edges_per_vertex=float(edges_per_vertex),
               edge_within_k_fraction={str(k): float((rank < k).mean()) for k in ks},
               rank_p50_p90_p99=[float(np.percentile(rank, q)) for q in (50, 90, 99)])
    # Scheme: per vertex, k flags (is my j-th neighbour connected to me, counted once per edge)
    for k in (8, 16):
        p = (rank < k).sum() / (n * k)
        h = float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))) if 0 < p < 1 else 0.0
        out[f'flag_scheme_k{k}_bits_per_vertex'] = float(h * k)
        out[f'flag_scheme_k{k}_uncovered_edges_per_vertex'] = float((rank >= k).sum() / n)
    out['note'] = ('Edges only; faces still need ~1 extra symbol per face. Zero-order flag entropy; '
                   'a context model (distance rank, neighbour flags) should do better.')
    return out


def delaunay_stats(xyz_q, faces, mode, crop_frac, seed):
    from scipy.spatial import Delaunay
    n = len(xyz_q)
    rng = np.random.default_rng(seed)
    if mode == 'full':
        inner = np.ones(n, bool); pts_ids = np.arange(n)
    else:
        # box around a random vertex holding ~crop_frac of points; evaluate an inner box (margin excluded)
        lo, hi = xyz_q.min(0), xyz_q.max(0)
        c = xyz_q[rng.integers(n)]
        half = (hi - lo) / 2.0
        for _ in range(60):
            m = np.all(np.abs(xyz_q - c) <= half, 1)
            if m.mean() <= crop_frac: break
            half *= 0.85
        pts_ids = np.flatnonzero(m)
        inner = np.zeros(n, bool)
        inner[pts_ids[np.all(np.abs(xyz_q[pts_ids] - c) <= 0.8 * half, 1)]] = True
    log(f'Delaunay on {len(pts_ids)} points ({mode})')
    pts = xyz_q[pts_ids].astype(np.float64)
    uniq, first, back = np.unique(pts, axis=0, return_index=True, return_inverse=True)
    tri = Delaunay(uniq, qhull_options='Qbb Qc Qz Q12')
    s = pts_ids[first][tri.simplices]                              # global ids of unique points
    cand = np.sort(np.concatenate([s[:, [0, 1, 2]], s[:, [0, 1, 3]], s[:, [0, 2, 3]], s[:, [1, 2, 3]]]), 1)
    cand = np.unique(cand, axis=0)
    # map duplicate (post-quantization) vertices onto their representative
    rep = np.arange(n); rep[pts_ids] = pts_ids[first][back.reshape(-1)]
    fm = np.sort(rep[faces], 1)
    evalf = inner[faces].all(1) & (fm[:, 0] != fm[:, 1]) & (fm[:, 1] != fm[:, 2])
    fm = fm[evalf]
    nmax = n + 1
    kc = (cand[:, 0] * nmax + cand[:, 1]) * nmax + cand[:, 2]
    kf = (fm[:, 0] * nmax + fm[:, 1]) * nmax + fm[:, 2]
    hit = np.isin(kf, kc)
    cand_inner = inner[cand].all(1).sum()
    nin = int(inner.sum())
    p_sel = hit.sum() / max(cand_inner, 1)
    h = float(-(p_sel * np.log2(p_sel) + (1 - p_sel) * np.log2(1 - p_sel))) if 0 < p_sel < 1 else 0.0
    exceptions = (~hit).sum()
    return dict(mode=mode, points=int(len(pts_ids)), evaluated_vertices=nin, evaluated_faces=int(evalf.sum()),
                duplicate_points_after_quantization=int(len(pts) - len(uniq)),
                mesh_faces_in_delaunay_fraction=float(hit.mean()),
                delaunay_candidate_faces_per_vertex=float(cand_inner / max(nin, 1)),
                selected_fraction_of_candidates=float(p_sel),
                est_membership_bits_per_vertex=float(h * cand_inner / max(nin, 1)),
                est_exception_faces_per_vertex=float(exceptions / max(nin, 1)),
                bias='crop mode drops far-away points, which can only ADD Delaunay faces, so the hit rate is '
                     'biased upward; confirm with --delaunay full on a large-memory node' if mode == 'crop' else 'exact',
                note='Zero-order membership bits; a context model (face size, neighbour membership) should do better. '
                     'Exceptions (non-Delaunay faces) still need explicit coding.')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--shared', help='shared asset.mslb (original vertex order) for dictionary-index stats')
    p.add_argument('--pkg', help='extracted meshsplatting-*-complete-v1 folder (for its unpack)')
    p.add_argument('--delaunay', choices=['off', 'crop', 'full'], default='crop')
    p.add_argument('--knn', action='store_true', help='mesh-edge vs k-nearest-neighbour coverage (a few minutes, ~1.5 GB)')
    p.add_argument('--crop-frac', type=float, default=0.05)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out = {}
    log('loading', a.ckpt); s = load_ckpt(a.ckpt)
    n = len(s['xyz']); faces = s['faces']
    log('topology'); out['topology'], adj, labels = topology_stats(faces, n)
    log('bfs order'); rank, depth = bfs_rank(adj, labels); out['topology']['bfs_depth'] = depth
    ci, cj = causal_edges(adj, rank)
    log('dc'); out['dc_q8'] = dc_stats(quantize(s['dc'], 8), n, ci, cj, rank)
    if 'vertex_weight' in s:
        w = 1 / (1 + np.exp(-s['vertex_weight'].astype(np.float64)))
        op = 0.999 + 0.001 * w
        out['vertex_weight'] = dict(opacity_min=float(op.min()), opacity_max=float(op.max()),
                                    sigmoid_p01_p50_p99=[float(np.percentile(w, q)) for q in (1, 50, 99)],
                                    note='alpha=min(0.999, opacity*C): opacity in [0.999,1] changes alpha by <=0.1%')
    Path(a.out).write_text(json.dumps(out, indent=2))          # partial results survive later failures
    if a.shared:
        try:
            log('shared asset'); sh = load_shared_any(a.shared)
            z = to_np(sh['app_mapping']).astype(np.int64)
            if len(z) != n: raise ValueError('shared asset vertex count differs from checkpoint')
            if 'features_dc' in sh:
                out['shared_order_check_max_abs_dc_diff'] = float(np.abs(to_np(sh['features_dc']).reshape(-1, 3) - s['dc']).max())
            out['primary_index'] = index_stats(z, n, ci, cj, len(to_np(sh['app_bank'])))
            if 'app_residual_mapping' in sh:
                rows = to_np(sh['app_rows']).astype(np.int64)
                if len(rows) == n:
                    u = np.empty(n, np.int64); u[rows] = to_np(sh['app_residual_mapping']).astype(np.int64)
                    out['residual_index'] = index_stats(u, n, ci, cj, len(to_np(sh['app_residual_bank'])))
                    out['joint_primary_residual_zero_order_bits'] = entropy_bits(np.stack([z, u], 1))
        except Exception as e:
            out['shared_error'] = repr(e); log('shared asset skipped:', repr(e))
        Path(a.out).write_text(json.dumps(out, indent=2))
    if a.knn:
        log('knn'); out['knn'] = knn_stats(quantize(s['xyz'], 18).astype(np.float64), faces, n)
        Path(a.out).write_text(json.dumps(out, indent=2))
    if a.delaunay != 'off':
        log('delaunay'); out['delaunay'] = delaunay_stats(quantize(s['xyz'], 18), faces, a.delaunay, a.crop_frac, a.seed)
    out['current_bits_per_vertex_dense_41MB'] = dict(topology=30.43, xyz=29.92, dc=15.06, app_mapping=10.79,
                                                     app_residual_mapping=10.24, vertex_weight=5.26, banks=1.92)
    Path(a.out).write_text(json.dumps(out, indent=2))
    log('wrote', a.out); print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
