"""delaunay_canon_v1.py (2026-10-26): canonical, decoder-reproducible Delaunay tetrahedralisation of a quantised vertex set.

Shared by train_rd_v4.py (encoder side: late restricted-Delaunay rebuild on the delivery grid) and delaunay_check_v1.py
(decoder side: verify that every delivered face is a face of Delaunay(delivered vertices) and estimate the cost of the
face-selection labels).  Pure numpy / scipy, no torch.

Definition (both sides must agree bit for bit):
  q          = delivery-grid integers of the vertices (same rule as stage2 codec/delivery.quantize: low = per-axis min,
               scale = span / (2^bits - 1), q = rint((x - low) / scale))
  perm       = lexsort of the rows of q (x major, then y, then z); canonical rank r = position in perm
  P[r]       = float64(q[perm[r]]) * float32(scale) + float32(low)  + jitter * scale * (hash(r, axis) - 0.5)
  P32        = float32(P)                      <- the array scipy.spatial.Delaunay receives, in canonical order
The jitter (default 2 % of a grid step, hash = splitmix64 of 3*r+axis) breaks the co-sphericity of grid points so the
Delaunay is unique and any correct implementation returns the same tetrahedra; it is far below the grid step, so it does
not change what the mesh looks like (rendering uses the unjittered dequantised positions).
"""
import numpy as np

VERSION = 'delaunay_canon_v1 2026-10-26'
_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def quantize_xyz(x, bits):
    """Identical to stage2 codec/delivery.quantize for mode q<bits>: returns (q uint, low float64[3], scale float64[3])."""
    flat = np.asarray(x, np.float32).reshape(-1, 3).astype(np.float64)
    if not np.isfinite(flat).all():
        raise ValueError('non-finite positions')
    low = flat.min(0); span = flat.max(0) - low
    scale = span / ((1 << bits) - 1); safe = np.where(scale > 0, scale, 1.)
    q = np.rint((flat - low) / safe).clip(0, (1 << bits) - 1)
    return q.astype(np.uint16 if bits <= 16 else np.uint32), low, scale


def dequantize_xyz(q, low, scale):
    """Identical to delivery.dequantize (kind 2): float32 positions."""
    return (np.asarray(q, np.float64).reshape(-1, 3) * np.asarray(scale) + np.asarray(low)).astype(np.float32)


def _hash01(idx):
    """splitmix64 of uint64 idx -> uniform in [0,1); platform independent (pure uint64 arithmetic)."""
    with np.errstate(over='ignore'):
        x = (np.asarray(idx, np.uint64) * np.uint64(0x9E3779B97F4A7C15)) & _M64
        x ^= x >> np.uint64(30); x = (x * np.uint64(0xBF58476D1CE4E5B9)) & _M64
        x ^= x >> np.uint64(27); x = (x * np.uint64(0x94D049BB133111EB)) & _M64
        x ^= x >> np.uint64(31)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def canonical_points(q, low, scale, jitter=0.02):
    """-> (perm, P32).  P32[r] is the Delaunay input point of original vertex perm[r]."""
    q = np.asarray(q).reshape(-1, 3)
    perm = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    qs = q[perm].astype(np.float64)
    if len(np.unique(qs, axis=0)) != len(qs):
        raise ValueError('duplicate grid points: merge them before the canonical Delaunay')
    low32 = np.asarray(low, np.float32).astype(np.float64); scale32 = np.asarray(scale, np.float32).astype(np.float64)
    P = qs * scale32 + low32
    if jitter > 0:
        r = np.arange(len(qs), dtype=np.uint64)[:, None] * np.uint64(3) + np.arange(3, dtype=np.uint64)[None, :]
        P = P + (jitter * scale32)[None, :] * (_hash01(r) - 0.5)
    return perm, P.astype(np.float32)


def delaunay_tets(P32):
    """Delaunay tetrahedra of the canonical points (scipy / Qhull, default options)."""
    from scipy.spatial import Delaunay
    return Delaunay(np.asarray(P32, np.float32)).simplices.astype(np.int64)


def interior_faces(tets):
    """Faces shared by exactly two tetrahedra -> (faces [M,3] sorted rows, tet_pair [M,2]).  Boundary (hull) faces are
    never surface candidates (effrdel keeps only faces with two incident tets)."""
    T = len(tets)
    f = np.concatenate([tets[:, [1, 2, 3]], tets[:, [0, 2, 3]], tets[:, [0, 1, 3]], tets[:, [0, 1, 2]]])
    f = np.sort(f, axis=1)
    owner = np.tile(np.arange(T, dtype=np.int64), 4)
    key = np.lexsort((f[:, 2], f[:, 1], f[:, 0]))
    f, owner = f[key], owner[key]
    same = np.all(f[1:] == f[:-1], axis=1)
    first = np.flatnonzero(same)  # f[first] == f[first+1]
    if len(first) > 1 and np.any(np.diff(first) == 1):
        raise RuntimeError('a face owned by three tetrahedra (degenerate Delaunay)')
    return f[first], np.stack([owner[first], owner[first + 1]], 1)


def face_rows_to_ids(faces, table):
    """Index of each row of `faces` (sorted triples) in `table` (sorted-row, lexsorted unique table), -1 if absent."""
    faces = np.sort(np.asarray(faces, np.int64), axis=1)
    key = np.lexsort((table[:, 2], table[:, 1], table[:, 0]))
    ts = table[key]
    # encode rows as one int (vertex count < 2^21 -> 63 bits)
    def enc(a): return (a[:, 0].astype(np.int64) << 42) | (a[:, 1].astype(np.int64) << 21) | a[:, 2].astype(np.int64)
    te, fe = enc(ts), enc(faces)
    pos = np.searchsorted(te, fe)
    pos_c = np.minimum(pos, len(te) - 1)
    hit = (te[pos_c] == fe) if len(te) else np.zeros(len(fe), bool)
    out = np.full(len(fe), -1, np.int64); out[hit] = key[pos_c[hit]]
    return out


def label_entropy(cand_faces, tet_pair, selected_ids, n_vertices):
    """Bits per vertex to code the 'selected' bit of every candidate face.
    H0: order-0; H1: conditioned on how many of the 6 sibling faces (other faces of the two incident tets) are selected
    (non-causal, so a lower bound for a causal context coder)."""
    M = len(cand_faces); sel = np.zeros(M, bool); sel[selected_ids[selected_ids >= 0]] = True
    n1 = int(sel.sum()); p = n1 / max(M, 1)

    def H(p):
        return 0.0 if p <= 0 or p >= 1 else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    H0 = M * H(p) / n_vertices
    T = int(tet_pair.max()) + 1 if M else 0
    per_tet = np.bincount(tet_pair.reshape(-1), weights=np.repeat(sel, 2), minlength=T)  # selected faces per tet
    ctx = (per_tet[tet_pair[:, 0]] + per_tet[tet_pair[:, 1]] - 2 * sel).astype(np.int64)  # 0..6 siblings selected
    H1 = 0.0; table = {}
    for c in range(7):
        m = ctx == c; n = int(m.sum())
        if n:
            pc = float(sel[m].mean()); H1 += n * H(pc); table[c] = (n, round(pc, 4))
    return dict(candidates=M, selected=n1, candidates_per_vertex=round(M / n_vertices, 3), p_selected=round(p, 4),
                H0_bits_per_vertex=round(H0, 3), H1_bits_per_vertex=round(H1 / n_vertices, 3), context_table=table)


def selftest():
    rng = np.random.default_rng(0)
    x = (rng.random((3000, 3)) * 4).astype(np.float32); x[:, 2] *= 0.3  # flat-ish cloud
    q, low, scale = quantize_xyz(x, 10)            # coarse grid -> many co-spherical points and some duplicates
    _, idx = np.unique(q, axis=0, return_index=True); q = q[np.sort(idx)]   # merge duplicates for the test
    x2 = dequantize_xyz(q, low, scale); q2, low2, scale2 = quantize_xyz(x2, 10)
    assert np.array_equal(q2, q) and np.allclose(low2, low) and np.allclose(scale2, scale), 'quantiser not idempotent'
    perm, P = canonical_points(q, low, scale)
    perm_b, P_b = canonical_points(q[::-1], low, scale)         # different input order -> same canonical points
    assert np.array_equal(P, P_b), 'canonical points depend on input order'
    t1 = delaunay_tets(P); t2 = delaunay_tets(P_b)
    f1, _ = interior_faces(t1); f2, _ = interior_faces(t2)
    assert np.array_equal(f1, f2), 'Delaunay differs for identical canonical input'
    # no-jitter grid points are degenerate: Qhull may still succeed but the jittered result must be a proper Delaunay
    ids = face_rows_to_ids(f1[::7], f1); assert np.array_equal(ids, np.arange(0, len(f1), 7))
    assert face_rows_to_ids(np.array([[0, 1, 2]]), f1)[0] in (-1, int(np.flatnonzero((f1 == [0, 1, 2]).all(1))[0]) if ((f1 == [0, 1, 2]).all(1)).any() else -1)
    pick = f1[::9]; ent = label_entropy(f1, _, face_rows_to_ids(pick, f1), len(q))
    assert ent['selected'] == len(pick) and ent['H1_bits_per_vertex'] <= ent['H0_bits_per_vertex'] + 1e-9
    print('delaunay_canon_v1 selftest PASS |', len(q), 'points', len(t1), 'tets', len(f1), 'interior faces |', ent)


if __name__ == '__main__':
    selftest()
