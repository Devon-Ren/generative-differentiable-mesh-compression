"""rd_rebuild_v2.py (2026-10-26): the v5 rebuild events used by train_rd_v5.py (v4's rd_rebuild_v1.py is kept unchanged).

What v4 got wrong (Room probe: coverage 100 % but -1.8 dB): every round after the first used the PREVIOUS round's Delaunay
output as the restriction soup, so the surface drifted (1.34 M -> 1.04 M faces over 7 rounds, never converged), and the
single event at 26000 left only 4000 appearance-only steps to absorb a topology that is ~30 % new faces.
v5: the restriction target is built ONCE per event from the cleaned mesh (effrdel's own pieces: LBVH on that soup,
uploaded once); a round only changes the Delaunay vertex set (unused vertices removed) and queries the SAME soup.
Events (train_rd_v5.py --rebuild_iters 20000 26000):
  early     clean (faces + vertices) -> restricted Delaunay of the float positions against the cleaned mesh ->
            face-only clean -> unused vertices dropped; XYZ keeps training (nothing is frozen)
  final     clean -> snap XYZ to the delivery grid (--xyz qN, stage2 quantiser), merge duplicate grid points ->
            canonical jittered points (delaunay_canon_v1) -> restricted Delaunay against the snapped cleaned mesh until
            every vertex is used -> face-only clean (vertices kept) -> XYZ frozen by the caller (vertex lr = 0)
  snap_only clean -> snap -> freeze, faces unchanged: the control that isolates the cost of quantise + freeze
The stats of each event (rebuild-<iter>.json) include the fraction of soup faces retained by the restriction and the
fraction of output faces that are new.  CPU unit test: python test_rd_rebuild_cpu_v2.py (fake restriction, no GPU)."""
import time
import numpy as np
import torch
from scipy.spatial import Delaunay
from delaunay_canon_v1 import quantize_xyz, dequantize_xyz, canonical_points, VERSION as CANON_VERSION

VERSION = 'rd_rebuild_v2 2026-10-26'
STAGES = ('early', 'final', 'snap_only')


def _reset_face_trackers(triangles):
    T = triangles._triangle_indices.shape[0]
    dev = triangles.vertices.device
    triangles.image_size = torch.zeros((T,), dtype=torch.float, device=dev)
    triangles.importance_score = torch.zeros((T,), dtype=torch.float, device=dev)
    triangles.pixel_count = torch.zeros((T,), dtype=torch.int, device=dev)


def _set_faces(triangles, faces_np):
    triangles._triangle_indices = torch.as_tensor(np.asarray(faces_np, dtype=np.int64), device=triangles.vertices.device).contiguous().to(torch.int32)
    _reset_face_trackers(triangles)


def _vertices_np(triangles):
    return triangles.vertices.detach().cpu().numpy().astype(np.float32)


def _faces_np(triangles):
    return triangles._triangle_indices.long().cpu().numpy()


def _face_keys(f):
    """Orientation-free unique face rows (sorted vertex triples)."""
    f = np.asarray(f, dtype=np.int64).reshape(-1, 3)
    return np.unique(np.sort(f, axis=1), axis=0) if len(f) else f


def _n_common(keys_a, keys_b):
    if len(keys_a) == 0 or len(keys_b) == 0:
        return 0
    _, cnt = np.unique(np.concatenate([keys_a, keys_b]), axis=0, return_counts=True)
    return int((cnt == 2).sum())


def _used_mask(triangles):
    V = int(triangles.vertices.shape[0])
    used = np.zeros(V, bool)
    f = _faces_np(triangles)
    if len(f):
        used[f.reshape(-1)] = True
    return used


class FixedSoupRestriction:
    """Restricted Delaunay against a FIXED triangle soup, assembled from effrdel's own pieces (rdel.run is exactly this
    with soup == Delaunay input).  __call__(P32) returns the interior Delaunay faces of P32 whose dual segment (between the
    circumcentres of the two incident tets) hits the soup, as (N, 3) int64 rows in P32's index space."""

    def __init__(self, soup_verts, soup_faces):
        import rdel  # effrdel; its namespace carries lb, Grid, compute_cc, ti_init, upload_bvh_and_mesh, query_segments_any_hit
        self.rdel = rdel
        v = np.ascontiguousarray(np.asarray(soup_verts, dtype=np.float32))
        f = np.ascontiguousarray(np.asarray(soup_faces, dtype=np.int32))
        rdel.ti_init()
        bvh = rdel.lb.build_lbvh(v, f)
        rdel.upload_bvh_and_mesh(bvh, v, f)
        self.n_soup_faces = int(len(f))
        self.calls = 0

    def __call__(self, P32):
        rdel = self.rdel
        self.calls += 1
        P = np.ascontiguousarray(np.asarray(P32, dtype=np.float32))
        tets_np = Delaunay(P).simplices.astype(np.int64)
        verts = torch.from_numpy(P).to('cuda')
        tets = torch.from_numpy(tets_np).to('cuda')
        grid = rdel.Grid(verts, tets)
        valid = grid.face_tet[:, 1] >= 0
        face_tet = grid.face_tet[valid].clone()
        faces = grid.faces[valid].clone()
        del grid
        torch.cuda.empty_cache()
        cc = rdel.compute_cc(verts, tets)
        torch.cuda.empty_cache()
        hit = rdel.query_segments_any_hit(cc[face_tet[:, 0]], cc[face_tet[:, 1]])
        out = faces[hit.to(faces.device)].cpu().numpy().astype(np.int64).reshape(-1, 3)
        del cc, faces, face_tet, verts, tets
        torch.cuda.empty_cache()
        return out


def snap_to_grid(triangles, bits):
    """Snap XYZ to the delivery grid, merging vertices that share a grid cell (faces remapped, degenerate / duplicate faces
    dropped).  Returns (q, low, scale, n_duplicates) with the positions set to dequantize(q) and the quantiser verified
    idempotent on them (exactly what the decoder reconstructs)."""
    dev = triangles.vertices.device
    x = _vertices_np(triangles)
    q, low, scale = quantize_xyz(x, bits)
    uq, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    n_dup = int(len(q) - len(uq))
    if n_dup:
        rep = first[inv.reshape(-1)]
        f = rep[_faces_np(triangles)]
        degenerate = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
        f = f[~degenerate]
        f = np.unique(np.sort(f, axis=1), axis=0) if len(f) else f
        _set_faces(triangles, f)
        keep = np.zeros(len(q), bool)
        keep[first] = True
        triangles._prune_vertices(torch.from_numpy(keep).to(dev))  # also drops now-unused vertices
        x = _vertices_np(triangles)
        q, low, scale = quantize_xyz(x, bits)
    xq = dequantize_xyz(q, low, scale)
    q2, low2, scale2 = quantize_xyz(xq, bits)
    if not np.array_equal(q2, q):
        raise RuntimeError(f'delivery quantiser is not idempotent at {bits} bits ({int((q2 != q).any(1).sum())} vertices move)')
    triangles.vertices.data = torch.from_numpy(xq).to(dev)
    return q2, low2, scale2, n_dup


def assert_on_grid(triangles, bits):
    """Positions must be exactly the dequantised grid points; returns the grid tuple the decoder will reconstruct."""
    x = _vertices_np(triangles)
    q, low, scale = quantize_xyz(x, bits)
    xq = dequantize_xyz(q, low, scale)
    drift = float(np.abs(xq - x).max()) if len(x) else 0.0
    if drift != 0.0:
        raise RuntimeError(f'vertices are not on the q{bits} grid (max drift {drift})')
    return q, low, scale


def rebuild_event(triangles, rd, log, t0, stage, iteration, restriction_factory=None, clean_full_fn=None, clean_faces_fn=None, post_clean=True):
    """One v5 event (stage in STAGES).  rd keys: xyz (qN), rebuild_jitter, rebuild_max_rounds.  Returns the stats dict
    (also written to log).  After 'final' / 'snap_only' the caller freezes XYZ and triangles._v4_grid holds (q, low, scale)."""
    if stage not in STAGES:
        raise ValueError(stage)
    if restriction_factory is None:
        restriction_factory = FixedSoupRestriction
    bits = int(str(rd['xyz'])[1:])
    jitter = float(rd['rebuild_jitter'])
    max_rounds = int(rd['rebuild_max_rounds'])
    dev = triangles.vertices.device
    V0, T0 = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
    if clean_full_fn is not None:
        clean_full_fn()
    V1, T1 = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
    stats = dict(event='rebuild', stage=stage, iteration=int(iteration), version='v5', module=VERSION, canon=CANON_VERSION, xyz=f'q{bits}', jitter=jitter,
                 vertices_before=V0, faces_before=T0, vertices_after_clean=V1, faces_after_clean=T1, duplicates_merged=0, rounds=[])
    if stage in ('final', 'snap_only'):
        _, _, _, n_dup = snap_to_grid(triangles, bits)
        stats['duplicates_merged'] = n_dup
        stats['vertices_after_snap'], stats['faces_after_snap'] = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
    if stage != 'snap_only':
        soup_v, soup_f = _vertices_np(triangles), _faces_np(triangles)
        soup_keys = _face_keys(soup_f)
        restrict = restriction_factory(soup_v, soup_f)        # built ONCE: the same soup for every round
        stats['soup_vertices'], stats['soup_faces'] = int(len(soup_v)), int(len(soup_f))
        for r in range(max_rounds + 1):
            if stage == 'final':
                x = _vertices_np(triangles)
                q, low, scale = quantize_xyz(x, bits)
                xq = dequantize_xyz(q, low, scale)
                q2, low2, scale2 = quantize_xyz(xq, bits)
                if not np.array_equal(q2, q):
                    raise RuntimeError('delivery quantiser is not idempotent after vertex removal')
                triangles.vertices.data = torch.from_numpy(xq).to(dev)   # re-snapped (low/scale may change when extremes are removed)
                perm, P32 = canonical_points(q2, low2, scale2, jitter)
                new_faces = perm[restrict(P32)]
            else:
                new_faces = restrict(_vertices_np(triangles))
            faces_in = int(triangles._triangle_indices.shape[0])
            _set_faces(triangles, new_faces)
            used = _used_mask(triangles)
            n_unused = int((~used).sum())
            row = dict(round=r, vertices=int(len(used)), faces_in=faces_in, faces_out=int(len(new_faces)), unused=n_unused)
            if r == 0:
                new_keys = _face_keys(new_faces)
                common = _n_common(soup_keys, new_keys)
                row['soup_faces_retained'] = round(common / max(1, len(soup_keys)), 4)
                row['output_faces_new'] = round(1.0 - common / max(1, len(new_keys)), 4)
            stats['rounds'].append(row)
            extra = f" | soup retained {row['soup_faces_retained']:.3f}, new faces {row['output_faces_new']:.3f}" if r == 0 else ''
            print(f'\n[RD rebuild {stage} round {r}] V {len(used)} faces {faces_in} -> {len(new_faces)} | unused vertices {n_unused}{extra}')
            if n_unused == 0 or r == max_rounds:
                break
            triangles._prune_vertices(torch.from_numpy(used).to(dev))
        stats['restriction_calls'] = int(getattr(restrict, 'calls', len(stats['rounds'])))
        stats['unused_kept'] = stats['rounds'][-1]['unused']
        if post_clean and clean_faces_fn is not None:
            Tb = int(triangles._triangle_indices.shape[0])
            clean_faces_fn()
            stats['faces_removed_by_post_clean'] = Tb - int(triangles._triangle_indices.shape[0])
            print(f"[RD rebuild {stage}] post clean removed {stats['faces_removed_by_post_clean']} faces")
        if stage == 'early':
            used = _used_mask(triangles)
            if not used.all():
                triangles._prune_vertices(torch.from_numpy(used).to(dev))
    if stage in ('final', 'snap_only'):
        q, low, scale = assert_on_grid(triangles, bits)
        triangles._v4_grid = (q.copy(), np.asarray(low).copy(), np.asarray(scale).copy())
        stats['unused_final_event'] = int((~_used_mask(triangles)).sum())
    stats['vertices'], stats['faces'] = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
    stats['elapsed_s'] = round(time.time() - t0)
    if log is not None:
        log.write(**stats)
    return stats
