"""rd_rebuild_v1.py (2026-10-26): the v4 late rebuild used by train_rd_v4.py, kept importable without the upstream
(CPU unit test: python test_rd_rebuild_cpu.py).  See train_rd_v4.py for the protocol."""
import time
import numpy as np
import torch
from delaunay_canon_v1 import quantize_xyz, dequantize_xyz, canonical_points, VERSION as CANON_VERSION

VERSION = 'rd_rebuild_v1 2026-10-26'

def _reset_face_trackers(triangles):
    T = triangles._triangle_indices.shape[0]
    triangles.image_size = torch.zeros((T,), dtype=torch.float, device=triangles.vertices.device)
    triangles.importance_score = torch.zeros((T,), dtype=torch.float, device=triangles.vertices.device)
    triangles.pixel_count = torch.zeros((T,), dtype=torch.int, device=triangles.vertices.device)


def _set_faces(triangles, faces_np):
    triangles._triangle_indices = torch.as_tensor(np.asarray(faces_np, dtype=np.int64), device=triangles.vertices.device).contiguous().to(torch.int32)
    _reset_face_trackers(triangles)


def late_rebuild(triangles, rd, log, t0, rdel_run=None, clean_fn=None):
    """v4 event: clean -> snap XYZ to the delivery grid -> merge duplicate grid points -> canonical restricted Delaunay
    until every vertex is used -> freeze XYZ (the caller sets the lr).  Returns the stats dict (also logged)."""
    if rdel_run is None:
        import rdel  # effrdel, the module upstream uses for run_restricted_delaunay
        rdel_run = lambda P, f: rdel.run(P, f, verbose=False, orient=False)  # noqa: E731
    bits = int(str(rd['xyz'])[1:]); jitter = float(rd['rebuild_jitter']); max_rounds = int(rd['rebuild_max_rounds'])
    dev = triangles.vertices.device
    V0, T0 = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
    if clean_fn is not None:
        clean_fn()
    V1, T1 = int(triangles.vertices.shape[0]), int(triangles._triangle_indices.shape[0])
    rounds = []
    q = low = scale = None
    for r in range(max_rounds + 1):
        x = triangles.vertices.detach().cpu().numpy().astype(np.float32)
        q, low, scale = quantize_xyz(x, bits)
        uq, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        n_dup = int(len(q) - len(uq))
        if n_dup:
            rep = first[inv.reshape(-1)]                      # representative (first occurrence) of each vertex's grid cell
            f = rep[triangles._triangle_indices.long().cpu().numpy()]
            degenerate = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
            f = f[~degenerate]
            f = np.unique(np.sort(f, axis=1), axis=0) if len(f) else f   # drop duplicate faces (orientation is not used)
            _set_faces(triangles, f)
            keep = np.zeros(len(q), bool); keep[first] = True
            triangles._prune_vertices(torch.from_numpy(keep).to(dev))    # also drops now-unused vertices
            x = triangles.vertices.detach().cpu().numpy().astype(np.float32)
            q, low, scale = quantize_xyz(x, bits)
        xq = dequantize_xyz(q, low, scale)
        q2, low2, scale2 = quantize_xyz(xq, bits)
        if not np.array_equal(q2, q):
            raise RuntimeError(f'delivery quantiser is not idempotent at {bits} bits ({int((q2 != q).any(1).sum())} vertices move)')
        q, low, scale = q2, low2, scale2                                  # exactly what the decoder will see
        triangles.vertices.data = torch.from_numpy(xq).to(dev)
        perm, P32 = canonical_points(q, low, scale, jitter)
        inv_perm = np.empty_like(perm); inv_perm[perm] = np.arange(len(perm))
        faces = triangles._triangle_indices.long().cpu().numpy()
        hit = np.asarray(rdel_run(P32, inv_perm[faces].astype(np.int32)), dtype=np.int64).reshape(-1, 3)
        new_faces = perm[hit]
        _set_faces(triangles, new_faces)
        V = int(triangles.vertices.shape[0])
        used = np.zeros(V, bool); used[new_faces.reshape(-1)] = True
        n_unused = int((~used).sum())
        rounds.append(dict(round=r, vertices=V, faces_in=int(len(faces)), faces_out=int(len(new_faces)), duplicates_merged=n_dup, unused=n_unused))
        print(f'\n[RD rebuild round {r}] V {V} faces {len(faces)} -> {len(new_faces)} | duplicates merged {n_dup} | unused vertices {n_unused}')
        if n_unused == 0 or r == max_rounds:
            break
        triangles._prune_vertices(torch.from_numpy(used).to(dev))
    stats = dict(event='rebuild', version='v4', canon=CANON_VERSION, xyz=f'q{bits}', jitter=jitter, vertices_before=V0, faces_before=T0,
                 vertices_after_clean=V1, faces_after_clean=T1, rounds=rounds, vertices=int(triangles.vertices.shape[0]),
                 faces=int(triangles._triangle_indices.shape[0]), unused_kept=rounds[-1]['unused'], elapsed_s=round(time.time() - t0))
    triangles._v4_grid = (q.copy(), np.asarray(low).copy(), np.asarray(scale).copy())
    if log is not None:
        log.write(**stats)
    return stats

