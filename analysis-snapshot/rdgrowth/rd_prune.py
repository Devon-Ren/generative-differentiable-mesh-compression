"""Rate-distortion controlled edge-collapse pruning for MeshSplatting (v3, 2026-10-19).

Pure torch, no rasterizer import, so it is unit-testable on CPU with a mock TriangleModel.

Idea.  After the restricted Delaunay rebuild upstream never removes a vertex again (only the final importance<=0.5
cleaning of triangles).  Here, at a few iterations of phase B (before supersampling), we run the split rule backwards:
collapse edge (u,v) into its midpoint iff

        dD_uv / dR_uv  <  lam * mu,        dD_uv = residual mass of the faces touching u or v   (same units as the split
                                            side: per-view averaged, per-megapixel pixel-L1 attributed through rend_ids)
                                            dR_uv = B_attr + n_faces(uv) * B_face   (bytes the collapse saves in the file:
                                            one vertex's attributes + the faces that disappear; interior edge 2 faces,
                                            boundary 1, non-manifold 3+ -> the price of a vertex varies with its topology)

Collapses are chosen as an independent set (no vertex takes part in two collapses in one round) by repeated locally-minimal
edge matching, cheapest score first, capped at a fraction of the vertices per round (prune_max_frac: the pruning twin of the
upstream growth limiter: lam decides which and when to stop, the cap only limits speed).  Collapses that would flip the
orientation of a surviving face are rejected (foldover guard).  The merged vertex keeps u's slot: parameters, vertex
opacity (averaged in opacity space, like the split) and Adam moments are averaged into u, v is deleted with the upstream
_prune_vertices path, degenerate and duplicate faces are dropped.
"""
import torch


def _face_normals(pos, faces):
    t = pos[faces]
    return torch.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0], dim=1)


@torch.no_grad()
def edge_table(faces):
    """Unique undirected edges of an int face tensor [T,3] -> (edges [E,2] long sorted, inverse [3T] long, count [E] long)."""
    f = faces.long()
    e = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]], dim=0)
    e, _ = torch.sort(e, dim=1)
    edges, inv, cnt = torch.unique(e, dim=0, return_inverse=True, return_counts=True)
    return edges, inv, cnt


@torch.no_grad()
def foldover_filter(pos, f, edges, sel):
    """Reject selected edges whose midpoint collapse flips the normal of a surviving face.  Returns (sel kept, n rejected)."""
    if sel.numel() == 0:
        return sel, 0
    dev = pos.device
    V = pos.shape[0]
    u, v = edges[sel, 0], edges[sel, 1]
    pos2 = pos.clone()
    mid = 0.5 * (pos2[u] + pos2[v])
    pos2[u] = mid
    pos2[v] = mid
    new_id = torch.arange(V, device=dev)
    new_id[v] = u
    f2 = new_id[f]
    degenerate = (f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2]) | (f2[:, 0] == f2[:, 2])
    n0 = _face_normals(pos, f)
    n1 = _face_normals(pos2, f)
    flipped = ((n0 * n1).sum(1) < 0) & ~degenerate
    if not bool(flipped.any()):
        return sel, 0
    pair_of_v = torch.full((V,), -1, dtype=torch.long, device=dev)
    k = torch.arange(sel.numel(), device=dev)
    pair_of_v[u] = k
    pair_of_v[v] = k
    bad = pair_of_v[f[flipped].reshape(-1)]
    bad = torch.unique(bad[bad >= 0])
    keep = torch.ones(sel.numel(), dtype=torch.bool, device=dev)
    keep[bad] = False
    return sel[keep], int((~keep).sum())


@torch.no_grad()
def select_collapses(vertices, faces, face_err, lam, mu, bytes_per_vertex, bytes_per_face, max_frac, max_passes=64):
    """Pick an independent set of edges to collapse.  Returns (pairs [P,2] long (u,v), stats dict).

    face_err: [T] distortion per face over the window (acc.distortion()).  Score per edge (float64):
      s_uv = (D_u + D_v - D_uv) / dR_uv, D_v = sum of face_err over faces touching v, D_uv = over faces containing the edge.
    Candidates: s_uv < lam * mu.  Cap: at most max_frac * V pairs (cheapest first).
    """
    dev = vertices.device
    V = vertices.shape[0]
    f = faces.long()
    T = f.shape[0]
    edges, inv, cnt = edge_table(f)
    E = edges.shape[0]
    err = face_err.double()
    D_v = torch.zeros(V, dtype=torch.float64, device=dev)
    for k in range(3):
        D_v.index_add_(0, f[:, k], err)
    D_uv = torch.zeros(E, dtype=torch.float64, device=dev)
    D_uv.index_add_(0, inv, err.repeat(3))
    b_attr = float(bytes_per_vertex) - 2.0 * float(bytes_per_face)  # attributes of one vertex (XYZ + DC + index), topology priced per face
    if b_attr <= 0:
        raise ValueError(f'bytes_per_vertex {bytes_per_vertex} must exceed 2*bytes_per_face {bytes_per_face}')
    dR = b_attr + cnt.double() * float(bytes_per_face)
    dD = D_v[edges[:, 0]] + D_v[edges[:, 1]] - D_uv
    score = dD / dR
    thr = float(lam) * float(mu)
    cand = score < thr
    n_cand = int(cand.sum())
    max_pairs = int(max_frac * V)
    # tie-break (many exact zeros where nothing was rendered): tiny seeded random jitter, float64 so it is not lost.
    # Random rather than index-ordered: index order makes the locally-minimal matching crawl one row per pass on grids.
    gen = torch.Generator(device=dev); gen.manual_seed(1234 + E)
    jit = torch.rand(E, dtype=torch.float64, device=dev, generator=gen) * 1e-9 * max(thr, 1e-30)
    s = torch.where(cand, score + jit, torch.full_like(score, float('inf')))
    avail = torch.ones(V, dtype=torch.bool, device=dev)
    picked = []
    n_picked = 0
    passes = 0
    for _ in range(max_passes):
        passes += 1
        live = cand & avail[edges[:, 0]] & avail[edges[:, 1]]
        if not bool(live.any()) or n_picked >= max_pairs:
            break
        sl = torch.where(live, s, torch.full_like(s, float('inf')))
        vmin = torch.full((V,), float('inf'), dtype=torch.float64, device=dev)
        vmin.scatter_reduce_(0, edges[:, 0], sl, reduce='amin')
        vmin.scatter_reduce_(0, edges[:, 1], sl, reduce='amin')
        pick = live & (sl == vmin[edges[:, 0]]) & (sl == vmin[edges[:, 1]])
        idx = torch.nonzero(pick, as_tuple=True)[0]
        if idx.numel() == 0:
            break
        use = torch.zeros(V, dtype=torch.long, device=dev)
        use.index_add_(0, edges[idx, 0], torch.ones_like(idx))
        use.index_add_(0, edges[idx, 1], torch.ones_like(idx))
        ok = (use[edges[idx, 0]] == 1) & (use[edges[idx, 1]] == 1)
        idx = idx[ok]
        if idx.numel() == 0:
            break
        picked.append(idx)
        n_picked += int(idx.numel())
        avail[edges[idx].reshape(-1)] = False
    sel = torch.cat(picked) if picked else torch.zeros(0, dtype=torch.long, device=dev)
    if sel.numel() > max_pairs:
        order = torch.argsort(s[sel])
        sel = sel[order[:max_pairs]]
    capped = bool(n_picked > max_pairs)
    sel, n_flip = foldover_filter(vertices.detach(), f, edges, sel)
    pairs = edges[sel]
    def share(idx_mask_or_idx):
        c = cnt[idx_mask_or_idx]
        n = max(int(c.numel()), 1)
        return dict(f1=round(float((c == 1).sum()) / n, 4), f2=round(float((c == 2).sum()) / n, 4), f3p=round(float((c >= 3).sum()) / n, 4))
    fin = score[torch.isfinite(score)]
    if fin.numel() > 8_000_000:  # torch.quantile size limit
        fin = fin[torch.randperm(fin.numel(), device=dev)[:8_000_000]]
    q = torch.quantile(fin, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, device=dev)).tolist() if fin.numel() > 1000 else [float('nan')] * 3
    stats = dict(vertices=V, faces=T, edges=E, n_cand=n_cand, cand_frac=round(n_cand / max(E, 1), 4), n_pass_pairs=n_picked,
                 n_pairs=int(pairs.shape[0]), max_pairs=max_pairs, capped=capped, n_flip_rejected=n_flip, passes=passes,
                 thr=thr, score_q10=q[0], score_q50=q[1], score_q90=q[2], dR_mean_sel=(float(dR[sel].mean()) if sel.numel() else None),
                 faces_removed_share_sel=share(sel), faces_removed_share_cand=share(cand), faces_removed_share_all=share(torch.ones(E, dtype=torch.bool, device=dev)))
    return pairs, stats


@torch.no_grad()
def collapse_edges(triangles, pairs):
    """Collapse each (u,v) pair into u (midpoint position, averaged attributes and Adam moments), delete v.

    Pairs must be vertex-disjoint (select_collapses guarantees it).  Uses only upstream members: vertices, vertex_weight,
    _features_dc, _features_rest, optimizer, opacity_activation / inverse_opacity_activation, opacity_floor, eps,
    _triangle_indices, image_size, importance_score, pixel_count, _prune_vertices.  Returns stats dict.
    """
    V = triangles.vertices.shape[0]
    T0 = triangles._triangle_indices.shape[0]
    if pairs.numel() == 0:
        return dict(vertices_before=V, vertices_after=V, faces_before=T0, faces_after=T0, faces_degenerate=0, faces_duplicate=0)
    dev = triangles.vertices.device
    u = pairs[:, 0].to(dev)
    v = pairs[:, 1].to(dev)
    if int(torch.unique(pairs.reshape(-1)).numel()) != 2 * pairs.shape[0]:
        raise RuntimeError('collapse pairs are not vertex-disjoint')
    # 1) merge parameters into slot u
    for name in ('vertices', '_features_dc', '_features_rest'):
        p = getattr(triangles, name)
        p.data[u] = 0.5 * (p.data[u] + p.data[v])
    w = triangles.vertex_weight
    op = triangles.opacity_activation(w.data)
    avg = torch.clamp(0.5 * (op[u] + op[v]), triangles.opacity_floor + triangles.eps, 1 - triangles.eps)
    w.data[u] = triangles.inverse_opacity_activation(avg)
    # 2) Adam moments of the merged vertex = average of the two
    for group in triangles.optimizer.param_groups:
        if group['name'] in ('vertices', 'vertex_weight', 'f_dc', 'f_rest'):
            st = triangles.optimizer.state.get(group['params'][0], None)
            if st is not None:
                for key in ('exp_avg', 'exp_avg_sq'):
                    m = st[key]
                    m[u] = 0.5 * (m[u] + m[v])
    # 3) faces: v -> u, drop degenerate and duplicate faces, keep per-face trackers aligned
    f = triangles._triangle_indices.long()
    new_id = torch.arange(V, device=dev)
    new_id[v] = u
    f2 = new_id[f]
    degenerate = (f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2]) | (f2[:, 0] == f2[:, 2])
    keep = ~degenerate
    key, _ = torch.sort(f2, dim=1)
    _, first = torch.unique(key, dim=0, return_inverse=True)
    # first occurrence of each sorted triple survives (orientation of that face kept)
    order = torch.arange(f2.shape[0], device=dev)
    firstpos = torch.full((int(first.max()) + 1,), f2.shape[0], dtype=torch.long, device=dev)
    firstpos.scatter_reduce_(0, first, order, reduce='amin')
    dup = firstpos[first] != order
    n_dup = int((dup & keep).sum())
    keep = keep & ~dup
    triangles._triangle_indices = f2[keep].to(torch.int32).contiguous()
    for name in ('image_size', 'importance_score', 'pixel_count'):
        t = getattr(triangles, name)
        if isinstance(t, torch.Tensor) and t.numel() == T0:
            setattr(triangles, name, t[keep])
    # 4) delete the v slots through the upstream path (prunes optimizer state and reindexes faces)
    vmask = torch.ones(V, dtype=torch.bool, device=dev)
    vmask[v] = False
    triangles._prune_vertices(vmask)
    V1 = int(triangles.vertices.shape[0])
    return dict(vertices_before=V, vertices_after=V1, pairs=int(pairs.shape[0]), vertices_isolated_removed=V - int(pairs.shape[0]) - V1,
                faces_before=T0, faces_after=int(triangles._triangle_indices.shape[0]), faces_degenerate=int(degenerate.sum()), faces_duplicate=n_dup)
