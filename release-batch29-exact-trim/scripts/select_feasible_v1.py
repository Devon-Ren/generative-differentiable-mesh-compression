# select_feasible_v1 (2026-11-12, batch29 v2): the text between the two marker lines is inserted verbatim into the training script.
# >>> select_feasible
@torch.no_grad()
def _match_cheapest(e, s, V, passes=24):
    """Vertex-disjoint subset of the candidate edges e [C,2] (scores s [C], distinct): repeatedly take the edges that are the cheapest at both endpoints."""
    dev = e.device; C = e.shape[0]; picked = torch.zeros(C, dtype=torch.bool, device=dev); avail = torch.ones(V, dtype=torch.bool, device=dev)
    inf = torch.full_like(s, float('inf'))
    for _ in range(passes):
        live = ~picked & avail[e[:, 0]] & avail[e[:, 1]]
        if not bool(live.any()): break
        sl = torch.where(live, s, inf)
        vmin = torch.full((V,), float('inf'), dtype=s.dtype, device=dev)
        vmin.scatter_reduce_(0, e[:, 0], sl, reduce='amin'); vmin.scatter_reduce_(0, e[:, 1], sl, reduce='amin')
        idx = torch.nonzero(live & (sl == vmin[e[:, 0]]) & (sl == vmin[e[:, 1]]), as_tuple=True)[0]
        if idx.numel() == 0: break
        use = torch.zeros(V, dtype=torch.long, device=dev)
        use.index_add_(0, e[idx, 0], torch.ones_like(idx)); use.index_add_(0, e[idx, 1], torch.ones_like(idx))
        idx = idx[(use[e[idx, 0]] == 1) & (use[e[idx, 1]] == 1)]
        if idx.numel() == 0: break
        picked[idx] = True; avail[e[idx].reshape(-1)] = False
    return picked


@torch.no_grad()
def select_feasible(pos, faces, face_err, kept_v, kept_f, b_attr, b_face, want_bytes, max_pairs, max_iters=40):
    """Collapses that free `want_bytes` of the delivered file, cheapest error per freed byte first, as a set that is vertex-disjoint and
    free of fold-overs when applied together.  Freed bytes of edge (u,v): b_attr if both endpoints are in the delivered set + b_face per
    delivered face on the edge; edges that free nothing are not candidates.  The set is built in batches: each batch is matched, tested
    together with everything accepted so far, the pairs that flip a face are banned, the rest is accepted.
    Returns (edges [E,2], sel [P] edge indices, stats)."""
    dev = pos.device; f = faces.long(); V = pos.shape[0]
    edges, inv, cnt = edge_table(f); E = edges.shape[0]; err = face_err.double().to(dev)
    D_v = torch.zeros(V, dtype=torch.float64, device=dev)
    for k in range(3): D_v.index_add_(0, f[:, k], err)
    D_uv = torch.zeros(E, dtype=torch.float64, device=dev); D_uv.index_add_(0, inv, err.repeat(3))
    kv = kept_v.to(dev).bool(); kf = kept_f.to(dev).bool()
    cnt_k = torch.zeros(E, dtype=torch.float64, device=dev); cnt_k.index_add_(0, inv, kf.double().repeat(3))
    dR = float(b_attr) * (kv[edges[:, 0]] & kv[edges[:, 1]]).double() + float(b_face) * cnt_k
    priced = dR > 0
    score = torch.where(priced, (D_v[edges[:, 0]] + D_v[edges[:, 1]] - D_uv) / torch.where(priced, dR, torch.ones_like(dR)), torch.full_like(dR, float('inf')))
    gen = torch.Generator(device=dev); gen.manual_seed(4321 + E)
    fin = score[priced]; scale = float(fin.mean()) if fin.numel() else 1.0
    s = score + torch.rand(E, dtype=torch.float64, device=dev, generator=gen) * 1e-9 * max(scale, 1e-30)  # distinct scores (ties: faces with no error)
    order = torch.argsort(s); order = order[priced[order]]
    n0 = _face_normals(pos, f)
    avail = torch.ones(V, dtype=torch.bool, device=dev); banned = torch.zeros(E, dtype=torch.bool, device=dev)
    pos2 = pos.clone(); new_id = torch.arange(V, device=dev)
    acc = []; n_acc = 0; acc_bytes = 0.0; n_banned = 0; it = 0; n_dropped_batches = 0
    while it < max_iters and acc_bytes < want_bytes and n_acc < max_pairs:
        it += 1
        cand = order[~banned[order] & avail[edges[order, 0]] & avail[edges[order, 1]]]
        if cand.numel() == 0: break
        mean_b = (acc_bytes / n_acc) if n_acc else float(dR[cand[:2000]].mean())
        need = max(1, min(max_pairs - n_acc, int(0.7 * (want_bytes - acc_bytes) / max(mean_b, 1e-6)) + 1))  # 70 % of what is left per batch: the last batches are small, so the total ends within a few pairs of the wanted bytes
        cand = cand[:4 * need + 256]
        new = cand[_match_cheapest(edges[cand], s[cand], V)][:need]
        converged = False
        for _ in range(10):
            if new.numel() == 0: break
            u, v = edges[new, 0], edges[new, 1]
            p2 = pos2.clone(); mid = 0.5 * (pos[u] + pos[v]); p2[u] = mid; p2[v] = mid
            nid = new_id.clone(); nid[v] = u; f2 = nid[f]
            degenerate = (f2[:, 0] == f2[:, 1]) | (f2[:, 1] == f2[:, 2]) | (f2[:, 0] == f2[:, 2])
            flipped = ((n0 * _face_normals(p2, f)).sum(1) < 0) & ~degenerate
            if not bool(flipped.any()): converged = True; break
            pair_of_v = torch.full((V,), -1, dtype=torch.long, device=dev); kk = torch.arange(new.numel(), device=dev); pair_of_v[u] = kk; pair_of_v[v] = kk
            bad = pair_of_v[f[flipped].reshape(-1)]; bad = torch.unique(bad[bad >= 0])
            if bad.numel() == 0: break
            keep = torch.ones(new.numel(), dtype=torch.bool, device=dev); keep[bad] = False
            banned[new[~keep]] = True; n_banned += int(bad.numel()); new = new[keep]
        if not converged or new.numel() == 0:
            if new.numel(): banned[new] = True; n_dropped_batches += 1
            continue
        u, v = edges[new, 0], edges[new, 1]; mid = 0.5 * (pos[u] + pos[v]); pos2[u] = mid; pos2[v] = mid; new_id[v] = u
        avail[u] = False; avail[v] = False
        acc.append(new); n_acc += int(new.numel()); acc_bytes += float(dR[new].sum())
    sel = torch.cat(acc) if acc else torch.zeros(0, dtype=torch.long, device=dev)
    stats = dict(n_pairs=int(sel.numel()), planned_bytes=acc_bytes, want_bytes=float(want_bytes), n_banned=n_banned, iters=it, n_priced=int(priced.sum()), n_edges=E, dropped_batches=n_dropped_batches,
                 score_max=(float(score[sel].max()) if sel.numel() else None))
    return edges, sel, stats
# <<< select_feasible
