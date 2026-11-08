"""make_v9d.py: build train_rd_adapt_v9d.py from train_rd_adapt_v9c.py by exact-count replacements (v9c untouched).
Code diff: phase-B collapse candidates priced on the DELIVERED set (rd_prune_v2.select_collapses with the probe's kept masks; edge_scores the same),
bytes-per-collapse prior = b_v + 2 b_f, version strings."""
import sys
src_path, dst_path = sys.argv[1], sys.argv[2]
s = open(src_path).read(); n_rep = 0


def rep(old, new, count=1):
    global s, n_rep
    c = s.count(old)
    assert c == count, f'anchor found {c}x (expected {count}): {old[:90]!r}'
    s = s.replace(old, new); n_rep += 1


HEADER = '''"""train_rd_adapt_v9d.py (adapt-v9d, 2026-11-08 = adapt-v9c with the phase-B collapse PRICED ON THE DELIVERED SET; everything else verbatim v9c).
What the v9c test (node-batch15: room 5, counter 5, kitchen 10, counter 10, kitchen 20 + lego 2.5) showed: the slope clamp works (the estimate never
read below the probe, kitchen 10's landing estimate rose 9.4 -> 11.3 MB as the kept set un-stacked, final 11.15), phase B ran 15-24 of 30 rounds,
yet the rows still ended 1.03-1.12 x target, because the rounds freed almost nothing: 4-8k pairs per round and 0.002-0.03 MB each (kitchen 10:
109 724 pairs collapsed, 0.5 MB freed of the 1.3 MB excess; both-kept share 0.35-0.57 on Mip-360, bytes per pair 0.6-2.4 against the 6.3 prior).
The cheapest dD/dR edges are the zero-residual ones: the triangles nobody renders, which the final cleaning drops anyway and the probe already
excludes.  The lambda-prune was trimming geometry that costs no bytes.  v9d:
 1. select_collapses (rd_prune_v2.py, new file; rd_prune.py untouched) takes the probe's kept-vertex / kept-face masks of this round and prices
    dR_uv = b_attr * [u and v both kept] + b_face * (kept faces on the edge); edges with dR_uv = 0 are not candidates.  edge_scores (n_lam, the
    rank threshold, the shadow price) uses the same price.  An edge that frees no delivered bytes has no price and no reason to collapse under
    a rate objective: this is 'the same price in reverse' applied to the file, not to the soup.
 2. Bytes-per-collapse prior before the first measured round: b_v + 2 b_f (both endpoints kept by construction, two kept faces), instead of the
    kept-share-weighted guess.  The exact kept-aware accounting of v7 is unchanged and now matches the price.
 3. version strings 'adapt-v9d'; the config event carries prune_version; the prune line prints the priced edge count.
Expected: the same 15-30 rounds now remove 5-7 B per pair, so the indoor rows that trimmed 5-10 % of V for 0.5 MB reach the target with margin;
Blender (both-kept share already 0.7-0.8) changes little.

adapt-v9c header follows.
'''
rep('"""train_rd_adapt_v9c.py (adapt-v9c, 2026-11-08 = ', HEADER + 'train_rd_adapt_v9c.py (adapt-v9c, 2026-11-08 = ')
rep("from rd_prune import select_collapses, collapse_edges, edge_table  # noqa: E402\n", "from rd_prune_v2 import select_collapses, collapse_edges, edge_table, PRUNE_VERSION  # noqa: E402  (v9d: kept-priced collapse candidates)\n")
rep('''def edge_scores(vertices, faces, face_err, b_attr, b_face):
    """Per-edge collapse score dD/dR with the formula of rd_prune.select_collapses (float64).  Returns (edges, cnt, score)."""
    f = faces.long(); V = vertices.shape[0]
    edges, inv, cnt = edge_table(f); err = face_err.double()
    D_v = torch.zeros(V, dtype=torch.float64, device=vertices.device)
    for k in range(3): D_v.index_add_(0, f[:, k], err)
    D_uv = torch.zeros(edges.shape[0], dtype=torch.float64, device=vertices.device); D_uv.index_add_(0, inv, err.repeat(3))
    dR = float(b_attr) + cnt.double() * float(b_face)
    score = (D_v[edges[:, 0]] + D_v[edges[:, 1]] - D_uv) / dR
    return edges, cnt, score
''', '''def edge_scores(vertices, faces, face_err, b_attr, b_face, kept_v=None, kept_f=None):
    """Per-edge collapse score dD/dR with the formula of rd_prune_v2.select_collapses (float64).  v9d: with the probe's kept masks the price counts
    only delivered bytes (b_attr if both endpoints are kept + b_face per kept face on the edge) and unpriced edges score +inf.  Returns (edges, cnt, score)."""
    f = faces.long(); V = vertices.shape[0]
    edges, inv, cnt = edge_table(f); err = face_err.double()
    D_v = torch.zeros(V, dtype=torch.float64, device=vertices.device)
    for k in range(3): D_v.index_add_(0, f[:, k], err)
    D_uv = torch.zeros(edges.shape[0], dtype=torch.float64, device=vertices.device); D_uv.index_add_(0, inv, err.repeat(3))
    dD = D_v[edges[:, 0]] + D_v[edges[:, 1]] - D_uv
    if kept_v is not None and kept_f is not None:
        kv = kept_v.to(vertices.device).bool(); kf = kept_f.to(vertices.device).bool()
        cnt_kept = torch.zeros(edges.shape[0], dtype=torch.float64, device=vertices.device); cnt_kept.index_add_(0, inv, kf.double().repeat(3))
        dR = float(b_attr) * (kv[edges[:, 0]] & kv[edges[:, 1]]).double() + cnt_kept * float(b_face); priced = dR > 0
        score = torch.where(priced, dD / torch.where(priced, dR, torch.ones_like(dR)), torch.full_like(dD, float('inf')))
    else:
        score = dD / (float(b_attr) + cnt.double() * float(b_face))
    return edges, cnt, score
''')
rep("                else: b_coll = size.b_v * (probe['vertices'] / max(probe['vertices_model'], 1)) ** 2 + 2.0 * size.b_f * float(probe['kept_frac'])  # prior: both endpoints kept, two kept faces\n",
    "                else: b_coll = size.b_v + 2.0 * size.b_f  # v9d prior: candidates are priced on the delivered set, so a collapse frees one kept vertex and (typically) two kept faces\n")
rep("                edges, cnt, score = edge_scores(triangles.vertices, triangles._triangle_indices, dD, size.b_v, size.b_f)\n",
    "                edges, cnt, score = edge_scores(triangles.vertices, triangles._triangle_indices, dD, size.b_v, size.b_f, kept_v, kept_f)  # v9d: priced on the probe's delivered set\n")
rep("                    pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac)\n",
    "                    pairs, ps = select_collapses(triangles.vertices, triangles._triangle_indices, dD, thr / rd['prune_mu'], rd['prune_mu'], b_v_full, size.b_f, frac, kept_v=kept_v, kept_f=kept_f)  # v9d\n")
rep("lam-admitted {n_lam} edges, cap {ps['max_pairs']} -> {ps['n_pairs']} collapsed", "lam-admitted {n_lam} of {ps['n_priced']} priced edges, cap {ps['max_pairs']} -> {ps['n_pairs']} collapsed")
rep("    log.write(event='config', version='adapt-v9c', ", "    log.write(event='config', version='adapt-v9d', prune_version=PRUNE_VERSION, ")
rep("version='adapt-v9c'", "version='adapt-v9d'", 1)
rep("| adapt-v9c: target {rd['target_mb']} MB, final probe", "| adapt-v9d: target {rd['target_mb']} MB, final probe")
rep("'| RD growth+prune adapt-v9c target_mb'", "'| RD growth+prune adapt-v9d target_mb'")
assert "version='adapt-v9c'" not in s and s.count("version='adapt-v9d'") == 2 and 'rd_prune_v2' in s and 'kept_v=kept_v, kept_f=kept_f' in s
open(dst_path, 'w').write(s)
print(f'{n_rep} replacements -> {dst_path}: {len(s.splitlines())} lines')
