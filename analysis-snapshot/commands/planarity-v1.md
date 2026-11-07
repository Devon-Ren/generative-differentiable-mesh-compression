# planarity-v1: 探针 P1 —— 训练出的 mesh 有多少是平的（2026-10-30）

**问题**：顶点有多少是在为颜色/三角化打工而不是为形状打工？一块平面 + 一张纹理只需要轮廓上的顶点。这是 plan/meshsplat-plus-2026-10-29.md 第 8 节的 P1，纯 CPU，用现有 checkpoint。
**方法**（`analysis/diag/planarity_v1.py`，numpy/scipy）：相邻两面若二面角 ≤ θ 且各自的对顶点到对方平面 ≤ ε，则"局部共面"；共面图的连通分量 = 候选平面区域；每个 ≥ 10 面的区域做面积加权 PCA 平面拟合，顶点到平面距离的 95 分位 ≤ ε 才算**验证平面**（球面这类靠小二面角串起来的会被这一步拒掉；selftest 里有盒子 + 球 + 随机 soup）。ε 取场景鲁棒对角线（顶点 2–98 分位包围盒）的 0.1 / 0.25 / 0.5 %，θ 取 5° / 10°；同时打印 q14 网格步长和中位边长作参照。
**报告的数**（每个 (ε, θ) 组合）：验证平面区域（≥ 10 / 100 / 1000 面）里的面占比、面积占比、顶点占比、**内部顶点占比**（所有相邻面都在同一平面区域里的顶点 —— 纹理能替掉的那些）、这些区域现在的面数 vs 按轮廓重新三角化需要的面数（压缩倍数）、二面角直方图、边界/流形/非流形边占比、平面区域内 DC 颜色标准差 / 全局 DC 标准差（≈ 1 说明平面上的颜色变化和别处一样多，即顶点在那里是为了颜色）、前 10 大区域。
**判据**（ε 0.25 % / θ 10°，≥ 100 面区域）：顶点占比 ≥ 40 % 且内部顶点 ≥ 30 % → 轴 4.1 + 4.2 至少值 2x 顶点；DC within/global ≈ 1 → 这些顶点确实是颜色驱动的。室内（room/kitchen/counter/bonsai）预期高，室外植被场景预期低 —— 两类的差本身就是 cards 那条轴的论据。
**成本**：λ 跑的 checkpoint（0.4–1.3 M 顶点）每个约 0.5–2 min，dense 原始（3 M 顶点、~7 M 面）每个 3–5 min、内存 4–6 GB；33 个 checkpoint 合计约 40 min CPU。在你当前拿着的节点上跑即可（8 GB 的 VS Code 小节点跑 dense 可能爆内存，λ 跑没问题）。

## §1 安装 + selftest（任意 shell，整块粘贴；不会退出你的终端）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN/diag $AN/planarity-v1
cat > $AN/diag/planarity_v1.py <<'PYEOF_PLANARITY'
"""planarity_v1.py (2026-10-30): CPU probe P1 of plan/meshsplat-plus-2026-10-29.md.

Question: how much of a trained MeshSplatting mesh is geometrically flat, i.e. how many vertices exist to carry
colour / triangulation rather than shape?  A plane + texture would need only the region's outline.

Method (numpy/scipy, no GPU):
  1. faces -> unit normals, areas, centroids; edges from sorted vertex pairs (non-manifold edges keep all their faces)
  2. two faces sharing an edge are "locally coplanar" iff |n_a . n_b| >= cos(theta) (orientation-free; the soup is not
     oriented) and each face's opposite vertex lies within eps of the other face's plane
  3. connected components of the "locally coplanar" face graph = candidate planar regions
  4. each region with >= min_faces faces gets an area-weighted PCA plane; it is VERIFIED planar iff the 95th percentile
     of its vertex distances to that plane is <= eps (this catches curved surfaces that chain through small dihedrals)
  5. statistics per (eps, theta): share of faces / area / vertices in verified planar regions (>= 10 / 100 / 1000
     faces), INTERIOR vertices (all incident faces in one verified region: the ones a texture would replace),
     boundary edges per region (= triangles a polygon would need), dihedral histogram, non-manifold / boundary edge
     shares, DC-colour spread inside planar regions vs the whole mesh (are the vertices there for colour?), top regions.
eps is given in % of the robust scene diagonal (2nd-98th percentile box of the vertices); the q14 grid step and the
median edge length are printed in the same units for reference.

Usage: python planarity_v1.py --state <point_cloud_state_dict.pt | shared-final.pt> --label "room v3 lam0.0062" --out x.json
       python planarity_v1.py --selftest
"""
import argparse
import itertools
import json
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

VERSION = 'planarity_v1 2026-10-30'
EPS_PCTS = (0.1, 0.25, 0.5)
THETAS = (5.0, 10.0)
MIN_FACES = 10


# ----------------------------------------------------------------------------- loading
def load_mesh(path):
    import torch
    sd = torch.load(path, map_location='cpu', weights_only=False)
    keys = list(sd.keys()) if isinstance(sd, dict) else []
    V = sd.get('triangles_points', sd.get('vertices'))
    F = sd.get('_triangle_indices', sd.get('triangle_indices', sd.get('faces')))
    if V is None or F is None:
        raise SystemExit(f'cannot find vertices/faces in {path}; keys = {keys}')
    V = np.asarray(V.detach().cpu().numpy() if hasattr(V, 'detach') else V, dtype=np.float64)
    F = np.asarray(F.detach().cpu().numpy() if hasattr(F, 'detach') else F, dtype=np.int64).reshape(-1, 3)
    DC = sd.get('features_dc')
    if DC is not None:
        DC = np.asarray(DC.detach().cpu().numpy() if hasattr(DC, 'detach') else DC, dtype=np.float64).reshape(len(V), -1)[:, :3]
    return V, F, DC, keys


# ----------------------------------------------------------------------------- geometry helpers
def face_geometry(V, F):
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    a2 = np.linalg.norm(n, axis=1)
    area = 0.5 * a2
    ok = a2 > 0
    nhat = np.zeros_like(n)
    nhat[ok] = n[ok] / a2[ok, None]
    c = (p0 + p1 + p2) / 3.0
    return nhat, area, c, ok


def robust_scale(V):
    lo = np.percentile(V, 2, axis=0)
    hi = np.percentile(V, 98, axis=0)
    return float(np.linalg.norm(hi - lo)), lo, hi


def edge_table(F):
    """All face-edge incidences: returns (edge_key sorted, face id, opposite vertex) sorted by key, and the group
    boundaries.  Edge key = min*N + max with N = max vertex id + 1."""
    N = int(F.max()) + 1
    a = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    b = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    opp = np.concatenate([F[:, 2], F[:, 0], F[:, 1]])
    fid = np.tile(np.arange(len(F), dtype=np.int64), 3)
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    key = lo * N + hi
    order = np.argsort(key, kind='stable')
    key, fid, opp = key[order], fid[order], opp[order]
    uniq, start, count = np.unique(key, return_index=True, return_counts=True)
    return key, fid, opp, start, count


def adjacent_pairs(fid, opp, start, count, max_all_pairs=6):
    """Face pairs sharing an edge.  Groups of k <= max_all_pairs faces give all k(k-1)/2 pairs, larger groups a chain."""
    pa, pb, oa, ob = [], [], [], []
    for k in np.unique(count):
        k = int(k)
        if k < 2:
            continue
        g = start[count == k]                       # group starts
        idx = g[:, None] + np.arange(k)[None, :]    # (G, k) incidence rows
        if k <= max_all_pairs:
            combos = np.array(list(itertools.combinations(range(k), 2)), dtype=np.int64)
            i, j = idx[:, combos[:, 0]].reshape(-1), idx[:, combos[:, 1]].reshape(-1)
        else:
            i, j = idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)
        pa.append(fid[i]); pb.append(fid[j]); oa.append(opp[i]); ob.append(opp[j])
    if not pa:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, z
    return np.concatenate(pa), np.concatenate(pb), np.concatenate(oa), np.concatenate(ob)


def group_percentile(values, labels, n_groups, q=0.95):
    """q-th percentile of values within each label group (vectorised: sort by (label, value))."""
    order = np.lexsort((values, labels))
    lab_s, val_s = labels[order], values[order]
    cnt = np.bincount(lab_s, minlength=n_groups)
    first = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    pos = first + np.clip(np.floor(q * (cnt - 1)).astype(np.int64), 0, None)
    out = np.full(n_groups, np.inf)
    has = cnt > 0
    out[has] = val_s[pos[has]]
    return out


# ----------------------------------------------------------------------------- the probe
def analyse(V, F, DC, eps, cos_theta, min_faces=MIN_FACES, precomputed=None):
    """One (eps, theta) pass.  Returns stats dict and per-face region labels (-1 = not in a verified planar region)."""
    N, M = len(V), len(F)
    nhat, area, cen, ok = precomputed['geom']
    key, fid, opp, start, count = precomputed['edges']
    pa, pb, oa, ob = precomputed['pairs']
    # local coplanarity test on every adjacent pair
    dot = np.abs((nhat[pa] * nhat[pb]).sum(1))
    da = np.abs(((V[ob] - cen[pa]) * nhat[pa]).sum(1))   # opposite vertex of b to plane a
    db = np.abs(((V[oa] - cen[pb]) * nhat[pb]).sum(1))
    good = ok[pa] & ok[pb] & (dot >= cos_theta) & (da <= eps) & (db <= eps)
    g = coo_matrix((np.ones(int(good.sum()), dtype=np.int8), (pa[good], pb[good])), shape=(M, M))
    n_comp, lab = connected_components(g, directed=False)
    comp_faces = np.bincount(lab, minlength=n_comp)
    comp_area = np.bincount(lab, weights=area, minlength=n_comp)
    big = comp_faces >= min_faces
    # area-weighted PCA plane per component (centroids), then vertex-distance verification
    w = area
    mu = np.stack([np.bincount(lab, weights=w * cen[:, i], minlength=n_comp) for i in range(3)], 1) / np.maximum(comp_area, 1e-30)[:, None]
    d = cen - mu[lab]
    cov = np.zeros((n_comp, 3, 3))
    for i in range(3):
        for j in range(i, 3):
            cij = np.bincount(lab, weights=w * d[:, i] * d[:, j], minlength=n_comp) / np.maximum(comp_area, 1e-30)
            cov[:, i, j] = cij
            cov[:, j, i] = cij
    bi = np.nonzero(big)[0]
    normal = np.zeros((n_comp, 3))
    if len(bi):
        evals, evecs = np.linalg.eigh(cov[bi])
        normal[bi] = evecs[:, :, 0]                       # smallest-variance direction
    # vertex distances to the component plane (3 incidences per face)
    inc_lab = np.repeat(lab, 3)
    inc_v = F.reshape(-1)
    dist = np.abs(((V[inc_v] - mu[inc_lab]) * normal[inc_lab]).sum(1))
    sel = big[inc_lab]
    p95 = np.full(n_comp, np.inf)
    mx = np.full(n_comp, np.inf)
    if sel.any():
        p95_b = group_percentile(dist[sel], inc_lab[sel], n_comp, 0.95)
        mx_b = np.full(n_comp, -1.0)
        np.maximum.at(mx_b, inc_lab[sel], dist[sel])
        p95[big] = p95_b[big]
        mx[big] = mx_b[big]
    verified = big & (p95 <= eps)
    region = np.where(verified[lab], lab, -1)
    # ---- statistics
    tot_area = float(area.sum())
    deg = np.bincount(F.reshape(-1), minlength=N)
    stats = dict(eps=float(eps), cos_theta=float(cos_theta), components=int(n_comp), big_components=int(big.sum()), verified_components=int(verified.sum()),
                 curved_rejected_components=int((big & ~verified).sum()), curved_rejected_faces=int(comp_faces[big & ~verified].sum()))
    # vertex -> region incidences (once): a vertex is INTERIOR to a region when every incident face lies in that region
    pairs = np.unique(np.stack([inc_v, inc_lab], 1), axis=0)
    cnt_vc = np.bincount(pairs[:, 0], minlength=N)                # number of distinct regions a vertex touches
    one_region = cnt_vc == 1
    reg_of_v = np.full(N, -1, dtype=np.int64)
    m1 = one_region[pairs[:, 0]]
    reg_of_v[pairs[m1, 0]] = pairs[m1, 1]
    # boundary edges per region (once): an (edge, region) incidence with fewer faces of that region than the edge has faces
    edge_id = np.repeat(np.arange(len(start)), count)             # per sorted incidence
    uniq_er, c_er = np.unique(np.stack([edge_id, lab[fid]], 1), axis=0, return_counts=True)
    e_ids, r_ids = uniq_er[:, 0], uniq_er[:, 1]
    is_boundary = (c_er < count[e_ids]) | (count[e_ids] == 1)      # region does not own the whole edge, or open mesh boundary
    bnd_per_region = np.bincount(r_ids[is_boundary], minlength=n_comp)
    for k in (10, 100, 1000):
        vk = verified & (comp_faces >= k)
        in_k = vk[lab]
        verts_k = np.zeros(N, bool)
        verts_k[F[in_k].reshape(-1)] = True
        interior = one_region & (reg_of_v >= 0) & vk[np.clip(reg_of_v, 0, None)]
        faces_now = int(comp_faces[vk].sum())
        faces_polygon = int(np.maximum(bnd_per_region[vk] - 2, 1).sum())
        stats[f'ge{k}'] = dict(regions=int(vk.sum()), face_share=float(comp_faces[vk].sum() / max(M, 1)), area_share=float(comp_area[vk].sum() / max(tot_area, 1e-30)),
                               vertex_share=float(verts_k.mean()), interior_vertex_share=float(interior.mean()),
                               faces_now=faces_now, faces_if_polygons=faces_polygon,
                               face_reduction_x=(faces_now / max(faces_polygon, 1)))
    # top regions
    order = np.argsort(-comp_area * verified)[:10]
    top = []
    for c in order:
        if not verified[c]:
            break
        top.append(dict(faces=int(comp_faces[c]), area_share=float(comp_area[c] / tot_area), normal=[round(float(x), 3) for x in normal[c]],
                        p95_over_eps=float(p95[c] / eps), max_over_eps=float(mx[c] / eps)))
    stats['top_regions'] = top
    # colour spread inside verified planar regions (>= 100 faces) vs the whole mesh, in DC units
    if DC is not None:
        vk = verified & (comp_faces >= 100)
        inc_sel = vk[inc_lab]
        if inc_sel.any():
            labs = inc_lab[inc_sel]
            vv = inc_v[inc_sel]
            cnt = np.bincount(labs, minlength=n_comp).astype(np.float64)
            s_within = []
            for ch in range(3):
                x = DC[vv, ch]
                m1 = np.bincount(labs, weights=x, minlength=n_comp) / np.maximum(cnt, 1)
                m2 = np.bincount(labs, weights=x * x, minlength=n_comp) / np.maximum(cnt, 1)
                var = np.clip(m2 - m1 * m1, 0, None)
                s_within.append(float(np.sqrt((var * cnt).sum() / cnt.sum())))   # incidence-weighted within-region std
            stats['dc_std_within_planar'] = float(np.mean(s_within))
            stats['dc_std_global'] = float(DC[:, :3].std(0).mean())
            stats['dc_within_over_global'] = stats['dc_std_within_planar'] / max(stats['dc_std_global'], 1e-12)
    return stats, region


def precompute(V, F):
    nhat, area, cen, ok = face_geometry(V, F)
    key, fid, opp, start, count = edge_table(F)
    pa, pb, oa, ob = adjacent_pairs(fid, opp, start, count)
    pre = dict(geom=(nhat, area, cen, ok), edges=(key, fid, opp, start, count), pairs=(pa, pb, oa, ob))
    # mesh-level facts
    N, M = len(V), len(F)
    dot = np.abs((nhat[pa] * nhat[pb]).sum(1)) if len(pa) else np.zeros(0)
    ang = np.degrees(np.arccos(np.clip(dot, -1, 1)))
    hist = {f'<{t}deg': float((ang < t).mean()) if len(ang) else 0.0 for t in (2, 5, 10, 30)}
    e_len = np.linalg.norm(V[key // (int(F.max()) + 1)] - V[key % (int(F.max()) + 1)], axis=1)
    facts = dict(vertices=N, faces=M, degenerate_faces=int((~ok).sum()), edges=int(len(start)),
                 boundary_edge_share=float((count == 1).mean()), manifold_edge_share=float((count == 2).mean()), nonmanifold_edge_share=float((count >= 3).mean()),
                 max_faces_per_edge=int(count.max()), adjacent_pairs=int(len(pa)), dihedral_hist=hist, median_edge_length=float(np.median(e_len)))
    return pre, facts


def run(V, F, DC, label, eps_pcts=EPS_PCTS, thetas=THETAS, min_faces=MIN_FACES):
    t0 = time.time()
    scale, lo, hi = robust_scale(V)
    full_span = (V.max(0) - V.min(0)).max()
    pre, facts = precompute(V, F)
    facts.update(label=label, version=VERSION, scene_diagonal=scale, q14_step=float(full_span / (2 ** 14 - 1)), q14_step_pct=float(100 * full_span / (2 ** 14 - 1) / scale),
                 median_edge_length_pct=float(100 * facts['median_edge_length'] / scale), min_faces=min_faces)
    rows = []
    for ep in eps_pcts:
        for th in thetas:
            eps = ep / 100.0 * scale
            st, _ = analyse(V, F, DC, eps, np.cos(np.radians(th)), min_faces, pre)
            st.update(eps_pct=ep, theta_deg=th)
            rows.append(st)
            g = st['ge100']
            print(f"  eps {ep:.2f} % theta {th:>4.0f} | verified regions {st['verified_components']} (curved rejected {st['curved_rejected_components']}) "
                  f"| >=100 faces: faces {100*g['face_share']:.1f} % area {100*g['area_share']:.1f} % vertices {100*g['vertex_share']:.1f} % interior {100*g['interior_vertex_share']:.1f} % "
                  f"| faces {g['faces_now']} -> polygons {g['faces_if_polygons']} ({g['face_reduction_x']:.1f}x)"
                  + (f" | DC std within/global {st['dc_within_over_global']:.2f}" if 'dc_within_over_global' in st else ''), flush=True)
    facts['elapsed_s'] = round(time.time() - t0, 1)
    return dict(facts=facts, rows=rows)


def summary_line(res):
    f = res['facts']
    r = next(x for x in res['rows'] if x['eps_pct'] == 0.25 and x['theta_deg'] == 10.0)
    g = r['ge100']
    top = ', '.join(f"{100*t['area_share']:.1f}%" for t in r['top_regions'][:5])
    dc = f" | DC std within/global {r['dc_within_over_global']:.2f}" if 'dc_within_over_global' in r else ''
    return (f"PLANARITY {f['label']} | V {f['vertices']} F {f['faces']} | edges boundary/manifold/nonmanifold {100*f['boundary_edge_share']:.0f}/{100*f['manifold_edge_share']:.0f}/{100*f['nonmanifold_edge_share']:.0f} % "
            f"| dihedral <5deg {100*f['dihedral_hist']['<5deg']:.0f} % | median edge {f['median_edge_length_pct']:.3f} % of diag, q14 step {f['q14_step_pct']:.4f} % "
            f"| eps 0.25 % theta 10: regions>=100 faces {g['regions']} = faces {100*g['face_share']:.1f} % area {100*g['area_share']:.1f} % vertices {100*g['vertex_share']:.1f} % interior {100*g['interior_vertex_share']:.1f} % "
            f"| faces {g['faces_now']} -> polygons {g['faces_if_polygons']} ({g['face_reduction_x']:.1f}x) | top-5 area {top}{dc} | {f['elapsed_s']} s")


# ----------------------------------------------------------------------------- selftest
def _grid_plane(nx, ny, origin, ex, ey, noise, rng):
    xs, ys = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing='ij')
    P = origin[None, :] + xs.reshape(-1, 1) * ex[None, :] + ys.reshape(-1, 1) * ey[None, :]
    nrm = np.cross(ex, ey); nrm /= np.linalg.norm(nrm)
    P = P + nrm[None, :] * rng.normal(0, noise, (len(P), 1))
    def vid(i, j): return i * (ny + 1) + j
    F = []
    for i in range(nx):
        for j in range(ny):
            F.append([vid(i, j), vid(i + 1, j), vid(i, j + 1)]); F.append([vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])
    return P, np.array(F, dtype=np.int64)


def _sphere(nu, nv, centre, r):
    u = np.linspace(0, np.pi, nu + 1); v = np.linspace(0, 2 * np.pi, nv, endpoint=False)
    P = []; 
    for a in u:
        for b in v:
            P.append(centre + r * np.array([np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)]))
    P = np.array(P)
    def vid(i, j): return i * nv + (j % nv)
    F = []
    for i in range(nu):
        for j in range(nv):
            F.append([vid(i, j), vid(i + 1, j), vid(i, j + 1)]); F.append([vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)])
    return P, np.array(F, dtype=np.int64)


def selftest():
    rng = np.random.default_rng(0)
    parts = []
    # a 10 x 10 x 10 box: 6 planar faces, each a 40x40 grid (3200 triangles), noise 1e-4 of the size
    s = 10.0
    axes = [(np.array([0, 0, 0.]), np.array([s / 40, 0, 0]), np.array([0, s / 40, 0])), (np.array([0, 0, s]), np.array([s / 40, 0, 0]), np.array([0, s / 40, 0])),
            (np.array([0, 0, 0.]), np.array([s / 40, 0, 0]), np.array([0, 0, s / 40])), (np.array([0, s, 0]), np.array([s / 40, 0, 0]), np.array([0, 0, s / 40])),
            (np.array([0, 0, 0.]), np.array([0, s / 40, 0]), np.array([0, 0, s / 40])), (np.array([s, 0, 0]), np.array([0, s / 40, 0]), np.array([0, 0, s / 40]))]
    for o, ex, ey in axes:
        parts.append(_grid_plane(40, 40, o, ex, ey, 1e-4 * s, rng))
    parts.append(_sphere(60, 120, np.array([5., 5., 5.]), 3.0))                  # curved: must be rejected at tight eps
    soupP = rng.random((3000, 3)) * s; soupF = rng.integers(0, 3000, (6000, 3))   # random soup: no planar regions
    soupF = soupF[(soupF[:, 0] != soupF[:, 1]) & (soupF[:, 1] != soupF[:, 2]) & (soupF[:, 0] != soupF[:, 2])]
    parts.append((soupP, soupF))
    V, F, off = [], [], 0
    for P, Fi in parts:
        V.append(P); F.append(Fi + off); off += len(P)
    V, F = np.concatenate(V), np.concatenate(F)
    n_box_faces = 6 * 3200; n_sphere_faces = 2 * 60 * 120
    DC = rng.normal(0, 1, (len(V), 3))
    res = run(V, F, DC, 'selftest', eps_pcts=(0.1, 0.25), thetas=(5.0, 10.0))
    r = next(x for x in res['rows'] if x['eps_pct'] == 0.25 and x['theta_deg'] == 10.0)
    g = r['ge1000']
    assert g['regions'] == 6, f"expected the 6 box faces as regions >= 1000 faces, got {g['regions']}"
    assert abs(g['faces_now'] - n_box_faces) <= 6 * 80, (g['faces_now'], n_box_faces)      # edge rows may be cut by the dihedral at the box edges
    assert r['curved_rejected_faces'] >= 0.8 * n_sphere_faces or r['verified_components'] == 6, 'sphere must not be verified planar'
    assert all(t['p95_over_eps'] <= 1.0 for t in r['top_regions'])
    assert 0.1 < g['interior_vertex_share'] < 0.9 and 10 < g['face_reduction_x'] < 40, g   # 3200 faces vs ~158 outline triangles per box face
    print('top regions:', [(t['faces'], round(t['area_share'], 3), t['normal']) for t in r['top_regions'][:6]])
    print('selftest PASS', VERSION)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state'); ap.add_argument('--label', default=''); ap.add_argument('--out', default=None)
    ap.add_argument('--eps', type=float, nargs='*', default=list(EPS_PCTS), help='eps in %% of the robust scene diagonal')
    ap.add_argument('--theta', type=float, nargs='*', default=list(THETAS), help='dihedral tolerance in degrees')
    ap.add_argument('--min_faces', type=int, default=MIN_FACES)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not a.state:
        ap.error('--state or --selftest')
    V, F, DC, keys = load_mesh(a.state)
    label = a.label or a.state
    print(f'[{VERSION}] {label}: V {len(V)} F {len(F)} DC {"yes" if DC is not None else "no"} | keys {keys[:6]}', flush=True)
    res = run(V, F, DC, label, tuple(a.eps), tuple(a.theta), a.min_faces)
    res['state'] = a.state
    line = summary_line(res)
    print(line, flush=True)
    if a.out:
        with open(a.out, 'w') as fh:
            json.dump(res, fh, indent=1)


if __name__ == '__main__':
    main()
PYEOF_PLANARITY
python3 -c "import ast; ast.parse(open('$AN/diag/planarity_v1.py').read()); print('compiled OK')" || echo "FAILED: compile"
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh; conda activate /scratch/zr2245/conda-envs/pmsh-cu126; source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
python $AN/diag/planarity_v1.py --selftest 2>&1 | tail -3 || echo "FAILED: selftest"
```

## §2 跑全部 checkpoint（整块粘贴；写脚本后用 bash 运行，脚本内部逐个 checkpoint 容错）

```bash
AN=/scratch/zr2245/analysis; mkdir -p $AN
cat > $AN/run_planarity_v1.sh <<'SHEOF_PLANRUN'
#!/usr/bin/env bash
# run_planarity_v1.sh (2026-10-30): CPU planarity probe on every existing checkpoint (lambda runs first, dense originals last).
# Restartable: a checkpoint whose json exists is skipped.  Rows -> $AN/planarity-v1.txt, json + log -> $AN/planarity-v1/.
source /scratch/zr2245/anaconda3/etc/profile.d/conda.sh
conda activate /scratch/zr2245/conda-envs/pmsh-cu126
source /scratch/zr2245/meshsplatting-original-lossless-v1/.venv/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
AN=/scratch/zr2245/analysis; OUT=$AN/planarity-v1; TAB=$AN/planarity-v1.txt; PY=$AN/diag/planarity_v1.py; mkdir -p $OUT
SCENES=${SCENES:-"room kitchen counter bonsai garden stump bicycle flowers treehill truck train"}
n_ok=0; n_fail=0; n_skip=0
run_one () {  # scene tag checkpoint
  local S=$1 T=$2 CK=$3 J=$OUT/$S-$T.json L=$OUT/$S-$T.log
  if [ ! -f $CK ]; then echo "missing: $S $T ($CK)"; n_skip=$((n_skip+1)); return; fi
  if [ -s $J ]; then echo "exists: $S $T"; n_skip=$((n_skip+1)); return; fi
  echo "=== $S $T ($(date +%H:%M:%S)) ==="
  if python $PY --state $CK --label "$S $T" --out $J > $L 2>&1; then
    grep -h "^PLANARITY" $L | tee -a $TAB; n_ok=$((n_ok+1))
  else
    echo "FAILED: $S $T"; tail -5 $L; n_fail=$((n_fail+1))
  fi
}
for S in $SCENES; do
  run_one $S v3-lam0.0062  /scratch/zr2245/$S-rd3-lam0.0062-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
  run_one $S v3-lam0.00185 /scratch/zr2245/$S-rd3-lam0.00185-mu1.0/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
done
for S in $SCENES; do
  run_one $S dense /scratch/zr2245/$S-vis-collapse/original/point_cloud/iteration_30000/point_cloud_state_dict.pt
done
echo "===== $TAB ====="; cat $TAB
echo "run_planarity_v1 done: ok $n_ok failed $n_fail skipped $n_skip  $(date)"
SHEOF_PLANRUN
bash $AN/run_planarity_v1.sh 2>&1 | tee $AN/planarity-v1-run.log
```

只跑部分场景：`SCENES="room garden" bash $AN/run_planarity_v1.sh`。单个 checkpoint 手动跑：`python $AN/diag/planarity_v1.py --state <point_cloud_state_dict.pt> --label "room v3 lam0.0062" --out x.json`（所有 (ε, θ) 组合各打印一行，最后一行 `PLANARITY ...` 是汇总）。

## §3 看结果

```bash
AN=/scratch/zr2245/analysis; cat $AN/planarity-v1.txt; tail -2 $AN/planarity-v1-run.log
```
把 `planarity-v1.txt` 整个贴回来（每个 checkpoint 一行）。想看某个 checkpoint 的全部 (ε, θ) 组合和前 10 大区域：`grep -h "eps\|top" $AN/planarity-v1/room-v3-lam0.0062.log`。

## 读法备注
- "内部顶点占比"是最硬的数：这些顶点的所有相邻面都在同一个验证平面里，换成平面 + 纹理后它们一个都不需要。
- "faces → polygons" 是上界估计：按区域轮廓边数重新三角化；soup 的锯齿轮廓会让它偏大，所以真实压缩倍数只会更高。
- 非流形边（一条边挂 ≥ 3 面）不会破坏统计：邻接对按边上所有面两两生成（≤ 6 面全配对，更多时链式）。
- ε 相对场景尺度定义；q14 步长（我们交付的量化精度）和中位边长同一单位打印出来，方便判断 ε 是否"肉眼不可见"。
