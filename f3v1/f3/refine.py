"""Snap vertices along a viewing ray to the observed depth; red-green edge refinement with h = k * dist(viewer)."""
import numpy as np
from scipy.spatial import cKDTree

def snap(V, idx, views, depths, tree, K=64, bound_abs=0.16, bound_rel=0.05):
    """V[idx] -> point observed along the ray from the nearest view (of K) whose depth agrees within the bound."""
    v = V[idx]; _, nn = tree.query(v, k=min(K, len(views)))
    nn = np.atleast_2d(nn); done = np.zeros(len(idx), bool); out = v.copy()
    for j in range(nn.shape[1]):
        for vi in np.unique(nn[~done, j]):
            sel = np.flatnonzero((~done) & (nn[:, j] == vi))
            view, D = views[vi], depths[vi]
            u, vv, z = view.project(v[sel])
            ins = (z > 0.05) & (u >= 0) & (u < view.W) & (vv >= 0) & (vv < view.H)
            s2 = sel[ins]
            d = D[vv[ins].astype(int), u[ins].astype(int)]
            ok = (d > 0) & (np.abs(d - z[ins]) < np.maximum(bound_abs, bound_rel * z[ins]))
            s3 = s2[ok]
            out[s3] = view.center + (v[s3] - view.center) * (d[ok] / z[ins][ok])[:, None]
            done[s3] = True
    V[idx] = out
    return done.mean()

def refine(V, F, mark_edge):
    """red-green refinement; closed in -> closed out, topology unchanged."""
    E = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
    Eu, inv = np.unique(E, axis=0, return_inverse=True); fe = inv.reshape(-1, 3)
    mk = mark_edge(V, Eu)
    if not mk.any(): return V, F, 0
    mid = np.full(len(Eu), -1); nm = int(mk.sum()); mid[mk] = len(V) + np.arange(nm)
    V = np.concatenate([V, 0.5 * (V[Eu[mk, 0]] + V[Eu[mk, 1]])])
    m = mk[fe]; cnt = m.sum(1); out = [F[cnt == 0]]
    for r in range(3):
        Fr = np.roll(F, -r, axis=1); er = np.roll(fe, -r, axis=1); mr = np.roll(m, -r, axis=1)
        a, b, c = Fr[:, 0], Fr[:, 1], Fr[:, 2]; m0, m1, m2 = mid[er[:, 0]], mid[er[:, 1]], mid[er[:, 2]]
        s1 = (cnt == 1) & mr[:, 0]
        out += [np.stack([a, m0, c], 1)[s1], np.stack([m0, b, c], 1)[s1]]
        s2 = (cnt == 2) & mr[:, 0] & mr[:, 2]
        out += [np.stack([a, m0, m2], 1)[s2], np.stack([m0, b, c], 1)[s2], np.stack([m0, c, m2], 1)[s2]]
        if r == 0:
            s3 = cnt == 3
            out += [np.stack([a, m0, m2], 1)[s3], np.stack([m0, b, m1], 1)[s3], np.stack([m2, m1, c], 1)[s3], np.stack([m0, m1, m2], 1)[s3]]
    return V, np.concatenate(out), nm

def build_surface(V, F, views, depths, k=0.012, vmax=3_000_000, rounds=6, frozen=None, log=print):
    """snap the coarse closed surface, then refine edges longer than k * (distance to the nearest training camera)."""
    centers = np.array([v.center for v in views]); tree = cKDTree(centers)
    movable = np.arange(len(V)) if frozen is None else np.flatnonzero(~frozen)
    fr = snap(V, movable, views, depths, tree); log(f'  snap coarse: {fr * 100:.1f}% of movable vertices')
    frozen = np.zeros(len(V), bool) if frozen is None else frozen.copy()
    for rd in range(rounds):
        if len(V) * 1.8 > vmax: break
        def mark(V, Eu):
            mid = 0.5 * (V[Eu[:, 0]] + V[Eu[:, 1]]); dd, _ = tree.query(mid)
            far = frozen[Eu[:, 0]] & frozen[Eu[:, 1]]
            return (np.linalg.norm(V[Eu[:, 0]] - V[Eu[:, 1]], axis=1) > k * np.maximum(dd, 0.1)) & ~far
        n0 = len(V); V, F, nm = refine(V, F, mark)
        frozen = np.concatenate([frozen, np.zeros(len(V) - n0, bool)])
        fr = snap(V, np.arange(n0, len(V)), views, depths, tree) if nm else 1.0
        log(f'  refine round {rd}: +{nm} -> V {len(V)} F {len(F)}  new snapped {fr * 100:.1f}%')
        if nm == 0: break
    return V, F, frozen

def topology(F, nV):
    E = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
    _, cnt = np.unique(E, axis=0, return_counts=True)
    chi = nV - len(cnt) + len(F)
    return dict(V=int(nV), F=int(len(F)), euler=int(chi), boundary_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()))

def init_colors(V, views, depths, images, K=8, rel=0.03, sky_color=None, lid=None):
    """vertex colour = pixel colour in the nearest view (of K) that sees the vertex (depth agrees within rel)."""
    centers = np.array([v.center for v in views]); tree = cKDTree(centers)
    _, nn = tree.query(V, k=min(K, len(views))); nn = np.atleast_2d(nn)
    C = np.full((len(V), 3), 0.5); done = np.zeros(len(V), bool)
    for j in range(nn.shape[1]):
        for vi in np.unique(nn[~done, j]):
            sel = np.flatnonzero((~done) & (nn[:, j] == vi)); view, D = views[vi], depths[vi]
            u, vv, z = view.project(V[sel])
            ins = (z > 0.05) & (u >= 0) & (u < view.W) & (vv >= 0) & (vv < view.H)
            s2 = sel[ins]; ui, vi_ = u[ins].astype(int), vv[ins].astype(int)
            d = D[vi_, ui]; ok = (d > 0) & (np.abs(d - z[ins]) < rel * z[ins])
            C[s2[ok]] = images[vi][vi_[ok], ui[ok]] / 255.0; done[s2[ok]] = True
    if lid is not None and sky_color is not None: C[lid] = sky_color
    return C, done.mean()
