"""F2c: one rule for resolution, h(x) = k * distance(x, viewing region).
topology: coarse free-space carving (uniform voxels) -> closed boundary.
accuracy: adaptive red-green edge refinement where edge > k * d(nearest observer), vertices snapped along an observer's ray
to what that observer sees. Edge splits keep the mesh closed and keep its topology."""
import numpy as np, sys, time, json
sys.path.insert(0, '.')
from f2lib import *
import s1
from s1 import inside_views, eval_views
from s3 import boundary_mesh
from s3b import pano_depth
from scipy.spatial import cKDTree
import trimesh

def observer_positions(S, c, region_r, n_pos, seed=7):
    rng = np.random.default_rng(seed); s1.rng = np.random.default_rng(seed)
    P = [c.copy()]
    for i in range(1, n_pos):
        rr = region_r * np.sqrt(rng.uniform())
        P.append(inside_views(S, c, rr, 1)[0][0])
    return np.array(P)

def carve_vox_pos(S, positions, vox):
    lo = S.mesh.bounds[0] - 2 * vox; hi = S.mesh.bounds[1] + 2 * vox; hi[1] += 1.0
    shape = np.ceil((hi - lo) / vox).astype(int)
    idx = np.indices(shape).reshape(3, -1).T
    P = (lo + (idx + 0.5) * vox).astype(np.float32)
    free = np.zeros(len(P), bool)
    for pos in positions:
        Dm = pano_depth(S, pos); H, W = Dm.shape
        for a in range(0, len(P), 4_000_000):
            q = P[a:a + 4_000_000] - pos.astype(np.float32)
            dist = np.linalg.norm(q, axis=1)
            th = np.arctan2(q[:, 0], q[:, 2]); ph = np.arcsin(np.clip(q[:, 1] / np.maximum(dist, 1e-6), -1, 1))
            u = ((th + np.pi) / (2 * np.pi) * W).astype(int) % W; v = np.clip(((ph + np.pi / 2) / np.pi * H).astype(int), 0, H - 1)
            free[a:a + 4_000_000] |= dist < Dm[v, u] - 0.6 * vox
    free = free.reshape(shape)
    # Omega = free component containing the first observer (definition)
    from scipy.ndimage import label
    lab, _ = label(free); i0 = tuple(np.floor((positions[0] - lo) / vox).astype(int))
    if lab[i0] > 0: free = lab == lab[i0]
    return free, lo

def refine(V, F, mark_edge):
    """red-green refinement: split every marked edge; faces get 2/3/4 children; closed in -> closed out."""
    E = np.sort(np.stack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1).reshape(-1, 2), axis=1)
    Eu, inv = np.unique(E, axis=0, return_inverse=True); fe = inv.reshape(-1, 3)
    mk = mark_edge(V, Eu)
    if not mk.any(): return V, F, 0
    mid = np.full(len(Eu), -1); nm = int(mk.sum())
    mid[mk] = len(V) + np.arange(nm)
    V = np.concatenate([V, 0.5 * (V[Eu[mk, 0]] + V[Eu[mk, 1]])])
    m = mk[fe]; cnt = m.sum(1)
    out = [F[cnt == 0]]
    for r in range(3):  # rotate faces so the pattern is canonical
        Fr = np.roll(F, -r, axis=1); er = np.roll(fe, -r, axis=1); mr = np.roll(m, -r, axis=1)
        a, b, c = Fr[:, 0], Fr[:, 1], Fr[:, 2]; m0, m1, m2 = mid[er[:, 0]], mid[er[:, 1]], mid[er[:, 2]]
        s1_ = (cnt == 1) & mr[:, 0]                                   # only edge ab
        out += [np.stack([a, m0, c], 1)[s1_], np.stack([m0, b, c], 1)[s1_]]
        s2 = (cnt == 2) & mr[:, 0] & mr[:, 2]                         # ab and ca
        out += [np.stack([a, m0, m2], 1)[s2], np.stack([m0, b, c], 1)[s2], np.stack([m0, c, m2], 1)[s2]]
        if r == 0:
            s3 = cnt == 3
            out += [np.stack([a, m0, m2], 1)[s3], np.stack([m0, b, m1], 1)[s3], np.stack([m2, m1, c], 1)[s3], np.stack([m0, m1, m2], 1)[s3]]
    return V, np.concatenate(out), nm

def snap(S, V, idx, positions, tree, K=16, bound_abs=0.24, bound_rel=0.05):
    """move V[idx] to the surface point seen along the ray from the nearest observer that sees it."""
    v = V[idx]; dists, nn = tree.query(v, k=K)
    done = np.zeros(len(idx), bool); newv = v.copy()
    for j in range(K):
        sel = np.where(~done)[0]
        if len(sel) == 0: break
        o = positions[nn[sel, j]]; d = v[sel] - o; L = np.linalg.norm(d, axis=1); d /= np.maximum(L, 1e-9)[:, None]
        loc, ir, _ = S.rt.intersects_location(o, d, multiple_hits=False)
        t = np.full(len(sel), np.inf); t[ir] = np.einsum('ij,ij->i', loc - o[ir], d[ir])
        ok = np.abs(t - L) < np.maximum(bound_abs, bound_rel * L)
        newv[sel[ok]] = o[ok] + d[ok] * t[ok, None]; done[sel[ok]] = True
    V[idx] = newv
    return done.mean()

def build(S, c, region, vox=0.08, k=0.012, n_pos=160, rounds=5, vmax=3_500_000, log=print):
    t0 = time.time()
    pos = observer_positions(S, c, region, n_pos); tree = cKDTree(pos)
    free, lo = carve_vox_pos(S, pos, vox)
    m = boundary_mesh(free, lo, vox)
    V, F = m.vertices.copy(), m.faces.copy()
    fr = snap(S, V, np.arange(len(V)), pos, tree)
    log(f'  coarse dOmega V {len(V)} F {len(F)} snapped {fr*100:.1f}%  ({time.time()-t0:.0f}s)')
    for rd in range(rounds):
        def mark(V, Eu):
            mid = 0.5 * (V[Eu[:, 0]] + V[Eu[:, 1]]); dd, _ = tree.query(mid)
            return np.linalg.norm(V[Eu[:, 0]] - V[Eu[:, 1]], axis=1) > k * np.maximum(dd, 0.1)
        n0 = len(V)
        if n0 * 1.8 >= vmax: break     # next round roughly doubles the mesh; stay inside the 15 GB container
        V, F, nm = refine(V, F, mark)
        fr = snap(S, V, np.arange(n0, len(V)), pos, tree) if nm else 1.0
        log(f'  round {rd}: +{nm} vertices -> V {len(V)} F {len(F)}, new snapped {fr*100:.1f}%  ({time.time()-t0:.0f}s)')
        if nm == 0: break
    return trimesh.Trimesh(V, F, process=False), pos

def colour_and_eval(S, m, c, TV, ytop=None):
    n = m.vertex_normals
    tt, rgb, _ = S.cast(m.vertices + n * 0.02, -n); tt2, rgb2, _ = S.cast(m.vertices - n * 0.02, n)
    C = np.where((tt <= tt2)[:, None], rgb, rgb2)
    X = m.vertices.copy()
    if ytop is not None:
        sky = X[:, 1] > ytop; C[sky] = SKY
        X[sky] = c + (X[sky] - c) / np.linalg.norm(X[sky] - c, axis=1, keepdims=True) * 1e4
    M = SphereMap(X, m.faces, C)
    return {r: eval_views(S, M, TV[r]) for r in TV}

def topo(m):
    comps = m.split(only_watertight=False); big = sorted(comps, key=lambda x: -len(x.faces))
    E = np.sort(big[0].edges, axis=1); _, cnt = np.unique(E, axis=0, return_counts=True)
    return dict(comps=len(comps), main_faces=len(big[0].faces), genus=(2 - big[0].euler_number) / 2,
                boundary_edges=int((cnt == 1).sum()), nonmanifold_edges=int((cnt > 2).sum()))

if __name__ == '__main__':
    region = float(sys.argv[1]); k = float(sys.argv[2]) if len(sys.argv) > 2 else 0.012
    S = GTScene('../gt/Sponza.gltf'); c = np.array([0, 1.6, 0.])
    s1.rng = np.random.default_rng(123)
    TV = {r: inside_views(S, c, r, 12) for r in [0, 0.5, 1, 2, 4]}
    print(f'== F2c region {region} m, k {k}', flush=True)
    m, pos = build(S, c, region, k=k, log=lambda s: print(s, flush=True))
    tp = topo(m); print('  topology', tp, flush=True)
    res = colour_and_eval(S, m, c, TV, ytop=S.mesh.bounds[1][1])
    for r, e in res.items():
        print(f'  test r={r}: psnr {e["psnr"]:.2f} bad-depth {e["bad"]*100:.1f}% holes {e["hole"]*100:.2f}%', flush=True)
    json.dump(dict(region=region, k=k, V=len(m.vertices), F=len(m.faces), topo=tp, tests={str(r): e for r, e in res.items()}),
              open(f's5b_region{region}_k{k}.json', 'w'), indent=1)
